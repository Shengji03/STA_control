"""
TaskRunner: Unified main loop connecting perception, LLM planning, and control execution.
"""

import numpy as np
import mujoco
import mujoco.viewer
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..llm_planner.dialog_planner import DialogPlanner

from ..perception.perception_module import PerceptionModule
from ..controller.sta_controller.sta_controller import STAController
from ..skills.skill_executor import SkillExecutor, SkillContext
from ..skills.skills import MoveSkill
from ..skills.base_skill import SkillType, SKILL_STA_PARAMS
from ..config.robot import DOF, INITIAL_JOINTS, MOBILE_ROBOT
from .arm_state import ArmState
from .navigation import NavigationTrajectory
from .plan_executor import PlanExecutor


# Compatibility aliases backed by shared Scene5 settings.
ARM_L_OFFSET = np.array(MOBILE_ROBOT.left_offset)
ARM_R_OFFSET = np.array(MOBILE_ROBOT.right_offset)
ARM_L_YAW = MOBILE_ROBOT.left_yaw
ARM_R_YAW = MOBILE_ROBOT.right_yaw
INIT_Q = np.array(INITIAL_JOINTS)
SENSOR_L_OFFSET = MOBILE_ROBOT.left_sensor_offset
SENSOR_R_OFFSET = MOBILE_ROBOT.right_sensor_offset


class TaskRunner:
    """Unified pipeline: Perception -> LLM Planner -> Skill Execution -> Control."""

    def __init__(
        self,
        scene_path: str,
        llm_planner: Optional["DialogPlanner"] = None,
        task_instruction: str = "",
        plan_dict: Optional[dict] = None
    ):
        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        # Perception
        self.perception = PerceptionModule(
            self.model, self.data,
            scene_path=scene_path,
            camera_names=['cam_global'],
            tracked_bodies=['shade_board', 'valve_body_1', 'valve_body_2',
                           'valve_body_3', 'cabinet_knob_body', 'mobile_cart'],
            tracked_sites=['tcp', 'tcp_R'],
            sensor_config={
                'arm_L_joints': (0, 6),
                'arm_R_joints': (6, 12),
                'cart_pos': (12, 15),
                'valve_1_angle': (15, 16),
                'valve_2_angle': (16, 17),
                'valve_3_angle': (17, 18),
                'cabinet_knob_angle': (18, 19),
            }
        )

        # LLM Planner
        self.llm_planner = llm_planner
        self.task_instruction = task_instruction
        self.plan_dict = plan_dict

        # 如果是 DialogPlanner, 注入 MuJoCo 环境以激活 FeedbackManager
        if hasattr(llm_planner, 'attach_env'):
            llm_planner.attach_env(
                model=self.model,
                data=self.data,
                arm_offsets={'L': ARM_L_OFFSET, 'R': ARM_R_OFFSET},
                arm_yaws={'L': ARM_L_YAW, 'R': ARM_R_YAW},
                dof=DOF,
            )

        # Plan Executor
        self.plan_executor = PlanExecutor(
            arm_offsets={'L': ARM_L_OFFSET, 'R': ARM_R_OFFSET},
            arm_yaws={'L': ARM_L_YAW, 'R': ARM_R_YAW}
        )
        from ..task_planning.planning_service import PlanningService
        self.planning_service = PlanningService(
            self.model, self.data, {'L': ARM_L_OFFSET, 'R': ARM_R_OFFSET},
            {'L': ARM_L_YAW, 'R': ARM_R_YAW},
        )

        # Arm states
        self.arm_L = ArmState('L', ARM_L_OFFSET, ARM_L_YAW, SENSOR_L_OFFSET)
        self.arm_R = ArmState('R', ARM_R_OFFSET, ARM_R_YAW, SENSOR_R_OFFSET)
        self.arms = {'L': self.arm_L, 'R': self.arm_R}

        # Initialize STA controllers for each arm
        ts = self.model.opt.timestep
        for arm in self.arms.values():
            arm.sta_controllers = [
                STAController(10, 25, 15, ts=ts) for _ in range(DOF)
            ]

        # TCP site IDs for trajectory recording
        self._tcp_L_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, 'tcp')
        self._tcp_R_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, 'tcp_R')
        self._tcp_L_history = []
        self._tcp_R_history = []
        self._traj_step_count = 0

        # Resolve actuator indices
        self._resolve_indices()
        self._total_time = 300.0
        self._is_started = False
        self._reset_runtime_state()
        self.reset_scene()

    def _resolve_indices(self):
        """Find ctrl array indices for arms and cart."""
        _id = lambda name: mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

        self._L_ctrl_start = _id("shoulder_pan")
        self._R_ctrl_start = _id("shoulder_pan_R")
        self._L_gripper_ctrl = _id("fingers_actuator")
        self._R_gripper_ctrl = _id("fingers_actuator_R")
        self._cart_x_ctrl = _id("cart_x_actuator")
        self._cart_y_ctrl = _id("cart_y_actuator")
        self._cart_yaw_ctrl = _id("cart_yaw_actuator")

        # Joint qpos addresses
        _jid = lambda name: mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        self._cart_x_qpos = self.model.jnt_qposadr[_jid("cart_x")]
        self._cart_y_qpos = self.model.jnt_qposadr[_jid("cart_y")]
        self._cart_yaw_qpos = self.model.jnt_qposadr[_jid("cart_yaw")]
        self._L_qpos_start = self.model.jnt_qposadr[_jid("shoulder_pan_joint")]
        self._R_qpos_start = self.model.jnt_qposadr[_jid("shoulder_pan_joint_R")]

    _MAX_PLAN_RETRIES = 2

    def _do_planning(self):
        """Perception -> LLM -> Parse plan, with retry on failure."""
        print("[TaskRunner] Planning...")

        if self.plan_dict:
            plan_result = self.plan_dict
            self._try_parse_plan(plan_result)
            return

        if not self.llm_planner:
            raise ValueError("No LLM planner or plan_dict provided")

        snapshot = self.perception.get_scene_snapshot(include_images=False)
        error_feedback = ""
        for attempt in range(1 + self._MAX_PLAN_RETRIES):
            plan_result = self.llm_planner.plan(
                snapshot, self.task_instruction, extra_context=error_feedback
            )
            print(f"[TaskRunner] LLM plan (attempt {attempt + 1}): "
                  f"{plan_result.get('reasoning', '')}")

            try:
                self._try_parse_plan(plan_result, snapshot)
                return
            except ValueError as e:
                error_feedback = (
                    f"## 上一次规划失败\n\n"
                    f"错误信息: {e}\n\n"
                    f"请修正 goals/stages，保留全部必做目标和必需辅助。"
                    f"如果目标超出可达范围，在 stage.nav 中调整小车停靠位置或 yaw。"
                )
                print(f"[TaskRunner] 规划失败 (attempt {attempt + 1}): {e}")
                if attempt == self._MAX_PLAN_RETRIES:
                    raise

    def prepare_plan(self, plan_result, snapshot=None, instruction=None):
        if snapshot is None:
            snapshot = self.perception.get_scene_snapshot(include_images=False)
        cart_pose = [*self._get_cart_state(), float(self.data.qpos[self._cart_yaw_qpos])]
        return self.planning_service.prepare(
            plan_result, snapshot, self.task_instruction if instruction is None else instruction, cart_pose,
        )

    def _try_parse_plan(self, plan_result: dict, snapshot=None):
        """Parse plan into phases. Raises ValueError on failure."""
        cart_state = self._get_cart_state()
        self.prepared_plan = self.prepare_plan(plan_result, snapshot)
        self.planning_report = self.prepared_plan.get('optimization', {})
        from ..task_planning.effects import EffectMonitor
        self._effect_monitor = EffectMonitor(self.model, self.data, self.perception.registry,
                                             self.prepared_plan.get('goals', []), self.planning_service.config)
        parsed = self.plan_executor.parse(self.prepared_plan, cart_state[:2],
                                          float(self.data.qpos[self._cart_yaw_qpos]))

        self._phases = parsed.get('phases', [])
        self._phase_index = 0

        for i, phase in enumerate(self._phases):
            nav = phase.get('nav')
            n_L = len(phase.get('L', []))
            n_R = len(phase.get('R', []))
            nav_desc = f"nav->{nav['target']}" if nav else "no-nav"
            print(f"[TaskRunner] Phase {i}: {nav_desc}, L:{n_L} skills, R:{n_R} skills")

    def _start_phase(self, phase_index: int):
        """Begin a phase: navigate if needed, then prepare arm executors."""
        if phase_index >= len(self._phases):
            self.state = 'DONE'
            print(f"[TaskRunner] All phases done at t={self.data.time:.2f}s")
            return

        phase = self._phases[phase_index]
        nav = phase.get('nav')

        if nav and nav.get('target'):
            self._start_navigation(nav['target'], yaw=nav.get('yaw'))
            self.state = 'PHASE_NAV'
        else:
            self._start_phase_execution(phase_index)

    def _make_return_skill(self):
        return MoveSkill(
            target_joints=INIT_Q.copy(),
            duration=3.0,
            sta_params=SKILL_STA_PARAMS[SkillType.MOVE],
            name="回到初始位姿",
        )

    def _setup_arm_executor(self, label: str, skills: list, phase_index: int,
                            append_return: bool = True):
        """Initialize executor for one arm with given skills."""
        arm = self.arms[label]
        skills = list(skills)
        if append_return:
            skills.append(self._make_return_skill())
        n = len(skills)
        suffix = " (含回初始位姿)" if append_return else " (hold最终位姿)"
        print(f"[TaskRunner] Phase {phase_index} Arm {label}: {n} skills{suffix}")
        arm.executor = SkillExecutor(skills, arm.sta_controllers, DOF)
        arm.ctx = SkillContext(
            self.model, self.data, arm.robot, DOF, arm.sensor_offset
        )
        flange_name = 'flange' if label == 'L' else 'flange_R'
        arm.ctx.flange_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, flange_name)
        arm.ctx.skill_start_time = self.data.time

    def _start_phase_execution(self, phase_index: int):
        """Execute the explicit single, parallel, or setup/hold/cleanup mode."""
        phase = self._phases[phase_index]
        self._init_arms()
        self._effect_monitor.start_phase(phase.get('goal_ids', []), phase['mode'] == 'assist_R_then_L')

        has_R = bool(phase.get('R', []))
        has_L = bool(phase.get('L', []))

        self._pending_L_skills = None
        self._L_waiting_for_R = False
        self._R_cleanup_pending = False
        self._R_cleanup_skills = None
        self._R_hold_position = None

        if has_R and has_L:
            if phase['mode'] == 'assist_R_then_L':
                r_skills = list(phase['R'])
                self._R_cleanup_skills = phase['R_cleanup']
                self._setup_arm_executor('R', r_skills, phase_index,
                                         append_return=False)
                self._pending_L_skills = phase['L']
                self._L_waiting_for_R = True
                self._R_cleanup_pending = True
                self.arm_L.executor = None
                print(f"[TaskRunner] Phase {phase_index}: "
                      f"顺序模式 (R臂先执行 → L臂后执行 → R臂放回)")
            else:
                self._setup_arm_executor('L', phase['L'], phase_index)
                self._setup_arm_executor('R', phase['R'], phase_index)
                print(f"[TaskRunner] Phase {phase_index}: "
                      f"并行模式 (L臂和R臂同时执行)")
        elif has_R:
            self._setup_arm_executor('R', phase['R'], phase_index)
            self.arm_L.executor = None
        elif has_L:
            self._setup_arm_executor('L', phase['L'], phase_index)
            self.arm_R.executor = None
        else:
            self.arm_L.executor = None
            self.arm_R.executor = None

        self.state = 'PHASE_EXEC'
        print(f"[TaskRunner] Phase {phase_index} executing...")

    def _get_cart_state(self) -> np.ndarray:
        """Returns [x, y] from qpos."""
        return np.array([
            self.data.qpos[self._cart_x_qpos],
            self.data.qpos[self._cart_y_qpos]
        ])

    def _start_navigation(self, target: list, yaw: float = None):
        """Create a move-then-turn profile from the current cart pose."""
        self._navigation = NavigationTrajectory(
            start=self._get_cart_state(), target=np.array(target),
            start_time=self.data.time,
            start_yaw=float(self.data.qpos[self._cart_yaw_qpos]),
            target_yaw=yaw,
        )
        print(f"[TaskRunner] Nav: {self._navigation.start} -> "
              f"{self._navigation.target}, duration={self._navigation.duration:.1f}s")

    def _update_navigation(self, t: float):
        """Write the navigation profile to the existing cart actuators."""
        if self._navigation is None:
            return
        position, yaw = self._navigation.sample(t)
        self.data.ctrl[self._cart_x_ctrl] = position[0]
        self.data.ctrl[self._cart_y_ctrl] = position[1]
        if yaw is not None:
            self.data.ctrl[self._cart_yaw_ctrl] = yaw

    def _cart_reached(self, target: np.ndarray) -> bool:
        """Check if cart reached target."""
        current = self._get_cart_state()
        return np.linalg.norm(current - target) < 0.05

    def _get_arm_sensor(self, arm: ArmState) -> np.ndarray:
        """Read 6 joint sensors for this arm."""
        s = arm.sensor_offset
        return self.data.sensordata[s:s + DOF].copy()

    def _compute_arm_control(self, arm: ArmState, desired: np.ndarray,
                             desired_velocity: np.ndarray = None) -> np.ndarray:
        """Compute STA control for one arm."""
        sensor = self._get_arm_sensor(arm)
        ts = self.model.opt.timestep
        vel = (sensor - arm.prev_sensor) / ts
        arm.prev_sensor = sensor.copy()

        error_pos = desired - sensor
        if desired_velocity is None:
            desired_velocity = np.zeros(DOF)
        error_vel = desired_velocity - vel

        ctrl = np.zeros(DOF)
        for i in range(DOF):
            ctrl[i] = arm.sta_controllers[i].control(error_pos[i], error_vel[i])

        return ctrl

    def _update_state(self, t: float):
        """State machine update with multi-phase support."""
        if self.state == 'INIT':
            self._do_planning()
            self._start_phase(self._phase_index)

        elif self.state == 'PHASE_NAV':
            self._update_navigation(t)
            navigation = self._navigation
            if (navigation is not None
                    and t - navigation.start_time >= navigation.duration):
                if self._cart_reached(navigation.target):
                    self._start_phase_execution(self._phase_index)

        elif self.state == 'PHASE_EXEC':
            R_done = True
            L_done = True

            if self.arm_R.executor:
                self.arm_R.ctx.current_time = t
                R_done = self.arm_R.executor.update(self.arm_R.ctx)

            # R臂遮光完成 → 保存hold位置, 启动L臂操作阀门
            if self._L_waiting_for_R and R_done and self._pending_L_skills:
                last_skill = self.arm_R.executor.skills[-1]
                if hasattr(last_skill, 'target_joints') and last_skill.target_joints is not None:
                    self._R_hold_position = last_skill.target_joints.copy()
                else:
                    self._R_hold_position = self._get_arm_sensor(self.arm_R)
                print(f"[{t:.2f}s] R臂遮光就位 (hold锁定), 启动L臂操作阀门")
                sensor = self._get_arm_sensor(self.arm_L)
                self.arm_L.robot.set_joint(sensor)
                self.arm_L.prev_sensor = sensor.copy()
                self._setup_arm_executor('L', self._pending_L_skills,
                                         self._phase_index)
                self._pending_L_skills = None
                self._L_waiting_for_R = False

            if self.arm_L.executor:
                self.arm_L.ctx.current_time = t
                L_done = self.arm_L.executor.update(self.arm_L.ctx)
            self._effect_monitor.sample(self.arms)

            # L臂完成(含回初始位姿) → 启动R臂放回遮光板
            if (L_done and R_done and self._R_cleanup_pending
                    and self._R_cleanup_skills is not None):
                print(f"[{t:.2f}s] L臂完成, R臂开始放回遮光板")
                prev_gripper = (self.arm_R.ctx.gripper_target
                                if self.arm_R.ctx else 0.0)
                sensor = self._get_arm_sensor(self.arm_R)
                self.arm_R.robot.set_joint(sensor)
                self.arm_R.prev_sensor = sensor.copy()
                cleanup_skills = self._R_cleanup_skills
                self._setup_arm_executor('R', cleanup_skills,
                                         self._phase_index)
                self.arm_R.ctx.gripper_target = prev_gripper
                self._R_cleanup_pending = False
                self._R_cleanup_skills = None
                R_done = False

            # 全部完成 → 进入下一阶段
            if (L_done and R_done
                    and not self._L_waiting_for_R
                    and not self._R_cleanup_pending):
                verified = self._effect_monitor.finish_phase(self._phases[self._phase_index].get('goal_ids', []))
                self.execution_report = self._effect_monitor.report()
                if not verified:
                    self.state = 'FAILED'
                    print('[TaskRunner] 执行效果未通过传感器校验:', self.execution_report)
                    return
                self._phase_index += 1
                if self._phase_index < len(self._phases):
                    print(f"[TaskRunner] Phase {self._phase_index - 1} complete, "
                          f"starting phase {self._phase_index}")
                    self._start_phase(self._phase_index)
                else:
                    self.state = 'DONE'
                    print(f"[TaskRunner] All phases done at t={t:.2f}s")

    def _init_arms(self):
        """Initialize arm contexts from current sensors."""
        for arm in self.arms.values():
            sensor = self._get_arm_sensor(arm)
            arm.robot.set_joint(sensor)
            arm.prev_sensor = sensor.copy()
            if arm.ctx:
                arm.ctx.skill_start_time = self.data.time

    @staticmethod
    def _draw_trajectory(user_scn, points, rgba, sphere_size=0.008):
        from ..visualization.trajectories import draw_trajectory
        draw_trajectory(user_scn, points, rgba, sphere_size)

    def _plot_trajectory_3d(self):
        from ..visualization.trajectories import plot_tcp_trajectories
        plot_tcp_trajectories(self._tcp_L_history, self._tcp_R_history)

    def _initialize_poses(self):
        """Set initial cart and arm positions in qpos."""
        # Cart start position
        self.data.qpos[self._cart_x_qpos] = MOBILE_ROBOT.cart_start[0]
        self.data.qpos[self._cart_y_qpos] = MOBILE_ROBOT.cart_start[1]
        self.data.qpos[self._cart_yaw_qpos] = MOBILE_ROBOT.cart_start[2]

        # Arm initial poses
        for i in range(DOF):
            self.data.qpos[self._L_qpos_start + i] = INIT_Q[i]
            self.data.qpos[self._R_qpos_start + i] = INIT_Q[i]

        # Set cart actuator targets
        self.data.ctrl[self._cart_x_ctrl] = MOBILE_ROBOT.cart_start[0]
        self.data.ctrl[self._cart_y_ctrl] = MOBILE_ROBOT.cart_start[1]
        self.data.ctrl[self._cart_yaw_ctrl] = MOBILE_ROBOT.cart_start[2]

    def _reset_runtime_state(self):
        """Reset phase/executor bookkeeping without rebuilding model/data."""
        self.state = 'IDLE'
        self._phases = []
        self._phase_index = 0
        self._navigation = None
        self._pending_L_skills = None
        self._L_waiting_for_R = False
        self._R_cleanup_pending = False
        self._R_cleanup_skills = None
        self._R_hold_position = None
        self.prepared_plan = None
        self.planning_report = {}
        self.execution_report = {}
        self._effect_monitor = None
        for arm in self.arms.values():
            arm.executor = None
            arm.ctx = None

    def reset_scene(self):
        """Reset MuJoCo data to the project's initial Scene5 state."""
        mujoco.mj_resetData(self.model, self.data)
        self._initialize_poses()
        mujoco.mj_forward(self.model, self.data)

        self._tcp_L_history.clear()
        self._tcp_R_history.clear()
        self._traj_step_count = 0

        for arm in self.arms.values():
            arm.prev_sensor = self._get_arm_sensor(arm)
            params = SKILL_STA_PARAMS[SkillType.MOVE]
            for index, controller in enumerate(arm.sta_controllers):
                controller.reset()
                controller.set_parameter(params['alpha'][index], params['beta'][index], params['lambda_s'][index])

        self._reset_runtime_state()
        self._is_started = False

    def set_llm_planner(self, llm_planner: Optional["DialogPlanner"]):
        """Attach or replace the planner used by later web-dispatched tasks."""
        self.llm_planner = llm_planner
        if hasattr(llm_planner, 'attach_env'):
            llm_planner.attach_env(
                model=self.model,
                data=self.data,
                arm_offsets={'L': ARM_L_OFFSET, 'R': ARM_R_OFFSET},
                arm_yaws={'L': ARM_L_YAW, 'R': ARM_R_YAW},
                dof=DOF,
            )

    def start(
        self,
        total_time: float = 300.0,
        show_trajectory: bool = True,
        task_instruction: Optional[str] = None,
        plan_dict: Optional[dict] = None,
        llm_planner: Optional["DialogPlanner"] = None,
    ):
        """Start one pipeline task without opening the native MuJoCo viewer."""
        print("[TaskRunner] Starting pipeline...")
        if task_instruction is not None:
            self.task_instruction = task_instruction
        self.plan_dict = plan_dict
        if llm_planner is not None:
            self.set_llm_planner(llm_planner)

        self.reset_scene()
        self.state = 'INIT'
        self._total_time = float(total_time)
        self._show_trajectory = show_trajectory
        self._is_started = True

    @property
    def is_active(self) -> bool:
        return self._is_started and self.state not in ('DONE', 'TIMEOUT', 'FAILED')

    def progress(self) -> float:
        if self.state == 'DONE':
            return 1.0
        if self._total_time <= 0:
            return 0.0
        return float(min(max(self.data.time / self._total_time, 0.0), 1.0))

    def step_once(self) -> bool:
        """Advance the original project pipeline by exactly one MuJoCo step."""
        if not self._is_started:
            return False

        if self.data.time > self._total_time:
            self.state = 'TIMEOUT'
            self._is_started = False
            print(f"[TaskRunner] Timeout at t={self.data.time:.2f}s")
            return False

        t = self.data.time
        self._update_state(t)

        if self.state in ('DONE', 'FAILED'):
            self._is_started = False
            return False

        self._apply_control_step(t)
        mujoco.mj_step(self.model, self.data)
        self._record_tcp_history()
        return True

    def _apply_control_step(self, t: float):
        """Compute and write arm/gripper controls for the current pipeline step."""
        for label, arm in self.arms.items():
            if arm.executor and not arm.executor.is_all_complete:
                arm.ctx.current_time = t
                desired = arm.executor.get_desired_position(arm.ctx)
                desired_velocity = arm.executor.get_desired_velocity(arm.ctx)
            elif (label == 'R' and
                  (self._R_cleanup_pending or self._L_waiting_for_R)):
                desired = (self._R_hold_position
                           if self._R_hold_position is not None
                           else self._get_arm_sensor(arm))
                desired_velocity = np.zeros(DOF)
            else:
                desired = INIT_Q
                desired_velocity = np.zeros(DOF)

            ctrl = self._compute_arm_control(arm, desired, desired_velocity)

            ctrl_start = self._L_ctrl_start if label == 'L' else self._R_ctrl_start
            for i in range(DOF):
                self.data.ctrl[ctrl_start + i] = ctrl[i]

            gripper_ctrl = self._L_gripper_ctrl if label == 'L' else self._R_gripper_ctrl
            gripper_target = arm.ctx.gripper_target if arm.ctx else 0.0
            self.data.ctrl[gripper_ctrl] = gripper_target

    def _record_tcp_history(self):
        self._traj_step_count += 1
        if self._traj_step_count % 10 != 0:
            return
        if self._tcp_L_site_id >= 0:
            self._tcp_L_history.append(
                self.data.site_xpos[self._tcp_L_site_id].copy())
        if self._tcp_R_site_id >= 0:
            self._tcp_R_history.append(
                self.data.site_xpos[self._tcp_R_site_id].copy())

    def run(self, total_time: float = 300.0, use_viewer: bool = True,
            show_trajectory: bool = True):
        """Run the full pipeline."""
        def step_loop():
            self.start(total_time=total_time, show_trajectory=show_trajectory,
                       plan_dict=self.plan_dict)
            while self.step_once():
                yield

        if use_viewer:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                viewer.cam.lookat[:] = [0.0, 0.3, 0.6]
                viewer.cam.distance = 4.0
                viewer.cam.azimuth = 150
                viewer.cam.elevation = -25

                if show_trajectory:
                    viz_L_points = []
                    viz_R_points = []
                    viz_interval = 50

                for step_i, _ in enumerate(step_loop()):
                    if not viewer.is_running():
                        break

                    if show_trajectory and step_i % viz_interval == 0:
                        new_point = False
                        if self._tcp_L_site_id >= 0:
                            viz_L_points.append(
                                self.data.site_xpos[self._tcp_L_site_id].copy())
                            new_point = True
                        if self._tcp_R_site_id >= 0:
                            viz_R_points.append(
                                self.data.site_xpos[self._tcp_R_site_id].copy())
                            new_point = True

                        if new_point:
                            with viewer.lock():
                                viewer.user_scn.ngeom = 0
                                self._draw_trajectory(
                                    viewer.user_scn, viz_L_points,
                                    rgba=[1, 0.2, 0.2, 0.8])
                                self._draw_trajectory(
                                    viewer.user_scn, viz_R_points,
                                    rgba=[0.2, 0.4, 1, 0.8])

                    viewer.sync()
        else:
            for _ in step_loop():
                pass

        print(f"[TaskRunner] Pipeline stopped: {self.state}.")
        if show_trajectory:
            self._plot_trajectory_3d()




