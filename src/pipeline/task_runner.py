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
from ..robot.ur5e import UR5e
from .plan_executor import PlanExecutor


# Scene5 hardcoded configuration
ARM_L_OFFSET = np.array([0.35, 0.0, 0.35])
ARM_R_OFFSET = np.array([-0.35, 0.0, 0.35])
ARM_L_YAW = 0.0
ARM_R_YAW = np.pi
INIT_Q = np.array([0, 0, np.pi/2, 0, -np.pi/2, 0])
SENSOR_L_OFFSET = 0
SENSOR_R_OFFSET = 6
DOF = 6


class ArmState:
    """State container for a single arm."""
    def __init__(self, label: str, offset: np.ndarray, yaw: float, sensor_offset: int):
        self.label = label
        self.offset = offset
        self.yaw = yaw
        self.sensor_offset = sensor_offset
        self.robot = UR5e()
        self.robot.set_joint(INIT_Q)
        self.sta_controllers = []
        self.executor = None
        self.ctx = None
        self.prev_sensor = INIT_Q.copy()


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

        # State machine
        self.state = 'INIT'
        self._phases = []
        self._phase_index = 0
        self._nav_start_time = None
        self._nav_start_pos = None
        self._nav_target = None
        self._nav_speed = 0.3
        self._nav_duration = 5.0
        self._pending_L_skills = None
        self._L_waiting_for_R = False
        self._R_cleanup_pending = False
        self._R_setup_skills = None
        self._R_hold_position = None

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

        snapshot = self.perception.get_scene_snapshot()

        if self.plan_dict:
            plan_result = self.plan_dict
            self._try_parse_plan(plan_result)
            return

        if not self.llm_planner:
            raise ValueError("No LLM planner or plan_dict provided")

        error_feedback = ""
        for attempt in range(1 + self._MAX_PLAN_RETRIES):
            plan_result = self.llm_planner.plan(
                snapshot, self.task_instruction, extra_context=error_feedback
            )
            print(f"[TaskRunner] LLM plan (attempt {attempt + 1}): "
                  f"{plan_result.get('reasoning', '')}")

            try:
                self._try_parse_plan(plan_result)
                return
            except ValueError as e:
                error_feedback = (
                    f"## 上一次规划失败\n\n"
                    f"错误信息: {e}\n\n"
                    f"请修正规划。如果目标超出臂的可达范围, "
                    f"必须先使用 NavSkill 将小车移动到目标附近, "
                    f"然后再使用 MoveSkill。"
                )
                print(f"[TaskRunner] 规划失败 (attempt {attempt + 1}): {e}")
                if attempt == self._MAX_PLAN_RETRIES:
                    raise

    def _try_parse_plan(self, plan_result: dict):
        """Parse plan into phases. Raises ValueError on failure."""
        cart_state = self._get_cart_state()
        parsed = self.plan_executor.parse(plan_result, cart_state[:2])

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

    def _build_R_cleanup_skills(self, r_setup_skills: list):
        """Build R arm cleanup skills: reverse MoveSkills back, release, return."""
        from ..skills.skills import GraspSkill as _GraspSkill
        p_move = SKILL_STA_PARAMS[SkillType.MOVE]
        p_grasp = SKILL_STA_PARAMS[SkillType.GRASP]

        cleanup = []
        reverse_targets = []
        for skill in r_setup_skills:
            if hasattr(skill, 'target_joints') and skill.target_joints is not None:
                reverse_targets.append(skill.target_joints.copy())

        for joints in reversed(reverse_targets):
            cleanup.append(MoveSkill(
                target_joints=joints, duration=3.0,
                sta_params=p_move, name="R:回程",
            ))

        cleanup.append(_GraspSkill(
            action='open', wait_time=0.8, sta_params=p_grasp,
            name="R:释放遮光板",
        ))
        cleanup.append(self._make_return_skill())
        return cleanup

    def _start_phase_execution(self, phase_index: int):
        """Set up arm executors for a phase. Sequential: R first, then L."""
        phase = self._phases[phase_index]
        self._init_arms()

        has_R = bool(phase.get('R', []))
        has_L = bool(phase.get('L', []))

        self._pending_L_skills = None
        self._L_waiting_for_R = False
        self._R_cleanup_pending = False
        self._R_setup_skills = None
        self._R_hold_position = None

        if has_R and has_L:
            r_skills = list(phase['R'])
            r_has_grasp = any(
                hasattr(s, 'action') for s in r_skills
            )

            if r_has_grasp:
                self._R_setup_skills = r_skills
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

    _YAW_SPEED = 0.5

    def _start_navigation(self, target: list, yaw: float = None):
        """Begin navigation: move to [x,y] first, then rotate yaw in place."""
        self._nav_start_pos = self._get_cart_state()
        self._nav_target = np.array(target)
        self._nav_start_time = self.data.time
        self._nav_start_yaw = self.data.qpos[self._cart_yaw_qpos]
        self._nav_target_yaw = yaw

        dist = np.linalg.norm(self._nav_target - self._nav_start_pos)
        self._nav_move_duration = max(dist / self._nav_speed, 2.0) if dist > 0.05 else 0.0

        if yaw is not None and abs(yaw - self._nav_start_yaw) > 0.01:
            yaw_diff = abs(yaw - self._nav_start_yaw)
            self._nav_yaw_duration = max(yaw_diff / self._YAW_SPEED, 2.0)
        else:
            self._nav_yaw_duration = 0.0

        self._nav_duration = max(self._nav_move_duration + self._nav_yaw_duration, 2.0)

        move_info = f"移动{dist:.2f}m/{self._nav_move_duration:.1f}s"
        yaw_info = f" -> 原地转向{self._nav_start_yaw:.2f}->{yaw:.2f}/{self._nav_yaw_duration:.1f}s" if self._nav_yaw_duration > 0 else ""
        print(f"[TaskRunner] Nav: {self._nav_start_pos} -> {self._nav_target} "
              f"({move_info}{yaw_info})")

    def _update_navigation(self, t: float):
        """Phase 1: move to position. Phase 2: rotate yaw in place."""
        if self._nav_start_time is None:
            return

        elapsed = t - self._nav_start_time

        if self._nav_move_duration > 0 and elapsed < self._nav_move_duration:
            s = elapsed / self._nav_move_duration
            current = self._nav_start_pos + s * (self._nav_target - self._nav_start_pos)
            self.data.ctrl[self._cart_x_ctrl] = current[0]
            self.data.ctrl[self._cart_y_ctrl] = current[1]
        else:
            self.data.ctrl[self._cart_x_ctrl] = self._nav_target[0]
            self.data.ctrl[self._cart_y_ctrl] = self._nav_target[1]

            if self._nav_yaw_duration > 0 and self._nav_target_yaw is not None:
                yaw_elapsed = elapsed - self._nav_move_duration
                s_yaw = min(yaw_elapsed / self._nav_yaw_duration, 1.0)
                current_yaw = self._nav_start_yaw + s_yaw * (self._nav_target_yaw - self._nav_start_yaw)
                self.data.ctrl[self._cart_yaw_ctrl] = current_yaw

    def _cart_reached(self, target: np.ndarray) -> bool:
        """Check if cart reached target."""
        current = self._get_cart_state()
        return np.linalg.norm(current - target) < 0.05

    def _get_arm_sensor(self, arm: ArmState) -> np.ndarray:
        """Read 6 joint sensors for this arm."""
        s = arm.sensor_offset
        return self.data.sensordata[s:s + DOF].copy()

    def _compute_arm_control(self, arm: ArmState, desired: np.ndarray) -> np.ndarray:
        """Compute STA control for one arm."""
        sensor = self._get_arm_sensor(arm)
        ts = self.model.opt.timestep
        vel = (sensor - arm.prev_sensor) / ts
        arm.prev_sensor = sensor.copy()

        error_pos = desired - sensor
        error_vel = -vel

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
            if t - self._nav_start_time >= self._nav_duration:
                if self._cart_reached(self._nav_target):
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

            # L臂完成(含回初始位姿) → 启动R臂放回遮光板
            if (L_done and R_done and self._R_cleanup_pending
                    and self._R_setup_skills is not None):
                print(f"[{t:.2f}s] L臂完成, R臂开始放回遮光板")
                prev_gripper = (self.arm_R.ctx.gripper_target
                                if self.arm_R.ctx else 0.0)
                sensor = self._get_arm_sensor(self.arm_R)
                self.arm_R.robot.set_joint(sensor)
                self.arm_R.prev_sensor = sensor.copy()
                cleanup_skills = self._build_R_cleanup_skills(
                    self._R_setup_skills)
                self._setup_arm_executor('R', cleanup_skills,
                                         self._phase_index,
                                         append_return=False)
                self.arm_R.ctx.gripper_target = prev_gripper
                self._R_cleanup_pending = False
                self._R_setup_skills = None
                R_done = False

            # 全部完成 → 进入下一阶段
            if (L_done and R_done
                    and not self._L_waiting_for_R
                    and not self._R_cleanup_pending):
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
        """Draw trajectory points and line segments in MuJoCo viewer."""
        max_geom = user_scn.maxgeom
        for i, pt in enumerate(points):
            if user_scn.ngeom >= max_geom - 2:
                break
            g_idx = user_scn.ngeom
            mujoco.mjv_initGeom(
                user_scn.geoms[g_idx],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.zeros(3),
                pt.astype(np.float64),
                np.zeros(9),
                np.array(rgba, dtype=np.float32),
            )
            user_scn.geoms[g_idx].size[:] = [sphere_size, 0, 0]
            user_scn.ngeom += 1

            if i > 0 and user_scn.ngeom < max_geom:
                prev = points[i - 1]
                mid = (pt + prev) * 0.5
                diff = pt - prev
                length = float(np.linalg.norm(diff))
                if length < 1e-6:
                    continue
                g_idx2 = user_scn.ngeom
                mujoco.mjv_initGeom(
                    user_scn.geoms[g_idx2],
                    mujoco.mjtGeom.mjGEOM_CAPSULE,
                    np.array([0.003, length / 2, 0]),
                    mid.astype(np.float64),
                    np.zeros(9),
                    np.array(rgba, dtype=np.float32),
                )
                d = diff / length
                up = np.array([0.0, 0.0, 1.0])
                if abs(np.dot(d, up)) > 0.99:
                    up = np.array([1.0, 0.0, 0.0])
                right = np.cross(up, d)
                right /= np.linalg.norm(right)
                up2 = np.cross(d, right)
                rot = np.array([right, up2, d]).T
                user_scn.geoms[g_idx2].mat[:] = rot
                user_scn.ngeom += 1

    def _plot_trajectory_3d(self):
        """Plot TCP trajectories in 3D after simulation."""
        if not self._tcp_L_history and not self._tcp_R_history:
            return
        try:
            import matplotlib
            matplotlib.rcParams['font.sans-serif'] = [
                'SimHei', 'Microsoft YaHei', 'DejaVu Sans']
            matplotlib.rcParams['axes.unicode_minus'] = False
            import matplotlib.pyplot as plt
        except ImportError:
            print("[TaskRunner] matplotlib not available, skipping 3D plot")
            return

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        if self._tcp_L_history:
            pts = np.array(self._tcp_L_history)
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                    'r-', linewidth=1.5, alpha=0.8, label='L臂 (主臂)')
            ax.scatter(*pts[0], color='red', s=80, marker='o',
                       edgecolors='black', zorder=5)
            ax.scatter(*pts[-1], color='red', s=80, marker='s',
                       edgecolors='black', zorder=5)

        if self._tcp_R_history:
            pts = np.array(self._tcp_R_history)
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                    'b-', linewidth=1.5, alpha=0.8, label='R臂 (从臂)')
            ax.scatter(*pts[0], color='blue', s=80, marker='o',
                       edgecolors='black', zorder=5)
            ax.scatter(*pts[-1], color='blue', s=80, marker='s',
                       edgecolors='black', zorder=5)

        pipe_x = np.linspace(-1, 4, 50)
        ax.plot(pipe_x, [0.4] * 50, [0.5] * 50,
                'gray', linewidth=3, alpha=0.3, label='管道A')

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('双臂末端TCP轨迹')
        ax.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig('tcp_trajectory_3d.png', dpi=150)
        print("[TaskRunner] 轨迹图已保存: tcp_trajectory_3d.png")
        plt.show()

    def _initialize_poses(self):
        """Set initial cart and arm positions in qpos."""
        # Cart start position
        self.data.qpos[self._cart_x_qpos] = -3.0
        self.data.qpos[self._cart_y_qpos] = -0.5
        self.data.qpos[self._cart_yaw_qpos] = 0.0

        # Arm initial poses
        for i in range(DOF):
            self.data.qpos[self._L_qpos_start + i] = INIT_Q[i]
            self.data.qpos[self._R_qpos_start + i] = INIT_Q[i]

        # Set cart actuator targets
        self.data.ctrl[self._cart_x_ctrl] = -3.0
        self.data.ctrl[self._cart_y_ctrl] = -0.5
        self.data.ctrl[self._cart_yaw_ctrl] = 0.0

    def _reset_runtime_state(self):
        """Reset phase/executor bookkeeping without rebuilding model/data."""
        self.state = 'IDLE'
        self._phases = []
        self._phase_index = 0
        self._nav_start_time = None
        self._nav_start_pos = None
        self._nav_target = None
        self._nav_start_yaw = None
        self._nav_target_yaw = None
        self._nav_move_duration = 0.0
        self._nav_yaw_duration = 0.0
        self._nav_speed = 0.3
        self._nav_duration = 5.0
        self._pending_L_skills = None
        self._L_waiting_for_R = False
        self._R_cleanup_pending = False
        self._R_setup_skills = None
        self._R_hold_position = None
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

        if self.state == 'DONE':
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
            elif (label == 'R' and
                  (self._R_cleanup_pending or self._L_waiting_for_R)):
                desired = (self._R_hold_position
                           if self._R_hold_position is not None
                           else self._get_arm_sensor(arm))
            else:
                desired = INIT_Q

            ctrl = self._compute_arm_control(arm, desired)

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
            self.start(total_time=total_time, show_trajectory=show_trajectory)
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

        print("[TaskRunner] Pipeline complete.")
        if show_trajectory:
            self._plot_trajectory_3d()




