"""
石油管道阀门旋拧任务控制器

四阶段解耦控制:
    Phase 1  小车导航到阀门1附近
    Phase 2  左臂操作阀门1, 右臂保持
    Phase 3  小车移动到阀门2附近
    Phase 4  右臂操作阀门2, 左臂保持

复用共享单臂 STA 控制器和技能库。
"""

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import mujoco.viewer
import numpy as np
from spatialmath import SE3, SO3

from src.config.paths import scene_path as bundled_scene
from src.skills.arm_controller import ArmController as SharedArmController
from src.perception import build_world_state
from src.skills import (
    SkillType, SKILL_STA_PARAMS,
    MoveSkill, GraspSkill, RotateSkill,
)


PHASE_NAV_1 = 0
PHASE_ARM_L = 1
PHASE_NAV_2 = 2
PHASE_ARM_R = 3
PHASE_DONE = 4

CART_SPEED = 0.2


# ============================================================================
# 场景配置
# ============================================================================

def pipeline_scene_config(world_state=None):
    if world_state is not None:
        objects = world_state["objects"]
        return {
            'cart_start':    np.array([-3.0, -0.5]),
            'cart_target_1': np.array(objects['cart_nav_target_1']['pose']['pos'][:2]),
            'cart_target_2': np.array(objects['cart_nav_target_2']['pose']['pos'][:2]),
            'arm_offset_L':  np.array([0.25, 0.0, 0.35]),
            'arm_offset_R':  np.array([-0.25, 0.0, 0.35]),
            'arm_yaw_R':     np.pi,
            'valve_1_hw':    np.array(objects['valve_1']['keypoints']['handle_center']['pos']),
            'valve_2_hw':    np.array(objects['valve_2']['keypoints']['handle_center']['pos']),
        }

    return {
        'cart_start':    np.array([-3.0, -0.5]),
        'cart_target_1': np.array([0.0, 0.0]),
        'cart_target_2': np.array([3.5, 0.0]),
        'arm_offset_L':  np.array([0.25, 0.0, 0.35]),
        'arm_offset_R':  np.array([-0.25, 0.0, 0.35]),
        'arm_yaw_R':     np.pi,
        'valve_1_hw':    np.array([0.45, 0.4, 0.65]),
        'valve_2_hw':    np.array([3.5, 0.4, 0.65]),
    }


# ============================================================================
# 目标关节角预计算
# ============================================================================

def precompute_valve_targets(cart_target, arm_offset, hw_pos, robot,
                              arm_yaw=0.0, tool_offset=0.155):
    """
    计算臂操作阀门的 IK 目标关节角

    Args:
        cart_target:  小车到达该阀门时的世界位置 [x, y]
        arm_offset:   臂在小车上的相对位置 [x, y, z]
        hw_pos:       手轮世界坐标 [x, y, z]
        robot:        UR5e 机器人对象
        arm_yaw:      臂绕 z 轴的旋转角 (右臂为 pi)
        tool_offset:  工具偏移
    """
    arm_world = np.array([
        cart_target[0] + arm_offset[0],
        cart_target[1] + arm_offset[1],
        arm_offset[2],
    ])

    delta_world = hw_pos - arm_world

    c, s = np.cos(arm_yaw), np.sin(arm_yaw)
    local_x =  c * delta_world[0] + s * delta_world[1]
    local_y = -s * delta_world[0] + c * delta_world[1]
    local_z =  delta_world[2]

    print(f"  臂世界坐标: ({arm_world[0]:.2f}, {arm_world[1]:.2f}, {arm_world[2]:.2f})")
    print(f"  手轮世界坐标: ({hw_pos[0]:.2f}, {hw_pos[1]:.2f}, {hw_pos[2]:.2f})")
    print(f"  臂局部偏移: ({local_x:.2f}, {local_y:.2f}, {local_z:.2f}), "
          f"距离={np.sqrt(local_x**2 + local_y**2 + local_z**2):.2f}m")

    grasp_h = local_z + tool_offset
    approach_h = grasp_h + 0.12
    retract_h = grasp_h + 0.15

    cart_targets = {
        'valve_approach': np.array([local_x, local_y, approach_h]),
        'valve_grasp':    np.array([local_x, local_y, grasp_h]),
        'valve_retract':  np.array([local_x, local_y, retract_h]),
    }

    down = SO3.Rx(np.pi)
    init_q = [0, 0, np.pi / 2, 0, -np.pi / 2, 0]
    robot.set_joint(init_q)
    robot.setRobotConfig(init_q)

    joint_targets = {}
    for name, pos in cart_targets.items():
        T = SE3.Rt(down.R, pos)
        q = robot.ikine(T)
        if len(q) > 0:
            joint_targets[name] = np.array(q)
            robot.set_joint(q)
            print(f"  {name}: OK, q(deg)={np.degrees(q)[:3]}...")
        else:
            print(f"  {name}: IK 失败! pos={pos}")
            joint_targets[name] = np.array(init_q)

    return joint_targets


# ============================================================================
# 技能序列
# ============================================================================

def build_valve_task(joint_targets, sta_params=None):
    t = joint_targets
    p_move   = sta_params or SKILL_STA_PARAMS[SkillType.MOVE]
    p_grasp  = sta_params or SKILL_STA_PARAMS[SkillType.GRASP]
    p_rotate = sta_params or SKILL_STA_PARAMS[SkillType.ROTATE]

    init_q = np.array([0, 0, np.pi / 2, 0, -np.pi / 2, 0])
    return [
        MoveSkill(t['valve_approach'], duration=3.0, sta_params=p_move,   name="移到手轮上方"),
        MoveSkill(t['valve_grasp'],    duration=2.5, sta_params=p_move,   name="下降到手轮"),
        GraspSkill(action='close', wait_time=1.0, gripper_closed=0.35,
                   sta_params=p_grasp, name="抓取手轮辐条"),
        RotateSkill(angle=np.pi, duration=5.0, joint_index=5,
                    sta_params=p_rotate, name="旋拧阀门"),
        GraspSkill(action='open', wait_time=0.5,     sta_params=p_grasp,  name="释放手轮"),
        MoveSkill(t['valve_retract'],  duration=2.5, sta_params=p_move,   name="退回安全位"),
        MoveSkill(init_q,             duration=3.0, sta_params=p_move,   name="回到初始姿态"),
    ]


# ============================================================================
# 单臂控制状态
# ============================================================================

class ArmController(SharedArmController):
    """Bind the shared controller to the valve experiment skill sequence."""

    def build_skills(self, joint_targets):
        self.set_skills(build_valve_task(joint_targets))


# ============================================================================
# 管道阀门任务控制器 (双臂版)
# ============================================================================

class PipelineValveTask:

    CART_POS_TOL = 0.05
    CART_YAW_TOL = 0.1

    def __init__(self, model_path: str, scene_cfg: dict = None):

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        self.dof = 6
        world_state = build_world_state(self.model, self.data, scene_path=model_path)
        self.scene_cfg = scene_cfg or pipeline_scene_config(world_state)

        # ---------- 索引查询 ----------
        def _act_id(name):
            return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        def _sens_id(name):
            return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)

        self._L_ctrl_start = _act_id("shoulder_pan")
        self._R_ctrl_start = _act_id("shoulder_pan_R")
        self._cart_x_ctrl = _act_id("cart_x_actuator")
        self._cart_y_ctrl = _act_id("cart_y_actuator")
        self._cart_yaw_ctrl = _act_id("cart_yaw_actuator")

        self._cart_x_sens = _sens_id("cart_x_pos")
        self._cart_y_sens = _sens_id("cart_y_pos")
        self._cart_yaw_sens = _sens_id("cart_yaw_pos")
        self._valve1_sens = _sens_id("valve_angle_1")
        self._valve2_sens = _sens_id("valve_angle_2")

        # ---------- 双臂控制器 ----------
        self.arm_L = ArmController(
            self.model, self.data, "左臂",
            "shoulder_pan_joint", "fingers_actuator",
            sensor_offset=0)
        self.arm_R = ArmController(
            self.model, self.data, "右臂",
            "shoulder_pan_joint_R", "fingers_actuator_R",
            sensor_offset=6)

        # ---------- 预计算目标 ----------
        cfg = self.scene_cfg
        print("预计算左臂目标 (阀门1)...")
        self.targets_L = precompute_valve_targets(
            cfg['cart_target_1'], cfg['arm_offset_L'], cfg['valve_1_hw'],
            self.arm_L.robot, arm_yaw=0.0)
        print("预计算右臂目标 (阀门2)...")
        self.targets_R = precompute_valve_targets(
            cfg['cart_target_2'], cfg['arm_offset_R'], cfg['valve_2_hw'],
            self.arm_R.robot, arm_yaw=cfg.get('arm_yaw_R', 0.0))

        self.arm_L.build_skills(self.targets_L)
        self.arm_R.build_skills(self.targets_R)

        print(f"\n左臂技能: {len(self.arm_L.skills)} 个")
        print(f"右臂技能: {len(self.arm_R.skills)} 个\n")

        # ---------- 运行时状态 ----------
        self.phase = PHASE_NAV_1
        self._nav_start_time = None
        self._nav_logged = False
        self._completion_time = None
        self._current_nav_target = None
        self._nav_duration = 0.0

        self.valve_1_history = []
        self.valve_2_history = []

    # ------------------------------------------------------------------
    # 小车控制
    # ------------------------------------------------------------------

    def _get_cart_state(self):
        return np.array([
            self.data.sensordata[self._cart_x_sens],
            self.data.sensordata[self._cart_y_sens],
            self.data.sensordata[self._cart_yaw_sens],
        ])

    def _set_cart_target(self, x, y, yaw=0.0):
        self.data.ctrl[self._cart_x_ctrl] = x
        self.data.ctrl[self._cart_y_ctrl] = y
        self.data.ctrl[self._cart_yaw_ctrl] = yaw

    def _cart_reached(self, target):
        state = self._get_cart_state()
        return (np.linalg.norm(state[:2] - target[:2]) < self.CART_POS_TOL
                and abs(state[2]) < self.CART_YAW_TOL)

    def _start_navigation(self, target, current_time):
        cart = self._get_cart_state()
        start = cart[:2]
        dist = np.linalg.norm(target - start)
        self._nav_start_time = current_time
        self._nav_duration = max(dist / CART_SPEED, 1.0)
        self._nav_start_pos = start.copy()
        self._current_nav_target = target
        self._nav_logged = False

    # ------------------------------------------------------------------
    # 臂传感器读取
    # ------------------------------------------------------------------

    def _get_arm_sensor(self, arm):
        s = arm.sensor_offset
        return self.data.sensordata[s:s + self.dof].copy()

    # ------------------------------------------------------------------
    # 阶段管理
    # ------------------------------------------------------------------

    def _update_phase(self, t):
        if self.phase == PHASE_NAV_1:
            target = self.scene_cfg['cart_target_1']
            if self._nav_start_time is None:
                self._start_navigation(target, t)
                print(f"[{t:.2f}s] Phase 1: 小车导航到阀门1附近 (预计 {self._nav_duration:.1f}s)")

            self._do_nav_interpolation(t)

            if self._nav_duration_elapsed(t) and self._cart_reached(target):
                print(f"[{t:.2f}s] 到达阀门1, 切换到 Phase 2: 左臂操作")
                self.phase = PHASE_ARM_L
                self._init_arm_phase(self.arm_L)

        elif self.phase == PHASE_ARM_L:
            self.arm_L.ctx.current_time = t
            done = self.arm_L.executor.update(self.arm_L.ctx)
            if done:
                if self._completion_time is None:
                    self._completion_time = t
                    v1 = np.degrees(self.data.sensordata[self._valve1_sens])
                    print(f"[{t:.2f}s] 左臂完成! 阀门1旋转: {v1:.1f}°")
                if t > self._completion_time + 1.0:
                    print(f"[{t:.2f}s] 切换到 Phase 3: 小车导航到阀门2")
                    self.phase = PHASE_NAV_2
                    self._nav_start_time = None
                    self._completion_time = None

        elif self.phase == PHASE_NAV_2:
            target = self.scene_cfg['cart_target_2']
            if self._nav_start_time is None:
                self._start_navigation(target, t)
                print(f"[{t:.2f}s] Phase 3: 小车导航到阀门2附近 (预计 {self._nav_duration:.1f}s)")

            self._do_nav_interpolation(t)

            if self._nav_duration_elapsed(t) and self._cart_reached(target):
                print(f"[{t:.2f}s] 到达阀门2, 切换到 Phase 4: 右臂操作")
                self.phase = PHASE_ARM_R
                self._init_arm_phase(self.arm_R)

        elif self.phase == PHASE_ARM_R:
            self.arm_R.ctx.current_time = t
            done = self.arm_R.executor.update(self.arm_R.ctx)
            if done:
                if self._completion_time is None:
                    self._completion_time = t
                    v2 = np.degrees(self.data.sensordata[self._valve2_sens])
                    print(f"[{t:.2f}s] 右臂完成! 阀门2旋转: {v2:.1f}°")
                if t > self._completion_time + 2.0:
                    self.phase = PHASE_DONE

    def _do_nav_interpolation(self, t):
        elapsed = t - self._nav_start_time
        alpha = min(elapsed / self._nav_duration, 1.0)
        interp = self._nav_start_pos + alpha * (self._current_nav_target - self._nav_start_pos)
        self._set_cart_target(interp[0], interp[1], 0.0)

    def _nav_duration_elapsed(self, t):
        return (t - self._nav_start_time) >= self._nav_duration

    def _init_arm_phase(self, arm):
        sensor = self._get_arm_sensor(arm)
        arm.robot.set_joint(sensor)
        arm.ctx.gripper_target = 0.0
        arm._prev_desired = sensor.copy()

    # ------------------------------------------------------------------
    # 单步控制
    # ------------------------------------------------------------------

    def _apply_arm_control(self, arm, ctrl_start, sensor_data, real_vel, desired):
        velocity = (arm.executor.get_desired_velocity(arm.ctx)
                    if self.phase in (PHASE_ARM_L, PHASE_ARM_R) and arm.executor
                    and not arm.executor.is_all_complete else np.zeros(self.dof))
        ctrl = arm.compute_control(desired, sensor_data, real_vel, self.model.opt.timestep,
                                   desired_velocity=velocity)
        for i in range(self.dof):
            self.data.ctrl[ctrl_start + i] = ctrl[i]
        self.data.ctrl[arm._gripper_ctrl] = arm.ctx.gripper_target

    # ------------------------------------------------------------------
    # 仿真主循环
    # ------------------------------------------------------------------

    def _run_simulation(self, total_time, use_viewer=True):
        mujoco.mj_resetData(self.model, self.data)

        cart_start = self.scene_cfg['cart_start']
        cart_x_qpos = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cart_x")]
        cart_y_qpos = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cart_y")]
        self.data.qpos[cart_x_qpos] = cart_start[0]
        self.data.qpos[cart_y_qpos] = cart_start[1]

        for arm in [self.arm_L, self.arm_R]:
            for i in range(self.dof):
                self.data.qpos[arm._arm_qpos_adr + i] = arm.initial_q[i]

        self._set_cart_target(cart_start[0], cart_start[1], 0.0)
        mujoco.mj_forward(self.model, self.data)

        real_pos_L = self._get_arm_sensor(self.arm_L)
        real_pos_L_prev = real_pos_L.copy()
        real_pos_R = self._get_arm_sensor(self.arm_R)
        real_pos_R_prev = real_pos_R.copy()

        ts = self.model.opt.timestep

        def step_loop():
            nonlocal real_pos_L, real_pos_L_prev, real_pos_R, real_pos_R_prev

            while self.data.time <= total_time:
                t = self.data.time
                self._update_phase(t)
                if self.phase == PHASE_DONE:
                    break

                # 左臂
                sensor_L = self._get_arm_sensor(self.arm_L)
                real_pos_L_prev = real_pos_L.copy()
                real_pos_L = sensor_L.copy()
                vel_L = (real_pos_L - real_pos_L_prev) / ts

                if self.phase == PHASE_ARM_L:
                    self.arm_L.ctx.current_time = t
                    desired_L = self.arm_L.executor.get_desired_position(self.arm_L.ctx)
                else:
                    desired_L = self.arm_L.initial_q

                self._apply_arm_control(
                    self.arm_L, self._L_ctrl_start, sensor_L, vel_L, desired_L)

                # 右臂
                sensor_R = self._get_arm_sensor(self.arm_R)
                real_pos_R_prev = real_pos_R.copy()
                real_pos_R = sensor_R.copy()
                vel_R = (real_pos_R - real_pos_R_prev) / ts

                if self.phase == PHASE_ARM_R:
                    self.arm_R.ctx.current_time = t
                    desired_R = self.arm_R.executor.get_desired_position(self.arm_R.ctx)
                else:
                    desired_R = self.arm_R.initial_q

                self._apply_arm_control(
                    self.arm_R, self._R_ctrl_start, sensor_R, vel_R, desired_R)

                self.valve_1_history.append(self.data.sensordata[self._valve1_sens])
                self.valve_2_history.append(self.data.sensordata[self._valve2_sens])

                mujoco.mj_step(self.model, self.data)
                yield

        if use_viewer:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                viewer.cam.lookat[:] = [0.6, 0.3, 0.5]
                viewer.cam.distance = 5.0
                viewer.cam.azimuth = 150
                viewer.cam.elevation = -25
                for _ in step_loop():
                    if not viewer.is_running():
                        break
                    viewer.sync()
        else:
            for _ in step_loop():
                pass

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def run(self, total_time=80.0):
        print("=" * 60)
        print("  Pipeline Dual-Arm Valve Task")
        print("  小车导航 → 左臂操作阀门1 → 移动 → 右臂操作阀门2")
        print("=" * 60 + "\n")

        self.phase = PHASE_NAV_1
        self._nav_start_time = None
        self._completion_time = None
        self.valve_1_history = []
        self.valve_2_history = []

        self.arm_L.build_skills(self.targets_L)
        self.arm_R.build_skills(self.targets_R)

        self._run_simulation(total_time, use_viewer=True)
        self._print_summary()

    def _print_summary(self):
        print("\n" + "=" * 60)
        print("  Dual-Arm Task Summary")
        print("=" * 60)
        if self.valve_1_history:
            print(f"  阀门1 最终旋转: {np.degrees(self.valve_1_history[-1]):.1f}°")
        if self.valve_2_history:
            print(f"  阀门2 最终旋转: {np.degrees(self.valve_2_history[-1]):.1f}°")
        print()

def main():
    scene_path = str(bundled_scene('scene4_pipeline'))

    print(f"Loading scene: {scene_path}")

    task = PipelineValveTask(scene_path)
    task.run(total_time=80.0)


if __name__ == '__main__':
    main()
