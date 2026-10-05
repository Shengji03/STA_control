if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import mujoco.viewer
import numpy as np
from spatialmath import SE3, SO3

from src.config.paths import scene_path as bundled_scene
from src.skills.arm_controller import ArmController
from src.perception import build_world_state
from src.robot import UR5e
from src.geometry import Capsule, Brick
from src.motion_planning.rrt import RRTMap, RRTPlanner, RobotRRTParameter
from src.robot.robot import get_transformation_mdh
from src.skills import (
    SkillType, SkillExecutor, SKILL_STA_PARAMS,
    MoveSkill, GraspSkill, RotateSkill,
)


PHASE_NAV      = 0
PHASE_SHADE    = 1
PHASE_VALVE    = 2
PHASE_L_RETURN = 3
PHASE_UNSHADE  = 4
PHASE_DONE     = 5

PHASE_NAMES = {
    0: "NAV",  1: "SHADE",  2: "VALVE",
    3: "L_RETURN",  4: "UNSHADE",  5: "DONE",
}

CART_SPEED = 0.2


# ============================================================================
# 场景配置
# ============================================================================

def glare_scene_config(world_state=None):
    if world_state is not None:
        objects = world_state["objects"]
        return {
            'cart_start':       np.array([-3.0, -0.5]),
            'cart_target':      np.array(objects['cart_nav_target']['pose']['pos'][:2]),
            'arm_offset_L':     np.array([0.25, 0.0, 0.35]),
            'arm_offset_R':     np.array([-0.25, 0.0, 0.35]),
            'arm_yaw_R':        np.pi,
            'valve_1_hw':       np.array(objects['valve_1']['keypoints']['handle_center']['pos']),
            'shade_handle_pos': np.array(objects['shade_board']['keypoints']['grasp_center']['pos']),
            'shade_block_pos':  np.array(objects['shade_block_target']['pose']['pos']),
        }

    return {
        'cart_start':       np.array([-3.0, -0.5]),
        'cart_target':      np.array([0.45, 0.0]),
        'arm_offset_L':     np.array([0.25, 0.0, 0.35]),
        'arm_offset_R':     np.array([-0.25, 0.0, 0.35]),
        'arm_yaw_R':        np.pi,
        'valve_1_hw':       np.array([0.45, 0.4, 0.65]),
        'shade_handle_pos': np.array([0.2, -0.8, 0.415]),
        'shade_block_pos':  np.array([0.45, -0.10, 0.95]),
    }


# ============================================================================
# RRT 避碰
# ============================================================================

class UR5eWithBoard(UR5e):
    """扩展 UR5e, get_geometries() 额外返回末端夹持的遮光板碰撞体"""

    BOARD_DIMS = np.array([0.30, 0.24, 0.01])
    BOARD_OFFSET_Z = 0.15

    def get_geometries(self):
        geoms = super().get_geometries()
        T = SE3()
        for i in range(self._dof):
            T = T * get_transformation_mdh(
                self.alpha_array[i], self.a_array[i], self.d_array[i],
                self.theta_array[i], self.sigma_array[i], self.q0[i])
        T_board = T * SE3.Trans(0, 0, self.BOARD_OFFSET_Z)
        geoms.append(Brick(T_board, self.BOARD_DIMS))
        return geoms


def get_l_arm_obstacles(arm_offset_L, arm_offset_R, arm_yaw_R, initial_q):
    """计算 L 臂在初始姿态下的连杆碰撞体, 变换到 R 臂基座系"""
    robot_L = UR5e()
    robot_L.set_joint(list(initial_q))
    geoms_L = robot_L.get_geometries()

    T_L = SE3.Trans(*arm_offset_L)
    T_R = SE3.Trans(*arm_offset_R) * SE3.Rz(arm_yaw_R)
    T_L_in_R = T_R.inv() * T_L

    obstacles = []
    for cap in geoms_L:
        obstacles.append(Capsule(T_L_in_R * cap.base, cap.radius, cap.length))
    return obstacles


def plan_rrt_shade_path(q_start, q_goal, robot, obstacles, max_iter=500):
    """用 RRT 在 R 臂关节空间规划从 q_start 到 q_goal 的无碰撞路径

    使用 UR5eWithBoard 作为碰撞模型, 同时检测臂连杆和夹持遮光板与 L 臂的碰撞。

    Returns:
        成功时返回关节角序列 [q0, q1, ..., qN] (从 start 到 goal),
        失败时返回 None。
    """
    robot_with_board = UR5eWithBoard()
    robot_with_board.set_joint(list(robot.get_joint()))

    area = [(-2 * np.pi, 2 * np.pi)] * 6
    rrt_map = RRTMap(area, obstacles)
    rrt_param = RobotRRTParameter(
        start=np.array(q_start), goal=np.array(q_goal), robot=robot_with_board,
        expand_dis=0.15, goal_sample_rate=15, max_iter=max_iter)
    planner = RRTPlanner(rrt_map, rrt_param)

    if not planner.success:
        print(f"  RRT ({max_iter} iter) 未找到路径, 尝试 {max_iter * 2} iter...")
        rrt_param2 = RobotRRTParameter(
            start=np.array(q_start), goal=np.array(q_goal), robot=robot_with_board,
            expand_dis=0.15, goal_sample_rate=15, max_iter=max_iter * 2)
        planner = RRTPlanner(rrt_map, rrt_param2)
        if not planner.success:
            print("  RRT 规划失败!")
            return None

    path_nodes = planner.get_final_course()[::-1]
    path = [node.get_t() for node in path_nodes]
    print(f"  RRT 规划成功: {len(path)} 个航点")
    return path


# ============================================================================
# IK 预计算
# ============================================================================

def _world_to_arm_local(world_pos, arm_world, arm_yaw):
    """将世界坐标转换到臂局部坐标 (考虑 yaw 旋转)"""
    delta = world_pos - arm_world
    c, s = np.cos(arm_yaw), np.sin(arm_yaw)
    local_x =  c * delta[0] + s * delta[1]
    local_y = -s * delta[0] + c * delta[1]
    local_z =  delta[2]
    return np.array([local_x, local_y, local_z])


def _solve_ik(robot, local_pos, tool_offset=0.155, label="", orientation=None):
    """在臂局部坐标系中求解 IK, tool_offset 沿末端 z 轴方向偏移"""
    if orientation is None:
        orientation = SO3.Rx(np.pi)
    z_ee = orientation.R @ np.array([0, 0, 1])
    pos = np.array(local_pos) - tool_offset * z_ee
    T = SE3.Rt(orientation.R, pos)
    q = robot.ikine(T)
    if len(q) > 0:
        robot.set_joint(q)
        return np.array(q)
    print(f"    [{label}] IK 失败! local_pos={local_pos}")
    return None


def precompute_valve_targets(cart_target, arm_offset, hw_pos, robot,
                              arm_yaw=0.0, tool_offset=0.155):
    """计算阀门操作的 IK 目标关节角 (与 pipeline_valve_task 相同逻辑)"""
    arm_world = np.array([
        cart_target[0] + arm_offset[0],
        cart_target[1] + arm_offset[1],
        arm_offset[2],
    ])
    local = _world_to_arm_local(hw_pos, arm_world, arm_yaw)
    print(f"  臂世界: ({arm_world[0]:.2f}, {arm_world[1]:.2f}, {arm_world[2]:.2f})")
    print(f"  手轮世界: ({hw_pos[0]:.2f}, {hw_pos[1]:.2f}, {hw_pos[2]:.2f})")
    print(f"  臂局部: ({local[0]:.2f}, {local[1]:.2f}, {local[2]:.2f})")

    grasp_h = local[2] + tool_offset
    targets_local = {
        'valve_approach': np.array([local[0], local[1], grasp_h + 0.12]),
        'valve_grasp':    np.array([local[0], local[1], grasp_h]),
        'valve_retract':  np.array([local[0], local[1], grasp_h + 0.15]),
    }

    init_q = [0, 0, np.pi / 2, 0, -np.pi / 2, 0]
    robot.set_joint(init_q)
    robot.setRobotConfig(init_q)
    down = SO3.Rx(np.pi)

    joint_targets = {}
    for name, pos in targets_local.items():
        T = SE3.Rt(down.R, pos)
        q = robot.ikine(T)
        if len(q) > 0:
            joint_targets[name] = np.array(q)
            robot.set_joint(q)
            print(f"  {name}: OK")
        else:
            print(f"  {name}: IK 失败!")
            joint_targets[name] = np.array(init_q)
    return joint_targets


def precompute_shade_targets(cart_target, arm_offset, handle_pos, block_pos,
                              robot, obstacles, arm_yaw=np.pi, tool_offset=0.155):

    arm_world = np.array([
        cart_target[0] + arm_offset[0],
        cart_target[1] + arm_offset[1],
        arm_offset[2],
    ])
    print(f"  R臂世界: ({arm_world[0]:.2f}, {arm_world[1]:.2f}, {arm_world[2]:.2f})")

    down_orient = SO3.Rx(np.pi)
    block_orient = SO3.Rx(7 * np.pi / 9)

    ik_targets = [
        ('shade_approach', np.array([handle_pos[0], handle_pos[1], handle_pos[2] + 0.12]), down_orient),
        ('shade_grasp',    np.array(handle_pos),                                           down_orient),
        ('shade_lift',     np.array([handle_pos[0], handle_pos[1], handle_pos[2] + 0.22]), down_orient),
        ('shade_block',    np.array(block_pos),                                            block_orient),
    ]

    init_q = [0, 0, np.pi / 2, 0, -np.pi / 2, 0]
    robot.set_joint(init_q)
    robot.setRobotConfig(init_q)

    joint_targets = {}
    for name, wpos, orient in ik_targets:
        local = _world_to_arm_local(wpos, arm_world, arm_yaw)
        print(f"  {name}: world=({wpos[0]:.2f},{wpos[1]:.2f},{wpos[2]:.2f}), "
              f"local=({local[0]:.2f},{local[1]:.2f},{local[2]:.2f})")
        q = _solve_ik(robot, local, tool_offset, label=name, orientation=orient)
        if q is not None:
            joint_targets[name] = q
            print(f"    IK OK")
        else:
            joint_targets[name] = np.array(init_q)

    print("\n  RRT* 规划 shade_lift → shade_block ...")
    fwd_path = plan_rrt_shade_path(
        joint_targets['shade_lift'], joint_targets['shade_block'],
        robot, obstacles)
    if fwd_path is not None:
        joint_targets['rrt_path_forward'] = fwd_path
        joint_targets['rrt_path_reverse'] = fwd_path[::-1]
    else:
        print("  WARNING: RRT 失败, 回退为直接移动 (可能碰撞)")
        joint_targets['rrt_path_forward'] = [
            joint_targets['shade_lift'], joint_targets['shade_block']]
        joint_targets['rrt_path_reverse'] = [
            joint_targets['shade_block'], joint_targets['shade_lift']]

    return joint_targets


# ============================================================================
# 技能序列构建
# ============================================================================

def build_valve_task(joint_targets):
    """主臂阀门旋拧技能序列"""
    t = joint_targets
    p_move   = SKILL_STA_PARAMS[SkillType.MOVE]
    p_grasp  = SKILL_STA_PARAMS[SkillType.GRASP]
    p_rotate = SKILL_STA_PARAMS[SkillType.ROTATE]
    return [
        MoveSkill(t['valve_approach'], duration=3.0, sta_params=p_move,   name="L:移到手轮上方"),
        MoveSkill(t['valve_grasp'],    duration=2.5, sta_params=p_move,   name="L:下降到手轮"),
        GraspSkill(action='close', wait_time=1.0, gripper_closed=0.35,
                   sta_params=p_grasp, name="L:抓取手轮"),
        RotateSkill(angle=np.pi, duration=5.0, joint_index=5,
                    sta_params=p_rotate, name="L:旋拧阀门"),
        GraspSkill(action='open', wait_time=0.5,     sta_params=p_grasp,  name="L:释放手轮"),
        MoveSkill(t['valve_retract'],  duration=2.5, sta_params=p_move,   name="L:退回安全位"),
    ]


RRT_TOTAL_DURATION = 6.5


def _rrt_path_to_skills(rrt_path, label_prefix="R:RRT"):
    """将 RRT 路径节点序列转为 MoveSkill 列表, 按关节距离分配时长"""
    p_move = SKILL_STA_PARAMS[SkillType.MOVE]
    if len(rrt_path) < 2:
        return []
    seg_dists = [np.linalg.norm(rrt_path[i + 1] - rrt_path[i])
                 for i in range(len(rrt_path) - 1)]
    total_dist = sum(seg_dists)
    if total_dist < 1e-6:
        return []
    skills = []
    for i, d in enumerate(seg_dists):
        dur = max(0.8, RRT_TOTAL_DURATION * d / total_dist)
        skills.append(MoveSkill(
            rrt_path[i + 1], duration=dur, sta_params=p_move,
            name=f"{label_prefix}[{i+1}/{len(seg_dists)}]"))
    return skills


def build_shade_skills(shade_targets):
    """从臂遮光技能序列 (Phase 1), lift→block 使用 RRT 路径"""
    t = shade_targets
    p_move  = SKILL_STA_PARAMS[SkillType.MOVE]
    p_grasp = SKILL_STA_PARAMS[SkillType.GRASP]
    skills = [
        MoveSkill(t['shade_approach'], duration=3.0, sta_params=p_move,
                  name="R:移到手柄上方"),
        MoveSkill(t['shade_grasp'],    duration=2.5, sta_params=p_move,
                  name="R:下降到手柄"),
        GraspSkill(action='close', wait_time=1.2,    sta_params=p_grasp,
                  name="R:夹取手柄"),
        MoveSkill(t['shade_lift'],     duration=2.5, sta_params=p_move,
                  name="R:提起遮光板"),
    ]
    skills.extend(_rrt_path_to_skills(t['rrt_path_forward'], "R:避碰前进"))
    return skills


def build_unshade_skills(shade_targets, initial_q):
    """从臂放回遮光板技能序列 (Phase 4), block→lift 使用 RRT 反向路径"""
    t = shade_targets
    p_move  = SKILL_STA_PARAMS[SkillType.MOVE]
    p_grasp = SKILL_STA_PARAMS[SkillType.GRASP]
    skills = list(_rrt_path_to_skills(t['rrt_path_reverse'], "R:避碰返回"))
    skills.extend([
        MoveSkill(t['shade_grasp'],    duration=2.5, sta_params=p_move,
                  name="R:下降放置"),
        GraspSkill(action='open', wait_time=0.8,     sta_params=p_grasp,
                  name="R:释放遮光板"),
        MoveSkill(t['shade_approach'], duration=2.0, sta_params=p_move,
                  name="R:退回安全位"),
        MoveSkill(np.array(initial_q), duration=3.0, sta_params=p_move,
                  name="R:回到初始姿态"),
    ])
    return skills


def build_l_return_skills(initial_q):
    """主臂复位技能序列 (Phase 3)"""
    p_move = SKILL_STA_PARAMS[SkillType.MOVE]
    return [
        MoveSkill(np.array(initial_q), duration=3.0, sta_params=p_move,
                  name="L:回到初始姿态"),
    ]


# ============================================================================
# 单臂控制状态
# ============================================================================



# ============================================================================
# 强光遮挡协同任务控制器
# ============================================================================

class GlareShieldTask:

    CART_POS_TOL = 0.05
    CART_YAW_TOL = 0.1

    def __init__(self, model_path: str, scene_cfg: dict = None):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.dof = 6
        world_state = build_world_state(self.model, self.data, scene_path=model_path)
        self.scene_cfg = scene_cfg or glare_scene_config(world_state)

        def _act_id(name):
            return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        def _sens_id(name):
            return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)

        self._L_ctrl_start  = _act_id("shoulder_pan")
        self._R_ctrl_start  = _act_id("shoulder_pan_R")
        self._cart_x_ctrl   = _act_id("cart_x_actuator")
        self._cart_y_ctrl   = _act_id("cart_y_actuator")
        self._cart_yaw_ctrl = _act_id("cart_yaw_actuator")

        self._cart_x_sens   = _sens_id("cart_x_pos")
        self._cart_y_sens   = _sens_id("cart_y_pos")
        self._cart_yaw_sens = _sens_id("cart_yaw_pos")
        self._valve1_sens   = _sens_id("valve_angle_1")

        # ---------- 双臂控制器 ----------
        self.arm_L = ArmController(
            self.model, self.data, "主臂(L)",
            "shoulder_pan_joint", "fingers_actuator", sensor_offset=0)
        self.arm_R = ArmController(
            self.model, self.data, "从臂(R)",
            "shoulder_pan_joint_R", "fingers_actuator_R", sensor_offset=6)

        # ---------- 预计算 IK 目标 ----------
        cfg = self.scene_cfg
        print("预计算主臂(L) 阀门操作目标...")
        self.valve_targets_L = precompute_valve_targets(
            cfg['cart_target'], cfg['arm_offset_L'], cfg['valve_1_hw'],
            self.arm_L.robot, arm_yaw=0.0)

        print("\n计算 L 臂障碍物 (用于 RRT 避碰)...")
        l_obstacles = get_l_arm_obstacles(
            cfg['arm_offset_L'], cfg['arm_offset_R'],
            cfg['arm_yaw_R'], self.arm_L.initial_q)
        print(f"  L 臂障碍物: {len(l_obstacles)} 个 Capsule")

        print("\n预计算从臂(R) 遮光操作目标...")
        self.shade_targets_R = precompute_shade_targets(
            cfg['cart_target'], cfg['arm_offset_R'],
            cfg['shade_handle_pos'], cfg['shade_block_pos'],
            self.arm_R.robot, l_obstacles, arm_yaw=cfg['arm_yaw_R'])

        # ---------- 运行时状态 ----------
        self.phase = PHASE_NAV
        self._nav_start_time = None
        self._nav_start_pos = None
        self._current_nav_target = None
        self._nav_duration = 0.0
        self._phase_wait_time = None
        self.R_hold_q = None
        self.valve_history = []

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

    def _do_nav_interpolation(self, t):
        elapsed = t - self._nav_start_time
        alpha = min(elapsed / self._nav_duration, 1.0)
        interp = self._nav_start_pos + alpha * (self._current_nav_target - self._nav_start_pos)
        self._set_cart_target(interp[0], interp[1], 0.0)

    def _nav_duration_elapsed(self, t):
        return (t - self._nav_start_time) >= self._nav_duration

    # ------------------------------------------------------------------
    # 臂传感器
    # ------------------------------------------------------------------

    def _get_arm_sensor(self, arm):
        s = arm.sensor_offset
        return self.data.sensordata[s:s + self.dof].copy()

    # ------------------------------------------------------------------
    # 阶段初始化
    # ------------------------------------------------------------------

    def _init_shade(self):
        """Phase 1: 从臂开始遮光操作"""
        sensor = self._get_arm_sensor(self.arm_R)
        self.arm_R.robot.set_joint(sensor)
        self.arm_R.ctx.gripper_target = 0.0
        self.arm_R._prev_desired = sensor.copy()
        self.shade_skills = build_shade_skills(self.shade_targets_R)
        self.arm_R.executor = SkillExecutor(
            self.shade_skills, self.arm_R.sta_controllers, self.dof)

    def _init_valve(self):
        """Phase 2: 主臂开始阀门操作"""
        self.R_hold_q = self._get_arm_sensor(self.arm_R).copy()
        sensor = self._get_arm_sensor(self.arm_L)
        self.arm_L.robot.set_joint(sensor)
        self.arm_L.ctx.gripper_target = 0.0
        self.arm_L._prev_desired = sensor.copy()
        self.valve_skills = build_valve_task(self.valve_targets_L)
        self.arm_L.executor = SkillExecutor(
            self.valve_skills, self.arm_L.sta_controllers, self.dof)

    def _init_l_return(self):
        """Phase 3: 主臂复位"""
        sensor = self._get_arm_sensor(self.arm_L)
        self.arm_L.robot.set_joint(sensor)
        self.arm_L._prev_desired = sensor.copy()
        self.l_return_skills = build_l_return_skills(self.arm_L.initial_q)
        self.arm_L.executor = SkillExecutor(
            self.l_return_skills, self.arm_L.sta_controllers, self.dof)

    def _init_unshade(self):
        """Phase 4: 从臂放回遮光板"""
        sensor = self._get_arm_sensor(self.arm_R)
        self.arm_R.robot.set_joint(sensor)
        self.arm_R._prev_desired = sensor.copy()
        self.unshade_skills = build_unshade_skills(
            self.shade_targets_R, self.arm_R.initial_q)
        self.arm_R.executor = SkillExecutor(
            self.unshade_skills, self.arm_R.sta_controllers, self.dof)

    # ------------------------------------------------------------------
    # 阶段管理
    # ------------------------------------------------------------------

    def _update_phase(self, t):
        if self.phase == PHASE_NAV:
            target = self.scene_cfg['cart_target']
            if self._nav_start_time is None:
                self._start_navigation(target, t)
                print(f"[{t:.2f}s] Phase 0 (NAV): 小车导航到阀门附近 "
                      f"(预计 {self._nav_duration:.1f}s)")
            self._do_nav_interpolation(t)
            if self._nav_duration_elapsed(t) and self._cart_reached(target):
                print(f"[{t:.2f}s] 到达阀门区域 → Phase 1 (SHADE)")
                self.phase = PHASE_SHADE
                self._init_shade()

        elif self.phase == PHASE_SHADE:
            self.arm_R.ctx.current_time = t
            done = self.arm_R.executor.update(self.arm_R.ctx)
            if done:
                print(f"[{t:.2f}s] 遮光就位 → Phase 2 (VALVE)")
                self.phase = PHASE_VALVE
                self._init_valve()

        elif self.phase == PHASE_VALVE:
            self.arm_L.ctx.current_time = t
            done = self.arm_L.executor.update(self.arm_L.ctx)
            if done:
                if self._phase_wait_time is None:
                    self._phase_wait_time = t
                    v1 = np.degrees(self.data.sensordata[self._valve1_sens])
                    print(f"[{t:.2f}s] 阀门操作完成! 旋转: {v1:.1f}°")
                if t > self._phase_wait_time + 1.0:
                    print(f"[{t:.2f}s] → Phase 3 (L_RETURN)")
                    self.phase = PHASE_L_RETURN
                    self._init_l_return()
                    self._phase_wait_time = None

        elif self.phase == PHASE_L_RETURN:
            self.arm_L.ctx.current_time = t
            done = self.arm_L.executor.update(self.arm_L.ctx)
            if done:
                if self._phase_wait_time is None:
                    self._phase_wait_time = t
                    print(f"[{t:.2f}s] 主臂已复位")
                if t > self._phase_wait_time + 0.5:
                    print(f"[{t:.2f}s] → Phase 4 (UNSHADE)")
                    self.phase = PHASE_UNSHADE
                    self._init_unshade()
                    self._phase_wait_time = None

        elif self.phase == PHASE_UNSHADE:
            self.arm_R.ctx.current_time = t
            done = self.arm_R.executor.update(self.arm_R.ctx)
            if done:
                if self._phase_wait_time is None:
                    self._phase_wait_time = t
                    print(f"[{t:.2f}s] 遮光板已放回, 全部完成!")
                if t > self._phase_wait_time + 2.0:
                    self.phase = PHASE_DONE

    # ------------------------------------------------------------------
    # 单步控制
    # ------------------------------------------------------------------

    def _apply_arm_control(self, arm, ctrl_start, sensor_data, real_vel, desired):
        ctrl = arm.compute_control(desired, sensor_data, real_vel, self.model.opt.timestep)
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

        prev_L = self._get_arm_sensor(self.arm_L)
        prev_R = self._get_arm_sensor(self.arm_R)
        ts = self.model.opt.timestep

        def step_loop():
            nonlocal prev_L, prev_R

            while self.data.time <= total_time:
                t = self.data.time
                self._update_phase(t)
                if self.phase == PHASE_DONE:
                    break

                # --- 主臂(L) ---
                sensor_L = self._get_arm_sensor(self.arm_L)
                vel_L = (sensor_L - prev_L) / ts
                prev_L = sensor_L.copy()

                if self.phase in (PHASE_VALVE, PHASE_L_RETURN):
                    self.arm_L.ctx.current_time = t
                    desired_L = self.arm_L.executor.get_desired_position(self.arm_L.ctx)
                else:
                    desired_L = self.arm_L.initial_q

                self._apply_arm_control(
                    self.arm_L, self._L_ctrl_start, sensor_L, vel_L, desired_L)

                # --- 从臂(R) ---
                sensor_R = self._get_arm_sensor(self.arm_R)
                vel_R = (sensor_R - prev_R) / ts
                prev_R = sensor_R.copy()

                if self.phase == PHASE_SHADE:
                    self.arm_R.ctx.current_time = t
                    desired_R = self.arm_R.executor.get_desired_position(self.arm_R.ctx)
                elif self.phase in (PHASE_VALVE, PHASE_L_RETURN):
                    desired_R = self.R_hold_q
                elif self.phase == PHASE_UNSHADE:
                    self.arm_R.ctx.current_time = t
                    desired_R = self.arm_R.executor.get_desired_position(self.arm_R.ctx)
                else:
                    desired_R = self.arm_R.initial_q

                self._apply_arm_control(
                    self.arm_R, self._R_ctrl_start, sensor_R, vel_R, desired_R)

                self.valve_history.append(self.data.sensordata[self._valve1_sens])
                mujoco.mj_step(self.model, self.data)
                yield

        if use_viewer:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                viewer.cam.lookat[:] = [0.0, 0.3, 0.6]
                viewer.cam.distance = 4.0
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

    def run(self, total_time=90.0):
        print("=" * 60)
        print("  Glare Shield Cooperative Task")
        print("  小车导航 → R遮光 → L旋拧阀门 → L复位 → R放回遮光板")
        print("=" * 60 + "\n")

        self.phase = PHASE_NAV
        self._nav_start_time = None
        self._phase_wait_time = None
        self.R_hold_q = None
        self.valve_history = []

        self._run_simulation(total_time, use_viewer=True)
        self._print_summary()

    def _print_summary(self):
        print("\n" + "=" * 60)
        print("  Glare Shield Task Summary")
        print("=" * 60)
        if self.valve_history:
            print(f"  阀门最终旋转: {np.degrees(self.valve_history[-1]):.1f}°")
        print(f"  最终阶段: {PHASE_NAMES.get(self.phase, '?')}")
        print()

def main():
    scene_path = str(bundled_scene('scene5_glare'))

    print(f"Loading scene: {scene_path}")
    task = GlareShieldTask(scene_path)
    task.run(total_time=90.0)


if __name__ == '__main__':
    main()
