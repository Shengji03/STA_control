"""
长时序装配任务控制器 (技能库架构)

架构:
    场景配置 (坐标) + 技能库 (行为) → 技能序列 (任务) → 技能执行器 (调度)
    控制层: STA 超螺旋滑模 + DDPG 补偿
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from spatialmath import SE3, SO3

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

from src.controller import STAController, DDPGAgent
from src.perception import build_world_state
from src.robot import UR5e
from src.skills import (
    SkillType, SkillContext, SkillExecutor, DEFAULT_STA_PARAMS, SKILL_STA_PARAMS,
    MoveSkill, GraspSkill, InsertSkill,
    ExtractSkill, RotateSkill, TranslateSkill,
)


# ============================================================================
# 场景配置
# ============================================================================

def default_scene_config(world_state=None):
    """
    默认场景坐标配置 (世界坐标系)

    修改场景布局时只需修改此函数, 无需改动技能或控制逻辑。
    """
    if world_state is not None:
        objects = world_state["objects"]
        block_pos = np.array(objects['block']['pose']['pos'])
        block_target_xy = np.array(objects['zone_block_target']['pose']['pos'][:2])
        place_b_xy = np.array(objects['zone_place_b']['pose']['pos'][:2])
        pin_a_pos = np.array(objects['pin_a']['pose']['pos'])
        return {
            'robot_base':    np.array(objects['ur5e_base']['pose']['pos']),
            'pin_a':         pin_a_pos,
            'pin_b_place':   np.array([place_b_xy[0], place_b_xy[1], pin_a_pos[2]]),
            'block':         block_pos,
            'block_target':  np.array([block_target_xy[0], block_target_xy[1], block_pos[2]]),
            'socket_a':      np.array(objects['socket_a']['pose']['pos']),
            'socket_b':      np.array(objects['socket_b']['pose']['pos']),
        }

    return {
        'robot_base':    np.array([0.8, 0.6, 0.745]),
        'pin_a':         np.array([1.45, 0.4, 0.78]),
        'pin_b_place':   np.array([1.45, 0.4, 0.78]),
        'block':         np.array([1.30, 0.4, 0.78]),
        'block_target':  np.array([1.30, 0.8, 0.78]),
        'socket_a':      np.array([1.45, 0.8, 0.74]),
        'socket_b':      np.array([1.15, 0.8, 0.74]),
    }


# ============================================================================
# 目标关节角预计算
# ============================================================================

def precompute_joint_targets(scene_cfg, robot, tool_offset=0.155):
    """
    根据场景坐标 + 机器人逆运动学, 预计算所有目标位置的关节角度

    Args:
        scene_cfg:   场景配置字典 (世界坐标)
        robot:       机器人对象 (提供 ikine)
        tool_offset: 工具偏移量

    Returns:
        dict: {名称: 关节角数组}
    """
    print("预计算目标关节角度...")

    base = scene_cfg['robot_base']
    down_orientation = SO3.Rx(np.pi)

    # 世界坐标 → 机器人坐标
    pin_a = scene_cfg['pin_a'] - base
    pin_b_place = scene_cfg['pin_b_place'] - base
    block = scene_cfg['block'] - base
    block_target = scene_cfg['block_target'] - base
    socket_a = scene_cfg['socket_a'] - base
    socket_b = scene_cfg['socket_b'] - base

    # 高度参数
    pin_center_z = 0.035
    pin_grasp_h = pin_center_z + tool_offset
    pin_approach_h = pin_grasp_h + 0.12
    pin_lift_h = pin_grasp_h + 0.15

    block_center_z = 0.035
    block_grasp_h = block_center_z + tool_offset
    block_approach_h = block_grasp_h + 0.12

    socket_target_z = 0.05
    insert_h = socket_target_z + tool_offset

    extract_center_z = 0.085
    extract_grasp_h = extract_center_z + tool_offset

    # 目标笛卡尔位置
    cart_targets = {
        'pin_a_above':      np.array([pin_a[0], pin_a[1], pin_approach_h]),
        'pin_a_grasp':      np.array([pin_a[0], pin_a[1], pin_grasp_h]),
        'pin_a_lift':       np.array([pin_a[0], pin_a[1], pin_lift_h]),
        'socket_a_above':   np.array([socket_a[0], socket_a[1], pin_lift_h]),
        'socket_a_insert':  np.array([socket_a[0], socket_a[1], insert_h]),
        'socket_a_retract': np.array([socket_a[0], socket_a[1], pin_approach_h]),
        'block_above':      np.array([block[0], block[1], block_approach_h]),
        'block_grasp':      np.array([block[0], block[1], block_grasp_h]),
        'block_translate':  np.array([block_target[0], block_target[1], block_grasp_h]),
        'block_retract':    np.array([block_target[0], block_target[1], block_approach_h]),
        'socket_b_above':   np.array([socket_b[0], socket_b[1], pin_lift_h]),
        'socket_b_grasp':   np.array([socket_b[0], socket_b[1], extract_grasp_h]),
        'place_b_above':    np.array([pin_b_place[0], pin_b_place[1], pin_lift_h]),
        'place_b_put':      np.array([pin_b_place[0], pin_b_place[1], pin_grasp_h]),
        'place_b_retract':  np.array([pin_b_place[0], pin_b_place[1], pin_approach_h]),
    }

    # IK 求解
    init_q = [0, 0, np.pi / 2, 0, -np.pi / 2, 0]
    robot.set_joint(init_q)
    robot.setRobotConfig(init_q)

    joint_targets = {}
    for name, pos in cart_targets.items():
        T_target = SE3.Rt(down_orientation.R, pos)
        q = robot.ikine(T_target)
        if len(q) > 0:
            joint_targets[name] = np.array(q)
            robot.set_joint(q)
            print(f"  {name}: pos={pos}, q(deg)={np.degrees(q)[:3]}...")
        else:
            print(f"  {name}: IK失败! pos={pos}")
            joint_targets[name] = np.array(init_q)

    return joint_targets


# ============================================================================
# 任务构建器 (技能序列)
# ============================================================================

def build_assembly_task(joint_targets, sta_params=None):
    """
    构建长时序装配任务的技能序列

    3 个阶段, 23 个技能, 每个技能使用对应类型的 STA 参数.

    Args:
        joint_targets: 预计算的目标关节角字典
        sta_params:    覆盖参数 (如提供则所有技能统一使用, 否则按技能类型区分)

    Returns:
        list[BaseSkill]: 技能序列
    """
    t = joint_targets
    p_move     = sta_params or SKILL_STA_PARAMS[SkillType.MOVE]
    p_grasp    = sta_params or SKILL_STA_PARAMS[SkillType.GRASP]
    p_insert   = sta_params or SKILL_STA_PARAMS[SkillType.INSERT]
    p_extract  = sta_params or SKILL_STA_PARAMS[SkillType.EXTRACT]
    p_rotate   = sta_params or SKILL_STA_PARAMS[SkillType.ROTATE]
    p_translate = sta_params or SKILL_STA_PARAMS[SkillType.TRANSLATE]

    skills = [
        # ===== Phase 1: 插销A -> 凹槽A =====
        MoveSkill(t['pin_a_above'],       duration=3.0, sta_params=p_move,    name="移到插销A上方"),
        MoveSkill(t['pin_a_grasp'],       duration=2.0, sta_params=p_move,    name="下降到插销A"),
        GraspSkill(action='close',                      sta_params=p_grasp,   name="抓取插销A"),
        MoveSkill(t['pin_a_lift'],        duration=2.0, sta_params=p_move,    name="提起插销A"),
        MoveSkill(t['socket_a_above'],    duration=3.0, sta_params=p_move,    name="移到凹槽A上方"),
        InsertSkill(t['socket_a_insert'], duration=4.0, sta_params=p_insert,  name="插入凹槽A"),
        RotateSkill(angle=2 * np.pi,      duration=5.0, sta_params=p_rotate,  name="旋拧A"),
        GraspSkill(action='open', wait_time=0.5,        sta_params=p_grasp,   name="释放A"),
        MoveSkill(t['socket_a_retract'],  duration=2.0, sta_params=p_move,    name="退出凹槽A"),

        # ===== Phase 2: 平移物块 =====
        MoveSkill(t['block_above'],         duration=3.0, sta_params=p_move,      name="移到物块上方"),
        MoveSkill(t['block_grasp'],         duration=2.0, sta_params=p_move,      name="下降到物块"),
        GraspSkill(action='close',                        sta_params=p_grasp,     name="抓取物块"),
        TranslateSkill(t['block_translate'],duration=4.0, sta_params=p_translate, name="平移物块"),
        GraspSkill(action='open', wait_time=0.5,          sta_params=p_grasp,     name="释放物块"),
        MoveSkill(t['block_retract'],       duration=2.0, sta_params=p_move,      name="退出物块区"),

        # ===== Phase 3: 拔出插销B -> 放置 =====
        MoveSkill(t['socket_b_above'],    duration=3.0, sta_params=p_move,    name="移到凹槽B上方"),
        MoveSkill(t['socket_b_grasp'],    duration=2.0, sta_params=p_move,    name="下降到插销B"),
        GraspSkill(action='close',                      sta_params=p_grasp,   name="抓取插销B"),
        ExtractSkill(t['socket_b_above'], duration=3.0, sta_params=p_extract, name="拔出插销B"),
        MoveSkill(t['place_b_above'],     duration=3.0, sta_params=p_move,    name="移到放置位上方"),
        MoveSkill(t['place_b_put'],       duration=2.0, sta_params=p_move,    name="下降放置"),
        GraspSkill(action='open', wait_time=0.5,        sta_params=p_grasp,   name="释放插销B"),
        MoveSkill(t['place_b_retract'],   duration=2.0, sta_params=p_move,    name="退出放置区"),
    ]

    return skills


# ============================================================================
# 长时序装配任务控制器
# ============================================================================

class LongSequenceTask:
    """
    长时序装配任务控制器 (STA 滑模控制 + DDPG 补偿 + 技能库架构)

    三层分离:
        - 场景配置: 定义物体坐标 (default_scene_config)
        - 任务构建: 组合技能序列 (build_assembly_task)
        - 控制执行: STA + DDPG + SkillExecutor
    """

    def __init__(self, model_path: str, enable_ddpg: bool = True,
                 disturbance_cfg: dict = None, scene_cfg: dict = None,
                 sta_params: dict = None):
        # ========== 扰动配置 ==========
        if disturbance_cfg is None:
            self.disturbance_cfg = {
                'enable': False,
                'sensor_noise_std': 0.0,
                'friction_bias': 0.0,
                'random_walk_std': 0.0,
                'sinusoidal_amp': 0.0,
                'sinusoidal_freqs': [3.0, 3.7, 4.3, 5.0, 5.5, 6.0],
                'coulomb_friction': 0.0,
            }
        else:
            self.disturbance_cfg = disturbance_cfg

        self._joint_torque_scale = np.array([1.0, 1.2, 1.0, 0.4, 0.4, 0.3])
        self._random_walk_torque = np.zeros(6)

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        self.robot = UR5e()
        self.dof = self.robot.dof

        # ========== 场景 & STA 参数 ==========
        world_state = build_world_state(self.model, self.data, scene_path=model_path)
        self.scene_cfg = scene_cfg or default_scene_config(world_state)
        self.sta_params = sta_params or DEFAULT_STA_PARAMS

        # ========== 预计算目标关节角 ==========
        self.joint_targets = precompute_joint_targets(
            self.scene_cfg, self.robot
        )
        self.initial_q = np.array([0, 0, np.pi / 2, 0, -np.pi / 2, 0])

        # ========== STA 控制器 ==========
        ts = self.model.opt.timestep
        self.sta_controllers = [
            STAController(
                self.sta_params['alpha'][0],
                self.sta_params['beta'][0],
                self.sta_params['lambda_s'][0],
                ts=ts
            ) for _ in range(self.dof)
        ]

        # ========== 技能序列 & 执行器 ==========
        self.skills = build_assembly_task(self.joint_targets, self.sta_params)
        self.executor = SkillExecutor(
            self.skills, self.sta_controllers, self.dof
        )
        self.ctx = SkillContext(self.model, self.data, self.robot, self.dof)

        self._print_skill_sequence(self.skills)

        # ========== DDPG ==========
        self.enable_ddpg = enable_ddpg
        if self.enable_ddpg:
            self._init_ddpg()

        # ========== 运行时状态 ==========
        self._prev_desired_pos = np.array(self.initial_q)
        self._initialized = False

        # ========== 数据记录 ==========
        self.time_history = []
        self.joint_desired_history = []
        self.joint_actual_history = []
        self.ddpg_reward_history = []
        self.ddpg_action_history = []
        self.control_sta_history = []
        self.control_total_history = []

    @staticmethod
    def _print_skill_sequence(skills):
        """打印技能序列"""
        print(f"\n技能序列已构建: 共 {len(skills)} 个技能")
        for i, s in enumerate(skills):
            print(f"  [{i + 1:>2}] {s}")
        print()

    # ------------------------------------------------------------------
    # DDPG 初始化
    # ------------------------------------------------------------------

    def _init_ddpg(self):
        """初始化 DDPG 补偿控制器 (18维状态, 6维动作)"""
        state_dim = self.dof * 3
        action_dim = self.dof
        action_bound = 2.0

        self.ddpg_agent = DDPGAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            action_bound=action_bound,
            actor_lr=1e-4,
            critic_lr=3e-4,
            gamma=0.95,
            tau=0.005,
            buffer_capacity=50000,
            batch_size=128,
            noise_sigma=0.15,
            noise_theta=0.15,
            warmup_steps=1000,
            hidden_dims=(256, 256, 128),
        )

        self._sta_output_scale = np.array([50.0, 80.0, 60.0, 25.0, 25.0, 25.0])
        self._prev_error = np.zeros(self.dof)
        self._prev_state = None
        self._prev_action = None
        self._prev_sta = np.zeros(self.dof)
        self._ddpg_step_count = 0
        self._ddpg_train_interval = 4

        print(f"[DDPG] 补偿控制器已初始化: state_dim={state_dim}, "
              f"action_dim={action_dim}, action_bound={action_bound}")

    # ------------------------------------------------------------------
    # Episode 管理
    # ------------------------------------------------------------------

    def reset_episode(self):
        """重置 episode (保留 DDPG 网络权重)"""
        self._prev_desired_pos = np.array(self.initial_q)
        self._initialized = False

        # 清空数据记录
        self.time_history = []
        self.joint_desired_history = []
        self.joint_actual_history = []
        self.ddpg_reward_history = []
        self.ddpg_action_history = []
        self.control_sta_history = []
        self.control_total_history = []

        # 重建技能序列 & 重置执行器
        self.skills = build_assembly_task(self.joint_targets, self.sta_params)
        self.executor = SkillExecutor(
            self.skills, self.sta_controllers, self.dof
        )

        # 重置 STA 控制器
        for ctrl in self.sta_controllers:
            ctrl.reset()

        # 重置 DDPG 临时状态 (保留网络权重和经验缓冲)
        if self.enable_ddpg:
            self._prev_error = np.zeros(self.dof)
            self._prev_state = None
            self._prev_action = None
            self._prev_sta = np.zeros(self.dof)
            self.ddpg_agent.reset_noise()

        self._random_walk_torque = np.zeros(self.dof)
        self.ctx.gripper_target = 0.0

    # ------------------------------------------------------------------
    # 任务更新 & 控制
    # ------------------------------------------------------------------

    def update(self, current_time):
        """更新任务 (委托给技能执行器)"""
        self.ctx.current_time = current_time

        if not self._initialized:
            sensor_data = self.data.sensordata[:self.dof].copy()
            self.robot.set_joint(sensor_data)
            print(f"Initial pose: {self.robot.get_cartesian().t}")
            self._initialized = True

        return self.executor.update(self.ctx)

    def get_desired_joint_position(self, current_time):
        """获取期望关节位置 (委托给技能执行器)"""
        self.ctx.current_time = current_time
        return self.executor.get_desired_position(self.ctx)

    def compute_control(self, desired_pos, sensor_data, real_vel):
        """计算控制量 (STA + DDPG 补偿)"""
        error_pos = desired_pos - sensor_data
        desired_vel = (desired_pos - self._prev_desired_pos) / self.model.opt.timestep
        error_vel = desired_vel - real_vel
        self._prev_desired_pos = desired_pos.copy()

        # STA 滑模控制
        ctrl_sta = np.zeros(self.dof)
        for i in range(self.dof):
            ctrl_sta[i] = self.sta_controllers[i].control(error_pos[i], error_vel[i])

        # DDPG 补偿
        if self.enable_ddpg:
            ctrl_ddpg = self._ddpg_step(error_pos, real_vel, ctrl_sta)
            ctrl_total = ctrl_sta + ctrl_ddpg
        else:
            ctrl_ddpg = np.zeros(self.dof)
            ctrl_total = ctrl_sta

        self.control_sta_history.append(ctrl_sta.copy())
        self.ddpg_action_history.append(ctrl_ddpg.copy())
        self.control_total_history.append(ctrl_total.copy())

        return ctrl_total

    def _ddpg_step(self, error_pos, real_vel, ctrl_sta):
        """DDPG 单步: 选择动作 + 存储经验 + 训练"""
        error_rate = (error_pos - self._prev_error) / self.model.opt.timestep
        sta_normalized = ctrl_sta / self._sta_output_scale
        current_state = np.concatenate([error_pos, error_rate, sta_normalized])

        if self._prev_state is not None and self._prev_action is not None:
            reward = self.ddpg_agent.compute_reward(
                error_pos, self._prev_error, self._prev_action
            )
            done = self.executor.is_all_complete
            self.ddpg_agent.store_transition(
                self._prev_state, self._prev_action,
                reward, current_state, done
            )
            self.ddpg_reward_history.append(reward)

        ctrl_ddpg = self.ddpg_agent.select_action(
            current_state,
            add_noise=self.ddpg_agent.training_enabled
        )

        self._ddpg_step_count += 1
        if self._ddpg_step_count % self._ddpg_train_interval == 0:
            self.ddpg_agent.train_step()

        self._prev_error = error_pos.copy()
        self._prev_state = current_state.copy()
        self._prev_action = ctrl_ddpg.copy()
        self._prev_sta = ctrl_sta.copy()

        return ctrl_ddpg

    # ------------------------------------------------------------------
    # 仿真主循环
    # ------------------------------------------------------------------

    def _run_simulation_loop(self, total_time, use_viewer=True,
                             train_ddpg=True, verbose=True):
        mujoco.mj_resetData(self.model, self.data)
        for i in range(self.dof):
            self.data.qpos[i] = self.initial_q[i]
        self.ctx.gripper_target = 0.0
        mujoco.mj_forward(self.model, self.data)

        sensor_data = self.data.sensordata[:self.dof].copy()
        real_pos = sensor_data.copy()
        real_pos_prev = real_pos.copy()

        log_interval = 1000
        step_count = 0

        def step_loop():
            nonlocal sensor_data, real_pos, real_pos_prev, step_count

            while self.data.time <= total_time:
                current_time = self.data.time

                task_complete = self.update(current_time)
                if task_complete:
                    if current_time > self.executor.completion_time + 2.0:
                        break

                desired_pos = self.get_desired_joint_position(current_time)

                sensor_data = self.data.sensordata[:self.dof].copy()

                dist_cfg = self.disturbance_cfg
                if dist_cfg.get('enable', False):
                    noise_std = dist_cfg.get('sensor_noise_std', 0.0)
                    if noise_std > 0:
                        sensor_data += noise_std * np.random.randn(self.dof)

                real_pos_prev = real_pos.copy()
                real_pos = sensor_data.copy()
                real_vel = (real_pos - real_pos_prev) / self.model.opt.timestep

                if self.enable_ddpg and not train_ddpg:
                    old_training = self.ddpg_agent.training_enabled
                    self.ddpg_agent.set_training(False)

                ctrl = self.compute_control(desired_pos, sensor_data, real_vel)

                if self.enable_ddpg and not train_ddpg:
                    self.ddpg_agent.set_training(old_training)

                if dist_cfg.get('enable', False):
                    bias = dist_cfg.get('friction_bias', 0.0)
                    if bias > 0:
                        ctrl += bias * self._joint_torque_scale * np.array(
                            [1, -1, 1, -1, 1, -1], dtype=np.float64
                        )

                    rw_std = dist_cfg.get('random_walk_std', 0.0)
                    if rw_std > 0:
                        self._random_walk_torque = (
                            0.9995 * self._random_walk_torque
                            + rw_std * self._joint_torque_scale
                            * np.random.randn(self.dof)
                        )
                        max_rw = max(bias, 0.3) * 3.0
                        self._random_walk_torque = np.clip(
                            self._random_walk_torque, -max_rw, max_rw
                        )
                        ctrl += self._random_walk_torque

                    sin_amp = dist_cfg.get('sinusoidal_amp', 0.0)
                    if sin_amp > 0:
                        sin_freqs = dist_cfg.get('sinusoidal_freqs',
                                                  [3.0, 3.7, 4.3, 5.0, 5.5, 6.0])
                        for i in range(self.dof):
                            ctrl[i] += (sin_amp * self._joint_torque_scale[i]
                                        * np.sin(2 * np.pi * sin_freqs[i] * current_time))

                    coulomb = dist_cfg.get('coulomb_friction', 0.0)
                    if coulomb > 0:
                        ctrl += (coulomb * self._joint_torque_scale
                                 * np.sign(real_vel))

                for i in range(self.dof):
                    self.data.ctrl[i] = ctrl[i]
                if len(self.data.ctrl) > self.dof:
                    self.data.ctrl[self.dof] = self.ctx.gripper_target

                self.time_history.append(current_time)
                self.joint_desired_history.append(desired_pos.copy())
                self.joint_actual_history.append(sensor_data.copy())

                step_count += 1
                if self.enable_ddpg and train_ddpg:
                    self.ddpg_agent.decay_noise(decay_rate=0.999995, min_sigma=0.02)
                    if verbose and step_count % log_interval == 0:
                        stats = self.ddpg_agent.get_stats()
                        avg_reward = np.mean(self.ddpg_reward_history[-log_interval:]) \
                            if self.ddpg_reward_history else 0
                        print(f"  [DDPG @ step {step_count}] "
                              f"buffer={stats['buffer_size']}, "
                              f"noise_σ={stats['noise_sigma']:.4f}, "
                              f"avg_reward={avg_reward:.4f}"
                              + (f", critic_loss={stats['recent_critic_loss']:.6f}"
                                 if 'recent_critic_loss' in stats else ""))

                mujoco.mj_step(self.model, self.data)
                yield

        if use_viewer:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                viewer.cam.lookat[:] = [1.3, 0.6, 0.8]
                viewer.cam.distance = 2.2
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
    # 运行接口
    # ------------------------------------------------------------------

    def run_training_episode(self, total_time=75.0, verbose=True):
        """运行一个训练 episode (无渲染)"""
        self.reset_episode()
        self._run_simulation_loop(total_time, use_viewer=False,
                                  train_ddpg=True, verbose=verbose)
        return self._compute_metrics(print_detail=verbose)

    def run(self, total_time=75.0, evaluate=False):
        """运行任务 (带可视化)"""
        print("=" * 60)
        print("  Long Sequence Assembly Task with STA + DDPG")
        print("  (技能库架构)")
        print("=" * 60)
        if self.enable_ddpg:
            if evaluate:
                print("[模式] STA + DDPG 补偿 (评估模式, 无探索噪声)")
            else:
                print("[模式] STA + DDPG 补偿 (训练模式)")
        else:
            print("[模式] 仅 STA 滑模控制器 (DDPG 已禁用)")
        print()

        self.reset_episode()
        self._run_simulation_loop(total_time, use_viewer=True,
                                  train_ddpg=(not evaluate), verbose=True)
        return self._compute_metrics(print_detail=True)

    # ------------------------------------------------------------------
    # 评价指标
    # ------------------------------------------------------------------

    def _compute_metrics(self, print_detail=True):
        iae_value = 0.0
        ise_value = 0.0
        f_value = 0.0

        if print_detail:
            print("\n" + "=" * 60)
            print("  Task Execution Summary")
            print("=" * 60)

        if self.joint_desired_history and self.joint_actual_history:
            desired_arr = np.array(self.joint_desired_history)
            actual_arr = np.array(self.joint_actual_history)
            ts = self.model.opt.timestep

            errors = desired_arr - actual_arr
            abs_errors = np.abs(errors)

            mae = np.mean(abs_errors, axis=0)
            if print_detail:
                print(f"各关节平均绝对跟踪误差 (rad):")
                for i in range(self.dof):
                    print(f"  Joint {i}: {mae[i]:.6f} rad ({np.degrees(mae[i]):.4f} deg)")
                print(f"  总平均误差: {np.mean(mae):.6f} rad ({np.degrees(np.mean(mae)):.4f} deg)")

            iae_value = np.sum(np.sum(abs_errors, axis=1)) * ts
            ise_value = np.sum(np.sum(errors ** 2, axis=1)) * ts
            f_value = 0.5 * iae_value + 0.5 * ise_value

            if print_detail:
                print(f"\n[论文评价指标]")
                print(f"  IAE  (绝对误差积分):  {iae_value:.6f}")
                print(f"  ISE  (平方误差积分):  {ise_value:.6f}")
                print(f"  F    (综合指标):      {f_value:.6f}")

        if print_detail and self.enable_ddpg:
            stats = self.ddpg_agent.get_stats()
            print(f"\n[DDPG 训练统计]")
            print(f"  总训练步数:     {stats['total_steps']}")
            print(f"  经验缓冲区:     {stats['buffer_size']}")
            print(f"  最终噪声 σ:     {stats['noise_sigma']:.4f}")
            if 'recent_critic_loss' in stats:
                print(f"  最近 Critic Loss: {stats['recent_critic_loss']:.6f}")
            if self.ddpg_reward_history:
                print(f"  平均奖励:       {np.mean(self.ddpg_reward_history):.4f}")
            if self.ddpg_action_history:
                ddpg_actions = np.array(self.ddpg_action_history)
                print(f"  补偿力矩 RMS:   {np.sqrt(np.mean(ddpg_actions**2)):.4f} N·m")

        if print_detail:
            print()

        return {
            'time': self.time_history,
            'desired': self.joint_desired_history,
            'actual': self.joint_actual_history,
            'iae': iae_value,
            'ise': ise_value,
            'f_value': f_value,
            'ddpg_actions': self.ddpg_action_history if self.ddpg_action_history else None,
            'sta_actions': self.control_sta_history if self.control_sta_history else None,
        }


# ============================================================================
# 对比分析工具
# ============================================================================

def print_comparison(result_sta, result_ddpg, label=""):
    print("\n" + "=" * 60)
    if label:
        print(f"  对比结果: {label}")
    else:
        print("  对比结果 ")
    print("=" * 60)
    print(f"  {'指标':<20} {'STA-only':>12} {'STA+DDPG':>12} {'改善率':>10}")
    print("  " + "-" * 54)

    for name, key in [('IAE (绝对误差积分)', 'iae'),
                      ('ISE (平方误差积分)', 'ise'),
                      ('F   (综合指标)',     'f_value')]:
        v_sta = result_sta[key]
        v_ddpg = result_ddpg[key]
        if v_sta > 1e-12:
            improvement = (v_sta - v_ddpg) / v_sta * 100
            print(f"  {name:<18} {v_sta:>12.6f} {v_ddpg:>12.6f} {improvement:>+9.2f}%")
        else:
            print(f"  {name:<18} {v_sta:>12.6f} {v_ddpg:>12.6f} {'N/A':>10}")
    print("=" * 60)


def plot_comparison(result_sta, result_ddpg, train_f_history=None, label=""):
    from scipy.ndimage import uniform_filter1d

    time_sta = np.array(result_sta['time'])
    desired_sta = np.array(result_sta['desired'])
    actual_sta = np.array(result_sta['actual'])
    time_ddpg = np.array(result_ddpg['time'])
    desired_ddpg = np.array(result_ddpg['desired'])
    actual_ddpg = np.array(result_ddpg['actual'])

    error_sta = desired_sta - actual_sta
    error_ddpg = desired_ddpg - actual_ddpg
    dof = error_sta.shape[1]
    joint_names = [f'Joint {i}' for i in range(dof)]

    n_ds = 400
    idx_sta = np.linspace(0, len(time_sta) - 1, n_ds, dtype=int)
    idx_ddpg = np.linspace(0, len(time_ddpg) - 1, n_ds, dtype=int)
    smooth_win = max(1, len(time_sta) // n_ds)

    fig1, axes1 = plt.subplots(3, 2, figsize=(16, 10), sharex=True)
    fig1.suptitle(f'各关节跟踪误差对比\n{label}', fontsize=14)
    for i in range(dof):
        ax = axes1[i // 2, i % 2]
        err_sta_s = uniform_filter1d(np.degrees(error_sta[:, i]),
                                     size=smooth_win, mode='nearest')
        err_ddpg_s = uniform_filter1d(np.degrees(error_ddpg[:, i]),
                                      size=smooth_win, mode='nearest')
        ax.plot(time_sta[idx_sta], err_sta_s[idx_sta],
                color='#E74C3C', alpha=0.8, linewidth=1.2, label='STA-only')
        ax.plot(time_ddpg[idx_ddpg], err_ddpg_s[idx_ddpg],
                color='#2E86C1', alpha=0.8, linewidth=1.2, label='STA+DDPG')
        ax.set_ylabel(f'{joint_names[i]} (deg)', fontsize=9)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.5)
    axes1[2, 0].set_xlabel('Time (s)')
    axes1[2, 1].set_xlabel('Time (s)')
    fig1.tight_layout()

    total_err_sta = np.degrees(np.sum(np.abs(error_sta), axis=1))
    total_err_ddpg = np.degrees(np.sum(np.abs(error_ddpg), axis=1))

    def downsample_smooth(t, y, n_points=300, window=51):
        y_smooth = uniform_filter1d(y, size=window, mode='nearest')
        idx = np.linspace(0, len(t) - 1, n_points, dtype=int)
        return t[idx], y_smooth[idx]

    t_sta_ds, err_sta_ds = downsample_smooth(time_sta, total_err_sta)
    t_ddpg_ds, err_ddpg_ds = downsample_smooth(time_ddpg, total_err_ddpg)

    fig2, ax2 = plt.subplots(figsize=(14, 5))
    ax2.fill_between(time_sta, 0, total_err_sta, color='#E74C3C', alpha=0.08)
    ax2.fill_between(time_ddpg, 0, total_err_ddpg, color='#2E86C1', alpha=0.08)
    ax2.plot(t_sta_ds, err_sta_ds, '-o', color='#E74C3C',
             linewidth=1.8, markersize=2, alpha=0.9, label='STA-only')
    ax2.plot(t_ddpg_ds, err_ddpg_ds, '-s', color='#2E86C1',
             linewidth=1.8, markersize=2, alpha=0.9, label='STA+DDPG')
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('Σ|error| (deg)', fontsize=11)
    ax2.set_title(f'总跟踪误差对比\n{label}', fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    metrics = ['IAE', 'ISE', 'F']
    keys = ['iae', 'ise', 'f_value']
    vals_sta = [result_sta[k] for k in keys]
    vals_ddpg = [result_ddpg[k] for k in keys]
    x = np.arange(len(metrics))
    width = 0.32
    bars1 = ax3.bar(x - width/2, vals_sta, width, label='STA-only',
                    color='#E74C3C', alpha=0.85, edgecolor='black', linewidth=0.5)
    bars2 = ax3.bar(x + width/2, vals_ddpg, width, label='STA+DDPG',
                    color='#2E86C1', alpha=0.85, edgecolor='black', linewidth=0.5)
    for bar in bars1:
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=8)
    for i, k in enumerate(keys):
        v_s, v_d = result_sta[k], result_ddpg[k]
        if v_s > 1e-12:
            imp = (v_s - v_d) / v_s * 100
            color = '#27AE60' if imp > 0 else '#C0392B'
            ax3.annotate(f'{imp:+.2f}%',
                         xy=(x[i] + width/2, max(v_s, v_d)),
                         xytext=(0, 18), textcoords='offset points',
                         ha='center', fontsize=9, fontweight='bold', color=color,
                         arrowprops=dict(arrowstyle='->', color=color, lw=1.2))
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics, fontsize=11)
    ax3.set_ylabel('Value (越小越好)')
    ax3.set_title(f'性能指标对比\n{label}', fontsize=13)
    ax3.legend(fontsize=10)
    ax3.grid(True, axis='y', alpha=0.3)
    fig3.tight_layout()

    if train_f_history:
        fig4, ax4 = plt.subplots(figsize=(10, 4))
        ax4.plot(range(1, len(train_f_history)+1), train_f_history,
                 '-o', color='#8E44AD', linewidth=1.5, markersize=4)
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('F value')
        ax4.set_title('DDPG 训练过程 F 值变化')
        ax4.grid(True, alpha=0.3)
        fig4.tight_layout()

    plt.show()


# ============================================================================
# 主函数
# ============================================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    scene_path = os.path.join(script_dir, '..', 'src', 'assets', 'scenes', 'scene3.xml')
    scene_path = os.path.normpath(scene_path)

    # DDPG 模型保存路径
    model_dir = os.path.join(script_dir, 'saved_models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'ddpg_model.npz')

    print(f"Loading scene: {scene_path}")

    total_time = 75.0

    disturbance_cfg = {
        'enable': True,
        'sensor_noise_std': 0.0,
        'friction_bias': 0.0,
        'random_walk_std': 0.0,
        'sinusoidal_amp': 0.8,
        'sinusoidal_freqs': [3.0, 3.7, 4.3, 5.0, 5.5, 6.0],
        'coulomb_friction': 0.4,
    }

    mode = 'train_and_compare'
    num_train_episodes = 5

    if mode == 'train_and_compare':
        # 阶段 1: STA-only 基线
        print("\n" + "#" * 60)
        print("  阶段 1: STA-only 基线 (有扰动)")
        print("#" * 60)
        task_sta = LongSequenceTask(scene_path, enable_ddpg=False,
                                    disturbance_cfg=disturbance_cfg)
        result_sta = task_sta.run(total_time=total_time)

        # 阶段 2: DDPG 训练
        print("\n" + "#" * 60)
        print(f"  阶段 2: DDPG 训练 ({num_train_episodes} episodes, 无渲染)")
        print("#" * 60)
        task_ddpg = LongSequenceTask(scene_path, enable_ddpg=True,
                                     disturbance_cfg=disturbance_cfg)

        # 如果存在已保存的模型, 加载后继续训练
        if os.path.exists(model_path):
            print(f"\n[自动加载] 发现已保存的模型: {model_path}")
            task_ddpg.ddpg_agent.load(model_path)

        train_f_history = []
        for ep in range(1, num_train_episodes + 1):
            verbose_ep = (ep == 1 or ep == num_train_episodes or ep % 10 == 0)
            if verbose_ep:
                print(f"\n--- 训练 Episode {ep}/{num_train_episodes} ---")
            result_train = task_ddpg.run_training_episode(
                total_time=total_time, verbose=verbose_ep
            )
            train_f_history.append(result_train['f_value'])

            stats = task_ddpg.ddpg_agent.get_stats()
            print(f"  Ep {ep}: F={result_train['f_value']:.6f}, "
                  f"IAE={result_train['iae']:.6f}, "
                  f"noise_σ={stats['noise_sigma']:.4f}, "
                  f"steps={stats['total_steps']}")

        # 训练完成后自动保存模型
        task_ddpg.ddpg_agent.save(model_path)
        print(f"[自动保存] 模型已保存到: {model_path}")

        print(f"\n[训练 F 值变化]")
        f_min = min(train_f_history) if train_f_history else 1
        f_max = max(train_f_history) if train_f_history else 1
        for ep, fv in enumerate(train_f_history, 1):
            bar_len = int((fv - f_min) / (f_max - f_min + 1e-9) * 30)
            bar = "█" * bar_len
            print(f"  Ep {ep:>2}: F={fv:.6f}  |{bar}")

        # 阶段 3: STA+DDPG 评估
        print("\n" + "#" * 60)
        print("  阶段 3: STA+DDPG 评估")
        print("#" * 60)
        task_ddpg.reset_episode()
        result_ddpg = task_ddpg.run(total_time=total_time, evaluate=True)

        compare_label = f"长时序任务 - 训练 {num_train_episodes} episodes 后"
        print_comparison(result_sta, result_ddpg, compare_label)
        plot_comparison(result_sta, result_ddpg, train_f_history, compare_label)

    elif mode == 'sta':
        task = LongSequenceTask(scene_path, enable_ddpg=False,
                                disturbance_cfg=disturbance_cfg)
        task.run(total_time=total_time)

    elif mode == 'ddpg':
        task = LongSequenceTask(scene_path, enable_ddpg=True,
                                disturbance_cfg=disturbance_cfg)
        # 如果存在已保存的模型, 自动加载
        if os.path.exists(model_path):
            print(f"\n[自动加载] 发现已保存的模型: {model_path}")
            task.ddpg_agent.load(model_path)
        task.run(total_time=total_time)


if __name__ == '__main__':
    main()
