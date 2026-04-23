"""
具体技能实现

7种基本技能:
    - MoveSkill:      关节空间点到点移动 (纯位置控制)
    - GraspSkill:     夹取 / 释放 (纯位置控制)
    - InsertSkill:    精确插入 (力位混合: 导纳+位置)
    - ExtractSkill:   从凹槽拔出 (力位混合: 导纳+位置)
    - TranslateSkill: 带物体平移 (纯位置控制)
    - RotateSkill:    绕关节旋拧 (力位混合: 导纳+位置)
"""

import numpy as np

from .base_skill import BaseSkill, SkillType
from src.motion_planning import (
    TrajectoryParameter, TrajectoryPlanner,
    JointParameter, CubicVelocityParameter,
)
from src.controller.admittance_controller import (
    AdmittanceController, AdmittanceMode,
)


def _create_joint_trajectory(start_q, end_q, duration):
    """创建关节空间轨迹"""
    joint_param = JointParameter(start_q, end_q)
    velocity_param = CubicVelocityParameter(duration)
    trajectory_param = TrajectoryParameter(joint_param, velocity_param)
    return TrajectoryPlanner(trajectory_param)


# ============================================================================
# 轨迹跟踪类技能 (公共基类)
# ============================================================================

class TrajectorySkill(BaseSkill):
    """
    轨迹跟踪类技能基类 (纯位置控制)
    """

    def __init__(self, skill_type, target_joints, duration,
                 sta_params=None, name=""):
        super().__init__(skill_type, sta_params, name)
        self.target_joints = (np.array(target_joints)
                              if target_joints is not None else None)
        self.duration = duration
        self._planner = None

    def setup(self, ctx):
        current_q = ctx.get_sensor_data()
        self._planner = _create_joint_trajectory(
            current_q, self.target_joints, self.duration
        )
        print(f"  [{self.name}] duration={self.duration}s")

    def is_complete(self, ctx):
        return ctx.elapsed >= self.duration

    def get_desired_position(self, ctx):
        t = min(ctx.elapsed, self.duration)
        joint_pos = np.array(self._planner.interpolate(t))
        ctx.robot.move_joint(joint_pos)
        return joint_pos


class AdmittanceTrajectorySkill(TrajectorySkill):
    """
    力位混合轨迹技能基类

    在 TrajectorySkill 的基础上添加导纳控制:
        q_cmd = q_trajectory + Δq_admittance
    导纳控制器将外力转为关节空间位移补偿, 叠加到轨迹期望位置上。
    """

    admittance_mode = None

    def __init__(self, skill_type, target_joints, duration,
                 sta_params=None, name=""):
        super().__init__(skill_type, target_joints, duration,
                         sta_params, name)
        self._admittance = None

    def setup(self, ctx):
        super().setup(ctx)
        if self.admittance_mode is not None:
            ts = ctx.model.opt.timestep
            self._admittance = AdmittanceController(
                self.admittance_mode, ts=ts)
            self._admittance.reset()
            print(f"  [{self.name}] 导纳控制已启用 "
                  f"(mode={self.admittance_mode.name})")

    def get_desired_position(self, ctx):
        t = min(ctx.elapsed, self.duration)
        q_trajectory = np.array(self._planner.interpolate(t))
        ctx.robot.move_joint(q_trajectory)

        if self._admittance is not None and ctx.flange_body_id >= 0:
            f_ext = ctx.get_external_force()
            jac = ctx.get_jacobian()
            dq = self._admittance.compute(f_ext, jac, ctx.dof)
            q_cmd = q_trajectory + dq
        else:
            q_cmd = q_trajectory

        return q_cmd


# ============================================================================
# 7 种具体技能
# ============================================================================

class MoveSkill(TrajectorySkill):
    """移动技能 - 关节空间点到点运动 (纯位置控制)"""

    def __init__(self, target_joints, duration=3.0,
                 sta_params=None, name="Move"):
        super().__init__(SkillType.MOVE, target_joints, duration,
                         sta_params, name)


class InsertSkill(AdmittanceTrajectorySkill):
    """插入技能 - 精确插入运动 (力位混合: z+Rx+Ry导纳)"""

    admittance_mode = AdmittanceMode.INSERT

    def __init__(self, target_joints, duration=4.0,
                 sta_params=None, name="Insert"):
        super().__init__(SkillType.INSERT, target_joints, duration,
                         sta_params, name)


class ExtractSkill(AdmittanceTrajectorySkill):
    """拔出技能 - 从凹槽中拔出 (力位混合: z导纳)"""

    admittance_mode = AdmittanceMode.EXTRACT

    def __init__(self, target_joints, duration=3.0,
                 sta_params=None, name="Extract"):
        super().__init__(SkillType.EXTRACT, target_joints, duration,
                         sta_params, name)


class TranslateSkill(TrajectorySkill):
    """平移技能 - 带物体平移 (纯位置控制)"""

    def __init__(self, target_joints, duration=4.0,
                 sta_params=None, name="Translate"):
        super().__init__(SkillType.TRANSLATE, target_joints, duration,
                         sta_params, name)


class GraspSkill(BaseSkill):
    """
    夹取 / 释放技能

    Args:
        action:         'close' 或 'open'
        wait_time:      等待时间 (秒), 确保夹爪完全闭合/打开
        gripper_open:   夹爪打开位置
        gripper_closed: 夹爪关闭位置
    """

    def __init__(self, action='close', wait_time=0.8,
                 gripper_open=0.0, gripper_closed=0.65,
                 sta_params=None, name=""):
        name = name or ("Grasp" if action == 'close' else "Release")
        super().__init__(SkillType.GRASP, sta_params, name)
        self.action = action
        self.wait_time = wait_time
        self.gripper_open = gripper_open
        self.gripper_closed = gripper_closed

    def setup(self, ctx):
        self._gripper_start = ctx.gripper_target
        if self.action == 'close':
            self._gripper_end = self.gripper_closed
        else:
            self._gripper_end = self.gripper_open
        print(f"  [{self.name}] action={self.action}, wait={self.wait_time}s")

    def is_complete(self, ctx):
        return ctx.elapsed >= self.wait_time

    def get_desired_position(self, ctx):
        alpha = min(ctx.elapsed / self.wait_time, 1.0)
        ctx.gripper_target = self._gripper_start + alpha * (self._gripper_end - self._gripper_start)
        return np.array(ctx.robot.get_joint())


class RotateSkill(AdmittanceTrajectorySkill):
    """
    旋拧技能 - 绕指定关节旋转 (力位混合: z+Rz导纳)

    Args:
        angle:       旋转角度 (rad)
        duration:    旋转持续时间 (秒)
        joint_index: 旋转的关节索引 (默认 5 = 腕关节3)
    """

    admittance_mode = AdmittanceMode.ROTATE

    def __init__(self, angle=2 * np.pi, duration=5.0, joint_index=5,
                 sta_params=None, name="Rotate"):
        super().__init__(SkillType.ROTATE, None, duration,
                         sta_params, name)
        self.angle = angle
        self.joint_index = joint_index

    def setup(self, ctx):
        current_q = ctx.get_sensor_data()
        self.target_joints = current_q.copy()
        self.target_joints[self.joint_index] += self.angle
        super().setup(ctx)
