"""
PlanExecutor: Converts LLM JSON plan into executable Skill objects.
"""

import numpy as np
from typing import Dict, Optional
from spatialmath import SE3, SO3

from ..config.robot import INITIAL_JOINTS, MOBILE_ROBOT

from ..skills.skills import (
    MoveSkill, GraspSkill, RotateSkill,
    InsertSkill, ExtractSkill, TranslateSkill
)
from ..skills.base_skill import BaseSkill, SkillType, SKILL_STA_PARAMS
from ..robot.ur5e import UR5e


class PlanExecutor:
    """Converts LLM plan JSON into executable Skill objects for each arm."""

    SKILL_MAP = {
        'MoveSkill':      (MoveSkill,      SkillType.MOVE),
        'GraspSkill':     (GraspSkill,     SkillType.GRASP),
        'RotateSkill':    (RotateSkill,    SkillType.ROTATE),
        'InsertSkill':    (InsertSkill,    SkillType.INSERT),
        'ExtractSkill':   (ExtractSkill,   SkillType.EXTRACT),
        'TranslateSkill': (TranslateSkill, SkillType.TRANSLATE),
    }

    def __init__(self, arm_offsets: Dict[str, np.ndarray], arm_yaws: Dict[str, float]):
        """
        Args:
            arm_offsets: {'L': [x,y,z], 'R': [x,y,z]} - arm positions relative to cart
            arm_yaws: {'L': 0.0, 'R': pi} - arm yaw rotations
        """
        self.arm_offsets = arm_offsets
        self.arm_yaws = arm_yaws

        # Maintain separate UR5e instances for IK solving
        self._robots = {
            'L': UR5e(),
            'R': UR5e(),
        }

        # Initialize robots to default config
        init_q = list(INITIAL_JOINTS)
        for robot in self._robots.values():
            robot.set_joint(init_q)
            robot.setRobotConfig(init_q)

    def parse(self, llm_result: dict, cart_pos: np.ndarray) -> dict:
        """
        Parse LLM plan result into phased skill lists.

        Each NavSkill starts a new phase. Skills after a NavSkill belong to
        that phase and are IK-solved with the post-navigation cart position.

        Returns:
            {
                'phases': [
                    {'nav': {'target': [x,y]} or None, 'L': [...], 'R': [...]},
                    ...
                ]
            }
        """
        plan_steps = llm_result.get('plan', [])

        phases = []
        current_phase = {'nav': None, 'L': [], 'R': []}
        effective_cart_pos = np.array(cart_pos, dtype=float)

        for step in plan_steps:
            arm = step.get('arm', 'L')
            skill_name = step.get('skill', '')

            if 'nav' in step or skill_name == 'NavSkill':
                params = step.get('params', {})
                target = params.get('target', None)
                if target:
                    if current_phase['L'] or current_phase['R'] or current_phase['nav']:
                        phases.append(current_phase)
                    yaw = params.get('yaw', None)
                    current_phase = {
                        'nav': {'target': target, 'yaw': yaw},
                        'L': [], 'R': [],
                    }
                    effective_cart_pos = np.array(target[:2], dtype=float)
                    print(f"[PlanExecutor] Phase {len(phases)}: NavSkill -> "
                          f"小车将移动到 {target}")
                continue

            try:
                skill = self._parse_step(step, effective_cart_pos)
                if skill:
                    if arm == 'L':
                        current_phase['L'].append(skill)
                    elif arm == 'R':
                        current_phase['R'].append(skill)
            except Exception as e:
                print(f"[PlanExecutor] Error parsing step {step.get('step')}: {e}")
                raise

        if current_phase['L'] or current_phase['R'] or current_phase['nav']:
            phases.append(current_phase)

        print(f"[PlanExecutor] 共 {len(phases)} 个执行阶段")
        return {'phases': phases}

    def _parse_step(self, step: dict, cart_pos: np.ndarray) -> Optional[BaseSkill]:
        """Convert a single plan step dict to a Skill instance."""
        skill_name = step.get('skill', '')
        params = step.get('params', {})
        arm = step.get('arm', 'L')

        if skill_name not in self.SKILL_MAP:
            raise ValueError(f"Unknown skill: {skill_name}")

        skill_class, skill_type = self.SKILL_MAP[skill_name]
        sta_params = SKILL_STA_PARAMS.get(skill_type, SKILL_STA_PARAMS[SkillType.MOVE])

        # GraspSkill - no IK needed
        if skill_name == 'GraspSkill':
            action = params.get('action', 'close')
            wait_time = params.get('wait_time', 0.8)
            gripper_closed = params.get('gripper_closed', 0.35)
            return GraspSkill(
                action=action,
                wait_time=wait_time,
                gripper_closed=gripper_closed,
                sta_params=sta_params
            )

        # RotateSkill - no IK needed
        elif skill_name == 'RotateSkill':
            angle = params.get('angle', np.pi)
            duration = params.get('duration', 5.0)
            joint_index = params.get('joint_index', 5)
            return RotateSkill(
                angle=angle,
                duration=duration,
                joint_index=joint_index,
                sta_params=sta_params
            )

        # Trajectory skills - need IK
        else:
            target_pos = params.get('target_pos')
            if not target_pos:
                raise ValueError(f"Step {step.get('step')}: {skill_name} requires 'target_pos'")

            duration = params.get('duration', 3.0)
            tilt = params.get('tilt', 0.0)
            tilt_y = params.get('tilt_y', 0.0)
            if tilt != 0.0 or tilt_y != 0.0:
                orientation = SO3.Rx(np.pi + tilt) * SO3.Ry(tilt_y)
            else:
                orientation = params.get('orientation', None)

            # Resolve joint angles via IK
            target_joints = self._resolve_target_joints(
                arm, target_pos, cart_pos, orientation
            )

            if target_joints is None:
                arm_offset = self.arm_offsets[arm]
                arm_world = np.array([
                    cart_pos[0] + arm_offset[0],
                    cart_pos[1] + arm_offset[1],
                    arm_offset[2]
                ])
                raise ValueError(
                    f"IK 求解失败: {skill_name} 目标 {target_pos}, "
                    f"{arm}臂基座 {arm_world.round(3).tolist()}。"
                    f"可能原因: 目标位置 z={target_pos[2]}m 过高或角度不可达。"
                    f"请调整 target_pos, 使其在臂的可达工作空间内 (建议 z < 0.9m)。"
                )

            return skill_class(
                target_joints=target_joints,
                duration=duration,
                sta_params=sta_params
            )

    MAX_REACH = MOBILE_ROBOT.arm_reach

    def _resolve_target_joints(
        self,
        arm: str,
        target_pos: list,
        cart_pos: np.ndarray,
        orientation: Optional[SO3] = None,
        tool_offset: float = MOBILE_ROBOT.tool_offset
    ) -> Optional[np.ndarray]:
        """
        Convert world Cartesian [x,y,z] to joint angles [6 floats].

        Args:
            arm: 'L' or 'R'
            target_pos: [x, y, z] in world coordinates
            cart_pos: [x, y] current cart position
            orientation: Desired end-effector orientation (default: pointing down)
            tool_offset: Tool length offset

        Returns:
            np.ndarray of 6 joint angles, or None if IK fails
        """
        arm_offset = self.arm_offsets[arm]
        arm_world = np.array([
            cart_pos[0] + arm_offset[0],
            cart_pos[1] + arm_offset[1],
            arm_offset[2]
        ])

        target_world = np.array(target_pos)
        dist = np.linalg.norm(target_world - arm_world)

        if dist > self.MAX_REACH:
            raise ValueError(
                f"目标 {target_pos} 距离 {arm} 臂基座 {arm_world.round(3).tolist()} "
                f"为 {dist:.3f}m, 超出工作半径 {self.MAX_REACH}m。"
                f"需要先使用 NavSkill 将小车移动到目标附近。"
            )

        local_pos = self._world_to_arm_local(target_world, arm_world, self.arm_yaws[arm])
        local_pos[2] += tool_offset

        if orientation is None:
            orientation = SO3.Rx(np.pi)

        T_target = SE3.Rt(orientation.R, local_pos)

        robot = self._robots[arm]

        # 先用当前构型尝试 IK
        q = robot.ikine(T_target)
        if len(q) > 0:
            return np.array(q)

        # 回退：遍历全部 8 种构型 (overhead x inline x wrist)
        orig_cfg = (
            robot.robot_config.overhead,
            robot.robot_config.inline,
            robot.robot_config.wrist,
        )
        print(f"  [IK] {arm}臂默认构型失败, 尝试其他构型 "
              f"(local_target={local_pos.round(4).tolist()}, dist={dist:.3f}m)")

        for overhead in (0, 1):
            for inline in (0, 1):
                for wrist in (0, 1):
                    if (overhead, inline, wrist) == orig_cfg:
                        continue
                    robot.robot_config.overhead = overhead
                    robot.robot_config.inline = inline
                    robot.robot_config.wrist = wrist
                    q = robot.ikine(T_target)
                    if len(q) > 0:
                        print(f"  [IK] 构型 (o={overhead},i={inline},w={wrist}) 成功")
                        return np.array(q)

        # 恢复原始构型
        robot.robot_config.overhead = orig_cfg[0]
        robot.robot_config.inline = orig_cfg[1]
        robot.robot_config.wrist = orig_cfg[2]

        return None

    def _world_to_arm_local(
        self,
        world_pos: np.ndarray,
        arm_world: np.ndarray,
        arm_yaw: float
    ) -> np.ndarray:
        """
        Transform world position to arm-local frame.
        Extracted from pipeline_glare_task.py lines 146-152.
        """
        delta = world_pos - arm_world
        cos_yaw = np.cos(-arm_yaw)
        sin_yaw = np.sin(-arm_yaw)
        local = np.array([
            cos_yaw * delta[0] - sin_yaw * delta[1],
            sin_yaw * delta[0] + cos_yaw * delta[1],
            delta[2]
        ])
        return local

