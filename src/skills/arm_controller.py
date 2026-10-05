"""Shared STA arm setup used by the scripted pipeline experiments."""

import mujoco
import numpy as np

from src.config.robot import DOF, INITIAL_JOINTS
from src.controller.sta_controller import STAController
from src.robot import UR5e

from .base_skill import DEFAULT_STA_PARAMS
from .skill_executor import SkillContext, SkillExecutor


class ArmController:
    def __init__(
        self, model, data, label, arm_joint_name,
        gripper_actuator_name, sensor_offset=0, dof=DOF,
    ):
        self.label = label
        self.dof = dof
        self.sensor_offset = sensor_offset
        self.robot = UR5e()
        self.initial_q = np.array(INITIAL_JOINTS)
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, arm_joint_name)
        self._arm_qpos_adr = model.jnt_qposadr[joint_id]
        self._gripper_ctrl = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, gripper_actuator_name,
        )
        params = DEFAULT_STA_PARAMS
        self.sta_controllers = [
            STAController(params['alpha'][i], params['beta'][i], params['lambda_s'][i],
                          ts=model.opt.timestep)
            for i in range(dof)
        ]
        self.executor = None
        self.ctx = SkillContext(model, data, self.robot, dof, sensor_offset=sensor_offset)
        self.gripper_target = 0.0
        self._prev_desired = self.initial_q.copy()

    def set_skills(self, skills) -> None:
        self.skills = list(skills)
        self.executor = SkillExecutor(self.skills, self.sta_controllers, self.dof)
        for controller in self.sta_controllers:
            controller.reset()

    def compute_control(self, desired_pos, sensor_data, real_vel, ts):
        desired_pos = np.asarray(desired_pos)
        error_pos = desired_pos - sensor_data
        desired_vel = (desired_pos - self._prev_desired) / ts
        self._prev_desired = desired_pos.copy()
        error_vel = desired_vel - real_vel
        return np.array([
            controller.control(error_pos[i], error_vel[i])
            for i, controller in enumerate(self.sta_controllers)
        ])
