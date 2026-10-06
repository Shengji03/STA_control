"""
技能执行器 - 按顺序执行技能序列

SkillContext:  技能运行时的共享上下文 (模型、传感器、机器人等)
SkillExecutor: 按序调度技能, 处理技能间切换与 STA 参数热切换
"""

import numpy as np
import mujoco


class SkillContext:
    """
    技能执行上下文

    持有仿真环境的共享引用, 供各技能在运行时读写。
    """

    def __init__(self, model, data, robot, dof, sensor_offset=0):
        self.model = model
        self.data = data
        self.robot = robot
        self.dof = dof
        self.sensor_offset = sensor_offset
        self.gripper_target = 0.0
        self.current_time = 0.0
        self.skill_start_time = 0.0
        # Sensor-buffer offsets are not generalized velocity DOF addresses.
        # Resolve each scalar joint-position sensor to its actual model joint.
        sensors = []
        for address in range(sensor_offset, sensor_offset + dof):
            matches = np.flatnonzero(
                (model.sensor_adr == address)
                & (model.sensor_dim == 1)
                & (model.sensor_type == mujoco.mjtSensor.mjSENS_JOINTPOS)
                & (model.sensor_objtype == mujoco.mjtObj.mjOBJ_JOINT)
            )
            if len(matches) != 1:
                raise ValueError(f'关节传感器地址 {address} 不能唯一关联到单自由度关节')
            sensors.append(int(matches[0]))
        self.joint_ids = np.asarray(model.sensor_objid[sensors], dtype=int)
        if len(set(self.joint_ids)) != dof:
            raise ValueError('机械臂关节传感器不能重复指向同一关节')
        self.joint_dof_indices = np.asarray(model.jnt_dofadr[self.joint_ids], dtype=int)
        self.joint_qpos_indices = np.asarray(model.jnt_qposadr[self.joint_ids], dtype=int)
        first_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(self.joint_ids[0])) or ''
        suffix = '_R' if first_name.endswith('_R') else ''
        self.flange_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'flange' + suffix)
        self.control_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'pinch' + suffix)
        self._wrench_body_id = None
        self._wrench_body_ids = np.array([], dtype=int)

    @property
    def elapsed(self):
        """当前技能已执行时间"""
        return self.current_time - self.skill_start_time

    def get_sensor_data(self):
        """获取关节位置传感器数据"""
        s = self.sensor_offset
        return self.data.sensordata[s:s + self.dof].copy()

    def get_external_force(self) -> np.ndarray:
        """Net external wrench on flange + gripper at the control point.

        Returns world-axis [force (N), torque (N.m)], exerted ON the tool.
        cfrc_ext contains only directly applied/contact forces on each body;
        it is torque:force ordered and referenced to the kinematic root CoM.
        Sum descendants to include finger contacts (internal contacts cancel),
        then translate the moment to the same point used by the Jacobian.
        """
        if self.flange_body_id < 0:
            raise ValueError('力反馈缺少法兰刚体')
        if self._wrench_body_id != self.flange_body_id:
            bodies = []
            for body in range(1, self.model.nbody):
                ancestor = body
                while ancestor > 0:
                    if ancestor == self.flange_body_id:
                        bodies.append(body)
                        break
                    ancestor = int(self.model.body_parentid[ancestor])
            self._wrench_body_ids = np.asarray(bodies, dtype=int)
            self._wrench_body_id = self.flange_body_id
        # These scenes have no acceleration/force sensors that trigger RNE.
        mujoco.mj_rnePostConstraint(self.model, self.data)
        wrench = np.sum(self.data.cfrc_ext[self._wrench_body_ids], axis=0)
        force = wrench[3:]
        root = int(self.model.body_rootid[self.flange_body_id])
        torque = wrench[:3] + np.cross(self.data.subtree_com[root] - self.control_point, force)
        return np.concatenate([force, torque])

    @property
    def control_point(self):
        """Gripper pinch point; historical 'tcp' sites are at the flange."""
        if self.control_site_id >= 0:
            return self.data.site_xpos[self.control_site_id].copy()
        if self.flange_body_id >= 0:
            return self.data.xpos[self.flange_body_id].copy()
        raise ValueError('缺少控制参考点')

    def get_jacobian(self) -> np.ndarray:
        """World-axis [linear; angular] Jacobian at the wrench reference point."""
        if self.flange_body_id < 0:
            raise ValueError('雅可比计算缺少法兰刚体')
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        if self.control_site_id >= 0:
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.control_site_id)
        else:
            mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.flange_body_id)
        jac = np.vstack([jacp[:, self.joint_dof_indices],
                         jacr[:, self.joint_dof_indices]])
        return jac


class SkillExecutor:
    """
    技能序列执行器

    按顺序执行一组技能, 自动处理:
        - 技能间的切换
        - STA 控制器参数热切换
        - STA 积分项重置
        - 完成状态判定
    """

    def __init__(self, skills, sta_controllers, dof):
        """
        Args:
            skills:          技能列表 [BaseSkill, ...]
            sta_controllers: STA 控制器列表 (每个关节一个)
            dof:             自由度
        """
        self.skills = skills
        self.sta_controllers = sta_controllers
        self.dof = dof
        self.current_index = -1
        self._skill_started = False
        self._complete = False
        self._completion_time = None

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------

    @property
    def current_skill(self):
        """当前正在执行的技能 (已完成则返回 None)"""
        if 0 <= self.current_index < len(self.skills):
            return self.skills[self.current_index]
        return None

    @property
    def is_all_complete(self):
        """是否所有技能已执行完成"""
        return self._complete

    @property
    def completion_time(self):
        """所有技能完成的时刻 (未完成返回 None)"""
        return self._completion_time

    @property
    def progress(self):
        """返回 (当前技能索引, 总技能数)"""
        return self.current_index, len(self.skills)

    # ------------------------------------------------------------------
    # 核心接口
    # ------------------------------------------------------------------

    def reset(self):
        """重置执行器"""
        self.current_index = -1
        self._skill_started = False
        self._complete = False
        self._completion_time = None

    def update(self, ctx) -> bool:
        """
        更新执行器状态

        Args:
            ctx: SkillContext 技能上下文

        Returns:
            bool: 是否所有技能已完成
        """
        if self._complete:
            return True

        # 启动第一个技能
        if self.current_index == -1:
            self.current_index = 0
            self._skill_started = False

        if self.current_index >= len(self.skills):
            self._mark_complete(ctx)
            return True

        skill = self.skills[self.current_index]

        # 初始化新技能
        if not self._skill_started:
            self._transition_to_skill(skill, ctx)
            self._skill_started = True

        # 检查当前技能是否完成
        if skill.is_complete(ctx):
            self.current_index += 1
            self._skill_started = False

            if self.current_index >= len(self.skills):
                self._mark_complete(ctx)
                return True

            # 立即启动下一个技能
            return self.update(ctx)

        return False

    def get_desired_position(self, ctx) -> np.ndarray:
        """获取当前技能的期望关节位置"""
        skill = self.current_skill
        if skill is None:
            return np.array(ctx.robot.get_joint())
        return skill.get_desired_position(ctx)

    def get_desired_velocity(self, ctx) -> np.ndarray:
        skill = self.current_skill if not self._complete else None
        return skill.get_desired_velocity(ctx) if skill else np.zeros(self.dof)

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _mark_complete(self, ctx):
        """标记全部完成"""
        self._complete = True
        if self._completion_time is None:
            self._completion_time = ctx.current_time
            print(f"[{ctx.current_time:.2f}s] 所有技能执行完成! "
                  f"(共 {len(self.skills)} 个技能)")

    def _transition_to_skill(self, skill, ctx):
        """切换到新技能"""
        prev_name = (self.skills[self.current_index - 1].name
                     if self.current_index > 0 else "START")
        print(f"[{ctx.current_time:.2f}s] Skill: {prev_name} -> {skill.name} "
              f"({self.current_index + 1}/{len(self.skills)})")

        ctx.skill_start_time = ctx.current_time

        # 重置 STA 控制器积分项
        for ctrl in self.sta_controllers:
            ctrl.reset()

        # 应用技能专属 STA 参数
        params = skill.sta_params
        for i in range(self.dof):
            self.sta_controllers[i].set_parameter(
                params['alpha'][i], params['beta'][i], params['lambda_s'][i]
            )

        # 调用技能初始化
        skill.setup(ctx)
