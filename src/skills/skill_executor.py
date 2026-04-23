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
        self.flange_body_id = -1

    @property
    def elapsed(self):
        """当前技能已执行时间"""
        return self.current_time - self.skill_start_time

    def get_sensor_data(self):
        """获取关节位置传感器数据"""
        s = self.sensor_offset
        return self.data.sensordata[s:s + self.dof].copy()

    def get_external_force(self) -> np.ndarray:
        """获取末端法兰处的6维外力 [fx, fy, fz, tx, ty, tz] (世界坐标系)"""
        if self.flange_body_id < 0:
            return np.zeros(6)
        return self.data.cfrc_ext[self.flange_body_id].copy()

    def get_jacobian(self) -> np.ndarray:
        """计算末端TCP处的6xN雅可比矩阵 (世界坐标系)"""
        if self.flange_body_id < 0:
            return np.zeros((6, self.dof))
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr,
                          self.flange_body_id)
        s = self.sensor_offset
        jac = np.vstack([jacp[:, s:s + self.dof],
                         jacr[:, s:s + self.dof]])
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
