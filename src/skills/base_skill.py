"""
技能基类 - 所有技能原语的抽象接口

技能原语 (Skill Primitive) 是机器人操作任务的基本单元。
每个技能封装了一种原子行为 (移动、抓取、插入等),
并携带自身的控制参数 (STA 滑模参数)。
"""

import abc
import numpy as np
from enum import Enum, auto


class SkillType(Enum):
    """6种基本技能类型"""
    MOVE = auto()
    GRASP = auto()
    TRANSLATE = auto()
    INSERT = auto()
    EXTRACT = auto()
    ROTATE = auto()


# 默认 STA 参数 (通用)
DEFAULT_STA_PARAMS = {
    'alpha':    [10, 10, 10, 8, 8, 8],
    'beta':     [25, 25, 25, 18, 18, 18],
    'lambda_s': [15, 12, 25, 20, 15, 15],
}

# 按技能区分的 STA 参数 (参考 PID 版各技能增益比例, alpha=8~15, beta=15~40)
SKILL_STA_PARAMS = {
    SkillType.MOVE: {
        'alpha':    [10, 10, 10, 8,  8,  8],
        'beta':     [25, 28, 25, 18, 18, 18],
        'lambda_s': [15, 12, 25, 20, 15, 15],
    },
    SkillType.GRASP: {
        'alpha':    [10, 10, 10, 8,  8,  8],
        'beta':     [25, 28, 25, 18, 18, 18],
        'lambda_s': [15, 12, 25, 20, 15, 15],
    },
    SkillType.TRANSLATE: {
        'alpha':    [9,  9,  9,  7,  7,  7],
        'beta':     [22, 24, 22, 15, 15, 15],
        'lambda_s': [13, 10, 22, 18, 13, 13],
    },
    SkillType.INSERT: {
        'alpha':    [12, 13, 12, 10, 10, 10],
        'beta':     [30, 35, 30, 22, 22, 22],
        'lambda_s': [18, 15, 30, 24, 18, 18],
    },
    SkillType.EXTRACT: {
        'alpha':    [11, 12, 11, 9,  9,  9],
        'beta':     [28, 32, 28, 20, 20, 20],
        'lambda_s': [16, 13, 28, 22, 16, 16],
    },
    SkillType.ROTATE: {
        'alpha':    [11, 12, 11, 9,  9,  12],
        'beta':     [28, 30, 28, 20, 20, 35],
        'lambda_s': [16, 13, 28, 22, 16, 20],
    },
}


class BaseSkill(abc.ABC):
    """
    技能基类

    所有具体技能必须实现:
        - setup(ctx):               技能开始时调用 (规划轨迹、设置夹爪等)
        - is_complete(ctx):         判断技能是否执行完成
        - get_desired_position(ctx): 获取当前时刻的期望关节位置
    """

    def __init__(self, skill_type: SkillType, sta_params: dict = None,
                 name: str = ""):
        self.skill_type = skill_type
        self.sta_params = sta_params if sta_params is not None else DEFAULT_STA_PARAMS
        self.name = name or skill_type.name

    @abc.abstractmethod
    def setup(self, ctx) -> None:
        """技能开始时调用 (规划轨迹、设置夹爪等)"""
        pass

    @abc.abstractmethod
    def is_complete(self, ctx) -> bool:
        """判断技能是否执行完成"""
        pass

    @abc.abstractmethod
    def get_desired_position(self, ctx) -> np.ndarray:
        """获取当前时刻的期望关节位置"""
        pass

    def get_desired_velocity(self, ctx) -> np.ndarray:
        """Stationary skills hold a fixed reference with zero desired velocity."""
        return np.zeros(ctx.dof)

    def __repr__(self):
        return f"{self.__class__.__name__}('{self.name}')"
