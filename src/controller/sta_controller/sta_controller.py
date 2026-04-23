import numpy as np

from ..controller import Controller


class STAController(Controller):
    """
    超螺旋滑模控制器 (Super-Twisting Algorithm)

    滑模面: s = ė + λ·e
        其中 e = q_d - q (位置误差), ė = q̇_d - q̇ (速度误差)
    控制律 (适用于 ṡ = φ - b·τ, b > 0 的系统):
        u1 = alpha * |s|^0.5 * sign(s)
        u2(k) = u2(k-1) + beta * sign(s) * ts
        tau = u1 + u2
    """

    def __init__(self, alpha: float, beta: float, lambda_s: float,
                 ts: float = 0.002, u2_max: float = 150.0,
                 delta: float = 0.1, dead_zone: float = 0.005,
                 u2_leak: float = 5.0):
        self.__alpha = alpha
        self.__beta = beta
        self.__lambda_s = lambda_s
        self.__ts = ts
        self.__u2_max = u2_max
        self.__delta = delta          # 边界层厚度, 用于消除动态抖振
        self.__dead_zone = dead_zone  # 死区阈值, |s| < dead_zone 时停止施加控制
        self.__u2_leak = u2_leak      # 积分项泄漏系数 (1/s), 防止静态残余力矩
        self.__u2 = 0.0

    def control(self, error_pos: float, error_vel: float) -> float:
        s = error_vel + self.__lambda_s * error_pos

        # 死区: 误差极小时停止施加控制, 让积分项自然衰减
        if np.abs(s) < self.__dead_zone:
            self.__u2 *= (1.0 - self.__u2_leak * self.__ts)
            return self.__u2

        sat = s / (np.abs(s) + self.__delta)   # 连续近似 sign(s)

        u1 = self.__alpha * np.sqrt(np.abs(s)) * sat

        self.__u2 += self.__beta * sat * self.__ts
        # 泄漏积分: 持续将 u2 向 0 衰减, 防止静态时残余力矩振荡
        self.__u2 *= (1.0 - self.__u2_leak * self.__ts)
        self.__u2 = np.clip(self.__u2, -self.__u2_max, self.__u2_max)

        return u1 + self.__u2

    def reset(self):
        self.__u2 = 0.0

    def set_parameter(self, alpha: float, beta: float, lambda_s: float):
        self.__alpha = alpha
        self.__beta = beta
        self.__lambda_s = lambda_s

    @property
    def alpha(self):
        return self.__alpha

    @property
    def beta(self):
        return self.__beta

    @property
    def lambda_s(self):
        return self.__lambda_s

    @property
    def ts(self):
        return self.__ts

    @ts.setter
    def ts(self, ts):
        self.__ts = ts
