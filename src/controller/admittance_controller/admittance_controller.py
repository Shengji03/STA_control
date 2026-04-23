"""
导纳控制器

将外力转换为位移补偿量, 与轨迹期望位置叠加后统一由STA控制器跟踪。

导纳模型: M·ẍ + B·ẋ + K·Δx = S_f · (F_ext - F_ref)
其中 S_f = (I - S) 是力控选择矩阵, S 是位控选择矩阵,
F_ref 是参考力(期望接触力)。

- F_ext = F_ref 时: 位移补偿为零 (平衡点)
- F_ext > F_ref 时: 产生正向退让
- F_ext < F_ref 时: 产生反向补偿 (主动趋近)

控制链路:
    F_ext → 导纳 → Δx(任务空间) → J^{-1} → Δq(关节空间)
    q_cmd = q_trajectory + Δq
    τ = STA(q_cmd - q_actual) + DDPG
"""

import numpy as np
from enum import Enum, auto


class AdmittanceMode(Enum):
    ROTATE = auto()
    INSERT = auto()
    EXTRACT = auto()


# 各技能的导纳参数和选择矩阵
# S = diag(s1..s6): 1=位控, 0=力控(导纳)
# F_ref: 参考力(期望接触力), 力控方向上的平衡点
# 顺序: [x, y, z, Rx, Ry, Rz]
ADMITTANCE_CONFIGS = {
    AdmittanceMode.ROTATE: {
        'S':     np.array([1, 1, 0, 1, 1, 0], dtype=np.float64),
        'M':     np.array([1.0, 1.0, 0.5, 1.0, 1.0, 0.3], dtype=np.float64),
        'B':     np.array([10., 10., 8.0, 10., 10., 6.0], dtype=np.float64),
        'K':     np.array([5.0, 5.0, 2.0, 5.0, 5.0, 1.0], dtype=np.float64),
        'F_ref': np.array([0., 0., 0., 0., 0., 0.], dtype=np.float64),
    },
    AdmittanceMode.INSERT: {
        'S':     np.array([1, 1, 0, 0, 0, 1], dtype=np.float64),
        'M':     np.array([1.0, 1.0, 0.5, 0.3, 0.3, 1.0], dtype=np.float64),
        'B':     np.array([10., 10., 6.0, 4.0, 4.0, 10.], dtype=np.float64),
        'K':     np.array([5.0, 5.0, 8.0, 3.0, 3.0, 5.0], dtype=np.float64),
        'F_ref': np.array([0., 0., -5.0, 0., 0., 0.], dtype=np.float64),
    },
    AdmittanceMode.EXTRACT: {
        'S':     np.array([1, 1, 0, 1, 1, 1], dtype=np.float64),
        'M':     np.array([1.0, 1.0, 0.8, 1.0, 1.0, 1.0], dtype=np.float64),
        'B':     np.array([10., 10., 8.0, 10., 10., 10.], dtype=np.float64),
        'K':     np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0], dtype=np.float64),
        'F_ref': np.array([0., 0., 3.0, 0., 0., 0.], dtype=np.float64),
    },
}


class AdmittanceController:
    """
    导纳控制器

    输入: 6维外力 F_ext (任务空间, 世界坐标系)
    输出: 6维位移补偿 Δq (关节空间)

    离散化: 欧拉前向积分
        ẍ = M^{-1} (S_f · (F_ext - F_ref) - B·ẋ - K·Δx)
        ẋ += ẍ · dt
        Δx += ẋ · dt
        Δq = J^{+} · (S_f · Δx)
    """

    def __init__(self, mode: AdmittanceMode, ts: float = 0.002):
        cfg = ADMITTANCE_CONFIGS[mode]
        self._S = cfg['S'].copy()
        self._S_f = 1.0 - self._S
        self._M_inv = 1.0 / cfg['M']
        self._B = cfg['B'].copy()
        self._K = cfg['K'].copy()
        self._F_ref = cfg['F_ref'].copy()
        self._ts = ts

        self._dx = np.zeros(6)
        self._ddx = np.zeros(6)
        self._vel = np.zeros(6)

        self._max_dx = 0.05
        self._max_dq = 0.1

    def reset(self):
        self._dx[:] = 0
        self._ddx[:] = 0
        self._vel[:] = 0

    def compute(self, f_ext: np.ndarray, jacobian: np.ndarray,
                dof: int = 6) -> np.ndarray:
        """
        计算导纳位移补偿量 (关节空间)

        导纳模型: M·ẍ + B·ẋ + K·Δx = S_f · (F_ext - F_ref)
        当 F_ext = F_ref 时 Δx = 0 (平衡点)

        Args:
            f_ext: 6维外力 [fx,fy,fz,tx,ty,tz] 世界坐标系
            jacobian: 6xN 雅可比矩阵
            dof: 关节自由度

        Returns:
            Δq: 关节空间位移补偿 (dof,)
        """
        f_error = f_ext - self._F_ref
        f_filtered = self._S_f * f_error

        self._ddx = self._M_inv * (f_filtered - self._B * self._vel
                                    - self._K * self._dx)
        self._vel += self._ddx * self._ts
        self._dx += self._vel * self._ts

        self._dx = np.clip(self._dx, -self._max_dx, self._max_dx)

        dx_selected = self._S_f * self._dx

        J = jacobian
        if J.shape[0] == 6 and J.shape[1] >= dof:
            J_use = J[:, :dof]
            try:
                dq = np.linalg.lstsq(J_use, dx_selected, rcond=None)[0]
            except np.linalg.LinAlgError:
                dq = np.zeros(dof)
        else:
            dq = np.zeros(dof)

        dq = np.clip(dq, -self._max_dq, self._max_dq)
        return dq

    @property
    def displacement(self) -> np.ndarray:
        return self._dx.copy()

    @property
    def selection_matrix(self) -> np.ndarray:
        return self._S.copy()
