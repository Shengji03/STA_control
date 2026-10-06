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
# F_ref: 环境施加在工具上的期望接触力（世界轴），与反馈采用相同符号。
# 顺序: [x, y, z, Rx, Ry, Rz]
ADMITTANCE_CONFIGS = {
    AdmittanceMode.ROTATE: {
        'S':     np.array([1, 1, 0, 1, 1, 0], dtype=np.float64),
        'M':     np.array([1.0, 1.0, 0.5, 1.0, 1.0, 0.3], dtype=np.float64),
        'B':     np.array([10., 10., 8.0, 10., 10., 6.0], dtype=np.float64),
        'K':     np.array([5.0, 5.0, 2.0, 5.0, 5.0, 1.0], dtype=np.float64),
        'F_ref': np.array([0., 0., 0., 0., 0., 0.], dtype=np.float64),
        # Keep the fingers engaged with the thin handwheel while yielding.
        'max_dx': np.array([.003, .003, .003, .02, .02, .02]),
        'max_velocity': np.array([.01, .01, .01, .1, .1, .1]),
    },
    AdmittanceMode.INSERT: {
        'S':     np.array([1, 1, 0, 0, 0, 1], dtype=np.float64),
        'M':     np.array([1.0, 1.0, 0.5, 0.3, 0.3, 1.0], dtype=np.float64),
        'B':     np.array([10., 10., 6.0, 4.0, 4.0, 10.], dtype=np.float64),
        'K':     np.array([5.0, 5.0, 8.0, 3.0, 3.0, 5.0], dtype=np.float64),
        'F_ref': np.array([0., 0., 5.0, 0., 0., 0.], dtype=np.float64),
        'max_dx': np.array([.05, .05, .05, .1, .1, .1]),
        'max_velocity': np.array([.05, .05, .05, .2, .2, .2]),
    },
    AdmittanceMode.EXTRACT: {
        'S':     np.array([1, 1, 0, 1, 1, 1], dtype=np.float64),
        'M':     np.array([1.0, 1.0, 0.8, 1.0, 1.0, 1.0], dtype=np.float64),
        'B':     np.array([10., 10., 8.0, 10., 10., 10.], dtype=np.float64),
        'K':     np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0], dtype=np.float64),
        'F_ref': np.array([0., 0., -3.0, 0., 0., 0.], dtype=np.float64),
        'max_dx': np.array([.05, .05, .05, .1, .1, .1]),
        'max_velocity': np.array([.05, .05, .05, .2, .2, .2]),
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
        Δq = J^T (J J^T + λ²I)^{-1} · (S_f · Δx)，再限制关节补偿幅值和速度。
    """

    def __init__(self, mode: AdmittanceMode, ts: float = 0.002):
        if not np.isfinite(ts) or ts <= 0:
            raise ValueError('导纳采样周期必须为正有限值')
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

        self._max_dx = cfg['max_dx'].copy()
        self._max_velocity = cfg['max_velocity'].copy()
        self._max_dq = 0.1
        self._max_dq_velocity = 0.5
        self._jacobian_damping = 0.01
        self._previous_dq = None

    def reset(self):
        self._dx[:] = 0
        self._ddx[:] = 0
        self._vel[:] = 0
        self._previous_dq = None

    def compute(self, f_ext: np.ndarray, jacobian: np.ndarray,
                dof: int = 6) -> np.ndarray:
        """
        计算导纳位移补偿量 (关节空间)

        导纳模型: M·ẍ + B·ẋ + K·Δx = S_f · (F_ext - F_ref)
        当 F_ext = F_ref 时 Δx = 0 (平衡点)

        Args:
            f_ext: 6维外力 [fx,fy,fz,tx,ty,tz] 世界坐标系
            jacobian: 控制参考点处的 6xdof 雅可比，前三行为世界轴线速度。
            dof: 关节自由度

        Returns:
            Δq: 关节空间位移补偿 (dof,)
        """
        f_ext = np.asarray(f_ext, dtype=float)
        jacobian = np.asarray(jacobian, dtype=float)
        if f_ext.shape != (6,) or not np.all(np.isfinite(f_ext)):
            raise ValueError('外力必须是有限的六维 [力, 力矩] 向量')
        if jacobian.shape != (6, dof) or not np.all(np.isfinite(jacobian)):
            raise ValueError(f'机械臂雅可比必须是有限的 6x{dof} 矩阵')
        f_error = f_ext - self._F_ref
        f_filtered = self._S_f * f_error

        self._ddx = self._M_inv * (f_filtered - self._B * self._vel
                                    - self._K * self._dx)
        self._vel += self._ddx * self._ts
        self._vel = np.clip(self._vel, -self._max_velocity, self._max_velocity)
        self._dx += self._vel * self._ts
        at_limit = np.abs(self._dx) >= self._max_dx
        self._dx = np.clip(self._dx, -self._max_dx, self._max_dx)
        # Do not accumulate outward velocity while displacement is saturated.
        outward = at_limit & (self._vel * self._dx > 0)
        self._vel[outward] = 0.0

        dx_selected = self._S_f * self._dx

        try:
            # Damped least squares keeps near-singular wrist poses bounded.
            normal = jacobian @ jacobian.T + self._jacobian_damping ** 2 * np.eye(6)
            dq = jacobian.T @ np.linalg.solve(normal, dx_selected)
        except np.linalg.LinAlgError as exc:
            raise ValueError('导纳雅可比映射求解失败') from exc
        if not np.all(np.isfinite(dq)):
            raise ValueError('导纳补偿包含非有限值')

        dq = np.clip(dq, -self._max_dq, self._max_dq)
        previous = np.zeros(dof) if self._previous_dq is None else self._previous_dq
        step = self._max_dq_velocity * self._ts
        dq = np.clip(dq, previous - step, previous + step)
        self._previous_dq = dq.copy()
        return dq

    @property
    def displacement(self) -> np.ndarray:
        return self._dx.copy()

    @property
    def selection_matrix(self) -> np.ndarray:
        return self._S.copy()
