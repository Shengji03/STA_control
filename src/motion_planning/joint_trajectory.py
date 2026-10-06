"""The cubic joint trajectory used by all current movement skills."""

import numpy as np


class JointTrajectory:
    def __init__(self, start, target, duration: float):
        self.start = np.asarray(start, dtype=float).copy()
        self.target = np.asarray(target, dtype=float).copy()
        self.duration = float(duration)
        if self.start.ndim != 1 or self.start.shape != self.target.shape:
            raise ValueError("Joint positions must be matching one-dimensional arrays")
        if not np.isfinite(self.duration) or self.duration <= 0:
            raise ValueError("Trajectory duration must be positive and finite")

    def interpolate(self, time: float) -> np.ndarray:
        progress = float(np.clip(time / self.duration, 0.0, 1.0))
        progress = 3.0 * progress ** 2 - 2.0 * progress ** 3
        return self.start + progress * (self.target - self.start)

    def velocity(self, time: float) -> np.ndarray:
        progress = float(np.clip(time / self.duration, 0.0, 1.0))
        rate = 6.0 * progress * (1.0 - progress) / self.duration
        return rate * (self.target - self.start)
