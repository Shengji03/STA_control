"""The existing move-then-turn navigation profile, independent of MuJoCo."""

from dataclasses import dataclass, field

import numpy as np

from src.config.robot import MOBILE_ROBOT


@dataclass
class NavigationTrajectory:
    start: np.ndarray
    target: np.ndarray
    start_time: float
    start_yaw: float
    target_yaw: float | None = None
    speed: float = MOBILE_ROBOT.navigation_speed
    yaw_speed: float = MOBILE_ROBOT.yaw_speed
    move_duration: float = field(init=False)
    yaw_duration: float = field(init=False)

    def __post_init__(self) -> None:
        self.start = np.asarray(self.start, dtype=float).copy()
        self.target = np.asarray(self.target, dtype=float).copy()
        if self.start.shape != (2,) or self.target.shape != (2,):
            raise ValueError("Navigation positions must contain [x, y]")
        if self.speed <= 0 or self.yaw_speed <= 0:
            raise ValueError("Navigation speeds must be positive")
        distance = float(np.linalg.norm(self.target - self.start))
        self.move_duration = max(distance / self.speed, 2.0) if distance > 0.05 else 0.0
        yaw_delta = 0.0 if self.target_yaw is None else abs(self.target_yaw - self.start_yaw)
        self.yaw_duration = max(yaw_delta / self.yaw_speed, 2.0) if yaw_delta > 0.01 else 0.0

    @property
    def duration(self) -> float:
        return max(self.move_duration + self.yaw_duration, 2.0)

    def sample(self, time: float) -> tuple[np.ndarray, float | None]:
        elapsed = max(time - self.start_time, 0.0)
        if self.move_duration > 0 and elapsed < self.move_duration:
            progress = elapsed / self.move_duration
            return self.start + progress * (self.target - self.start), None
        yaw = None
        if self.yaw_duration > 0 and self.target_yaw is not None:
            progress = min((elapsed - self.move_duration) / self.yaw_duration, 1.0)
            yaw = self.start_yaw + progress * (self.target_yaw - self.start_yaw)
        return self.target.copy(), yaw
