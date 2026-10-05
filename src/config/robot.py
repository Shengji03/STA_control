"""Geometry shared by the current Scene5 planner, perception and execution."""

from dataclasses import dataclass
from math import pi


DOF = 6
INITIAL_JOINTS = (0.0, 0.0, pi / 2, 0.0, -pi / 2, 0.0)


@dataclass(frozen=True)
class MobileRobotSettings:
    left_offset: tuple[float, float, float] = (0.35, 0.0, 0.35)
    right_offset: tuple[float, float, float] = (-0.35, 0.0, 0.35)
    left_yaw: float = 0.0
    right_yaw: float = pi
    left_sensor_offset: int = 0
    right_sensor_offset: int = DOF
    arm_reach: float = 0.85
    tool_offset: float = 0.155
    cart_start: tuple[float, float, float] = (-3.0, -0.5, 0.0)
    navigation_speed: float = 0.3
    yaw_speed: float = 0.5


MOBILE_ROBOT = MobileRobotSettings()
