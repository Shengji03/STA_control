"""State for one arm; coordination belongs to TaskRunner."""

import numpy as np
from src.config.robot import INITIAL_JOINTS
from src.robot.ur5e import UR5e

INIT_Q = np.array(INITIAL_JOINTS)


class ArmState:
    """State container for a single arm."""
    def __init__(self, label: str, offset: np.ndarray, yaw: float, sensor_offset: int):
        self.label = label
        self.offset = offset
        self.yaw = yaw
        self.sensor_offset = sensor_offset
        self.robot = UR5e()
        self.robot.set_joint(INIT_Q)
        self.sta_controllers = []
        self.executor = None
        self.ctx = None
        self.prev_sensor = INIT_Q.copy()
