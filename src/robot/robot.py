import abc
import copy
import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3

from typing import List
from src.geometry import Geometry3D


def get_transformation_mdh(alpha, a, d, theta, sigma, q) -> SE3:
    if sigma == 0:
        theta += q
    elif sigma == 1:
        d += q

    return SE3.Rx(alpha) * SE3.Tx(a) * SE3.Tz(d) * SE3.Rz(theta)


def wrap(theta) -> tuple:
    number = np.floor(theta / (2 * np.pi))
    theta -= 2 * np.pi * number
    if theta < -np.pi:
        theta += 2 * np.pi
        number -= 1
    elif theta > np.pi:
        theta -= 2 * np.pi
        number += 1

    return theta, number


class Robot(abc.ABC):

    def __init__(self) -> None:
        super().__init__()
        self.robot: rtb.Robot = None
        self._dof = 0
        self.q0 = [0.0 for _ in range(self._dof)]
        self.alpha_array = [0.0 for _ in range(self._dof)]
        self.a_array = [0.0 for _ in range(self._dof)]
        self.d_array = [0.0 for _ in range(self._dof)]
        self.theta_array = [0.0 for _ in range(self._dof)]
        self.sigma_array = [0 for _ in range(self._dof)]

    @property
    def dof(self) -> int:
        return self._dof

    def fkine(self, q) -> SE3:
        return self.robot.fkine(q)

    def get_cartesian(self):
        return self.fkine(self.q0)

    def move_joint(self, q):
        self.set_joint(q)

    def set_joint(self, q):
        self.q0 = q[:]
        self.setRobotConfig(self.q0)

    def setRobotConfig(self, q):
        pass

    def move_cartesian(self, T: SE3):
        q = self.ikine(T)

        assert len(q)  # inverse kinematics failure
        self.q0 = q[:]

    @abc.abstractmethod
    def ikine(self, Tep: SE3) -> list:
        pass

    def get_joint(self):
        return copy.deepcopy(self.q0)

    def get_geometries(self) -> List[Geometry3D]:
        pass

    def __getstate__(self):
        state = {"dof": self._dof,
                 "q0": self.q0,
                 "alpha_array": self.alpha_array,
                 "a_array": self.a_array,
                 "d_array": self.d_array,
                 "theta_array": self.theta_array,
                 "sigma_array": self.sigma_array,
                 }
        return state

    def __setstate__(self, state):
        self._dof = state["dof"]
        self.q0 = state["q0"]
        self.alpha_array = state["alpha_array"]
        self.a_array = state["a_array"]
        self.d_array = state["d_array"]
        self.theta_array = state["theta_array"]
        self.sigma_array = state.get("sigma_array", [0] * self._dof)
        links = []
        for i in range(self._dof):
            links.append(
                rtb.DHLink(d=self.d_array[i], alpha=self.alpha_array[i], a=self.a_array[i], offset=self.theta_array[i],
                           mdh=True))
        self.robot = rtb.DHRobot(links)
