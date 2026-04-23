from typing import List

import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3

from src.geometry import Geometry3D, Capsule
from src.utils import MathUtils
from .robot_config import RobotConfig
from .robot import Robot, get_transformation_mdh, wrap


class UR5e(Robot):
    def __init__(self) -> None:
        super().__init__()
        d1 = 0.163
        d4 = 0.134
        d5 = 0.1
        d6 = 0.1

        a3 = 0.425
        a4 = 0.392

        self._dof = 6
        self.q0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        alpha_array = [0.0, -np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2]
        a_array = [0.0, 0.0, a3, a4, 0.0, 0.0]
        d_array = [d1, 0.0, 0.0, d4, d5, d6]
        theta_array = [0.0, -np.pi / 2, 0.0, np.pi / 2, 0.0, 0.0]
        sigma_array = [0, 0, 0, 0, 0, 0]
        tool = SE3.Trans(0.0, 0.0, 0.2)

        links = []
        for i in range(6):
            links.append(rtb.DHLink(d=d_array[i], alpha=alpha_array[i], a=a_array[i], offset=theta_array[i], mdh=True))
        self.robot = rtb.DHRobot(links)

        self.alpha_array = alpha_array
        self.a_array = a_array
        self.d_array = d_array
        self.theta_array = theta_array
        self.sigma_array = sigma_array

        self.robot_config = RobotConfig()

    def ikine(self, Tep: SE3) -> list:

        wpc = self.cal_wpc(Tep)

        thetas = [0.0 for _ in range(self._dof)]

        # solve theta1
        theta1_condition = np.power(wpc[1], 2) + np.power(wpc[0], 2) - np.power(self.d_array[3], 2)
        if theta1_condition < 0:
            return []
        if MathUtils.near_zero(np.linalg.norm([wpc[1], wpc[0]])):
            thetas[0] = self.q0[0] - self.theta_array[0]  # overhead singularity
        else:
            if self.robot_config.overhead == 0:
                thetas[0] = np.arctan2(wpc[1], wpc[0]) - np.arctan2(self.d_array[3], np.sqrt(theta1_condition))
            elif self.robot_config.overhead == 1:
                thetas[0] = np.arctan2(wpc[1], wpc[0]) - np.arctan2(self.d_array[3], -np.sqrt(theta1_condition))
            else:
                return []

        # solve theta5
        theta5_condition = -Tep.a[0] * np.sin(thetas[0]) + Tep.a[1] * np.cos(thetas[0])
        if np.abs(theta5_condition) > 1:
            return []
        if self.robot_config.wrist == 0:
            thetas[4] = np.arccos(theta5_condition)
        elif self.robot_config.wrist == 1:
            thetas[4] = -np.arccos(theta5_condition)
        else:
            return []

        # solve theta6
        m1 = - Tep.n[0] * np.sin(thetas[0]) + Tep.n[1] * np.cos(thetas[0])
        n1 = - Tep.o[0] * np.sin(thetas[0]) + Tep.o[1] * np.cos(thetas[0])
        if MathUtils.near_zero(np.sin(thetas[4])):
            thetas[5] = self.q0[5] - self.theta_array[5]
        else:
            thetas[5] = np.arctan2(-n1 / np.sin(thetas[4]), m1 / np.sin(thetas[4]))

        # solve theta3
        T01 = get_transformation_mdh(self.alpha_array[0], self.a_array[0], self.d_array[0], thetas[0],
                                     self.sigma_array[0],
                                     0.0)
        T45 = get_transformation_mdh(self.alpha_array[4], self.a_array[4], self.d_array[4], thetas[4],
                                     self.sigma_array[4],
                                     0.0)
        T56 = get_transformation_mdh(self.alpha_array[5], self.a_array[5], self.d_array[5], thetas[5],
                                     self.sigma_array[5],
                                     0.0)
        T46 = T45 * T56
        T14 = T01.inv() * Tep * T46.inv()
        x = T14.t[0]
        y = T14.t[2]

        theta3_condition = (np.power(x, 2) + np.power(y, 2) - np.power(self.a_array[2], 2)
                            - np.power(self.a_array[3], 2)) / (2 * self.a_array[2] * self.a_array[3])
        if np.abs(theta3_condition) > 1.0:
            return []
        if self.robot_config.inline == 0:
            thetas[2] = np.arccos(theta3_condition)
        elif self.robot_config.inline == 1:
            thetas[2] = - np.arccos(theta3_condition)
        else:
            return []

        # solve theta2
        M = np.array([
            [self.a_array[3] * np.cos(thetas[2]) + self.a_array[2], -self.a_array[3] * np.sin(thetas[2])],
            [self.a_array[3] * np.sin(thetas[2]), self.a_array[3] * np.cos(thetas[2]) + self.a_array[2]]
        ])
        XY = np.array([
            [x],
            [-y]
        ])
        CS = np.linalg.inv(M) @ XY
        thetas[1] = np.arctan2(CS[1, 0], CS[0, 0])

        # solve theta4
        thetas[3] = np.arctan2(-T14.o[0], T14.n[0]) - thetas[1] - thetas[2]

        qs = [wrap(thetas[i] - self.theta_array[i])[0] for i in range(self._dof)]

        q0_s = list(map(wrap, self.q0))

        for i in range(self._dof):
            if qs[i] - q0_s[i][0] > np.pi:
                qs[i] += (q0_s[i][1] - 1) * 2 * np.pi
            elif qs[i] - q0_s[i][0] < -np.pi:
                qs[i] += (q0_s[i][1] + 1) * 2 * np.pi
            else:
                qs[i] += q0_s[i][1] * 2 * np.pi

        return qs

    def setRobotConfig(self, q):
        T = self.fkine(q)
        wpc = self.cal_wpc(T)
        thetas = [q[i] + self.theta_array[i] for i in range(self._dof)]

        if wrap(np.arctan2(wpc[1], wpc[0]) - wrap(thetas[0])[0])[0] <= np.pi / 2:
            self.robot_config.overhead = 0
        else:
            self.robot_config.overhead = 1

        # inline
        if wrap(thetas[2])[0] >= 0:
            self.robot_config.inline = 0
        else:
            self.robot_config.inline = 1

        # wrist
        if wrap(thetas[4])[0] >= 0:
            self.robot_config.wrist = 0
        else:
            self.robot_config.wrist = 1

    def cal_wpc(self, T: SE3) -> np.ndarray:
        t = T.t - self.d_array[5] * T.a
        t[2] -= self.d_array[0]
        return t

    def get_geometries(self) -> List[Geometry3D]:
        Ts = []
        T = SE3()
        for i in range(self._dof):
            T = T * get_transformation_mdh(self.alpha_array[i], self.a_array[i], self.d_array[i], self.theta_array[i],
                                           self.sigma_array[i], self.q0[i])
            Ts.append(T)

        T1 = Ts[0] * SE3.Trans(0, 0, -0.04)
        geometry1 = Capsule(T1, 0.06, 0.12)

        T2 = Ts[1] * SE3.Trans(0.2, 0, 0.138) * SE3.Ry(-np.pi / 2)
        geometry2 = Capsule(T2, 0.05, 0.4)

        T3 = Ts[2] * SE3.Trans(0.2, 0, 0.007) * SE3.Ry(-np.pi / 2)
        geometry3 = Capsule(T3, 0.038, 0.38)

        T4 = Ts[3] * SE3.Trans(0.0, 0.0, -0.07)
        geometry4 = Capsule(T4, 0.04, 0.14)

        T5 = Ts[4] * SE3.Trans(0.0, 0.0, -0.06)
        geometry5 = Capsule(T5, 0.04, 0.12)

        T6 = Ts[5] * SE3.Trans(0.0, 0.0, -0.06)
        geometry6 = Capsule(T6, 0.04, 0.12)

        return [geometry1, geometry2, geometry3, geometry4, geometry5, geometry6]


    def __getstate__(self):
        state = super().__getstate__()
        state['robot_config'] = {
            'overhead': self.robot_config.overhead,
            'inline': self.robot_config.inline,
            'wrist': self.robot_config.wrist,
        }
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self.robot_config = RobotConfig()
        if 'robot_config' in state:
            self.robot_config.overhead = state['robot_config']['overhead']
            self.robot_config.inline = state['robot_config']['inline']
            self.robot_config.wrist = state['robot_config']['wrist']


if __name__ == '__main__':
    ur_robot = UR5e()
    q0 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    T1 = ur_robot.fkine(q0)
    print(T1)
    ur_robot.move_cartesian(T1)
    q_new = ur_robot.get_joint()
    print(q_new)
