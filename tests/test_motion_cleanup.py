"""Regression checks for the retained trajectory and robot RRT geometry."""

import random

import numpy as np
import pytest
from spatialmath import SE3

from src.geometry import Brick, Capsule, Collision
from src.motion_planning import JointTrajectory
from src.motion_planning.rrt import RRTMap, RRTParameter, RRTPlanner, RobotRRTParameter
import src.motion_planning.rrt.rrt_planner as rrt_module


@pytest.mark.parametrize("time, expected", [[-0.1,[-0.2,0.1,0.3,-0.5,0.4,0.2]],[-0.051250000000000004,[-0.2,0.1,0.3,-0.5,0.4,0.2]],[0.8750000000000001,[-0.10107169121276144,0.04346953783586367,0.24346953783586367,-0.4010716912127614,0.2586738445896592,0.2706630777051704]],[1.85,[0.1499999999999998,-0.09999999999999992,0.10000000000000006,-0.1500000000000002,-0.09999999999999976,0.44999999999999984]],[2.825,[0.40107169121276104,-0.24346953783586348,-0.04346953783586349,0.10107169121276105,-0.45867384458965865,0.6293369222948293]],[3.75125,[0.49999999999999994,-0.30000000000000004,-0.10000000000000003,0.19999999999999996,-0.6,0.7]],[3.8,[0.49999999999999994,-0.30000000000000004,-0.10000000000000003,0.19999999999999996,-0.6,0.7]]])
def test_joint_trajectory_keeps_samples_recorded_before_cleanup(time, expected):
    trajectory = JointTrajectory(
        [-.2, .1, .3, -.5, .4, .2], [.5, -.3, -.1, .2, -.6, .7], 3.7,
    )
    np.testing.assert_allclose(trajectory.interpolate(time), expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("distance, angle, expected", [
    (0.0, 0.0, True), (.15, .3, True), (.3, 0.0, False),
    (.3, 1.2, True), (.7, 1.2, False),
])
def test_retained_collision_geometry_keeps_original_results(distance, angle, expected):
    capsule = Capsule(SE3.Trans(distance, 0, .1) * SE3.Ry(angle), .1, .4)
    brick = Brick(SE3(), np.array([.2, .3, .2]))
    assert bool(Collision.is_collision(capsule, brick)) is expected


def test_rrt_keeps_path_recorded_before_cleanup(monkeypatch):
    monkeypatch.setattr(rrt_module, "random", random.Random(31))
    planner = RRTPlanner(
        RRTMap([(-1.5, 1.5)] * 3, [Brick(SE3(), np.array([.2, .2, .2]))]),
        RRTParameter(start=[-1., 0., 0.], goal=[1., 0., 0.],
                     expand_dis=.25, goal_sample_rate=25, max_iter=500),
    )
    assert planner.success
    np.testing.assert_allclose(
        [node.get_t().tolist() for node in planner.get_final_course()],
        [[1,0,0],[0.8846990005622856,-0.012784061584031947,-0.0602685883918143],[0.6642026811051807,-0.0372317120011018,-0.17552346028430782],[0.4437063616480758,-0.06167936241817165,-0.2907783321768013],[0.261620797101524,-0.22522119661723522,-0.23979877107071168],[0.1960080106241799,-0.11314979994915966,-0.02617535405013942],[-0.05142382660879147,-0.14797211566115964,-0.034230926777770225],[-0.25,0,0],[-0.5,0,0],[-0.75,0,0],[-1,0,0]],
    )
    assert planner.get_path_length() == pytest.approx(2.3807289388339936, rel=0, abs=1e-12)


def test_robot_rrt_keeps_six_joint_collision_sampling():
    from experiments.pipeline_glare_task import UR5eWithBoard
    from src.config.robot import INITIAL_JOINTS

    robot = UR5eWithBoard()
    start = np.array(INITIAL_JOINTS)
    goal = start.copy()
    goal[0] += .3
    robot.set_joint(start)
    planner = RRTPlanner(
        RRTMap([(-2 * np.pi, 2 * np.pi)] * 6, []),
        RobotRRTParameter(start=start, goal=goal, robot=robot,
                          expand_dis=.15, goal_sample_rate=100, max_iter=5),
    )
    assert planner.success
    path = [node.get_t() for node in planner.get_final_course()]
    np.testing.assert_allclose(path[0], goal)
    np.testing.assert_allclose(path[-1], start)
    assert all(configuration.shape == (6,) for configuration in path)
