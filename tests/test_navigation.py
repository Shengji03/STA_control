import numpy as np
import pytest

from src.pipeline.navigation import NavigationTrajectory


def test_navigation_moves_before_rotating_and_holds_final_target():
    navigation = NavigationTrajectory(
        start=np.array([0.0, 0.0]), target=np.array([0.6, 0.0]),
        start_time=10.0, start_yaw=0.0, target_yaw=1.0,
    )
    position, yaw = navigation.sample(11.0)
    np.testing.assert_allclose(position, [0.3, 0.0])
    assert yaw is None
    position, yaw = navigation.sample(13.0)
    np.testing.assert_allclose(position, [0.6, 0.0])
    assert yaw == pytest.approx(0.5)
    position, yaw = navigation.sample(20.0)
    np.testing.assert_allclose(position, [0.6, 0.0])
    assert yaw == pytest.approx(1.0)
    assert navigation.duration == pytest.approx(4.0)


def test_in_place_rotation_keeps_position_and_minimum_duration():
    navigation = NavigationTrajectory(
        start=np.array([1.0, 2.0]), target=np.array([1.0, 2.0]),
        start_time=0.0, start_yaw=0.0, target_yaw=0.5,
    )
    position, yaw = navigation.sample(1.0)
    np.testing.assert_allclose(position, [1.0, 2.0])
    assert yaw == pytest.approx(0.25)
    assert navigation.duration == 2.0


def test_navigation_without_yaw_does_not_change_yaw_actuator():
    navigation = NavigationTrajectory(
        start=np.zeros(2), target=np.array([0.0, 0.6]), start_time=0.0, start_yaw=1.0,
    )
    position, yaw = navigation.sample(5.0)
    np.testing.assert_allclose(position, [0.0, 0.6])
    assert yaw is None
