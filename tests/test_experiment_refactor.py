import numpy as np
import mujoco
import pytest

from experiments.metrics import calculate_tracking_metrics
from experiments.pipeline_glare_task import ArmController
from src.config.paths import scene_path


def test_shared_controller_preserves_legacy_two_step_torques():
    model = mujoco.MjModel.from_xml_path(str(scene_path("scene4_pipeline")))
    data = mujoco.MjData(model)
    controller = ArmController(model, data, "L", "shoulder_pan_joint", "fingers_actuator")
    initial = controller.initial_q.copy()
    desired = initial + np.array([.01, -.02, .03, -.01, .015, -.005])
    velocity = np.array([.02, -.01, .03, -.02, .01, -.01])
    # Characterization values recorded from the original valve experiment.
    first = controller.compute_control(desired, initial, velocity, .002)
    second = controller.compute_control(initial, desired, velocity, .002)
    np.testing.assert_allclose(first, [
        22.264987946236495, -31.72376594448988, 39.44701996171133,
        -17.897813056217757, 21.971560899016655, -12.36602756463504,
    ])
    np.testing.assert_allclose(second, [
        -22.306672565322756, 31.706782647954864, -39.4743897487549,
        17.93464146771862, -21.965871701878402, 12.383656516890838,
    ])


def test_tracking_report_metrics_keep_existing_definition():
    metrics = calculate_tracking_metrics([[1.0, -2.0], [3.0, -4.0]], np.zeros((2, 2)), .1)
    np.testing.assert_allclose(metrics.mae, [2.0, 3.0])
    assert metrics.iae == pytest.approx(1.0)
    assert metrics.ise == pytest.approx(3.0)
    assert metrics.f_value == pytest.approx(2.0)
