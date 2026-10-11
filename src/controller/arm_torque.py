"""Shared direct-torque actuation for the pipeline and standalone experiments."""

from functools import lru_cache

import mujoco
import numpy as np


@lru_cache(maxsize=8)
def _mapping(model, actuator_ids):
    ids = np.asarray(actuator_ids, dtype=int)
    joints = model.actuator_trnid[ids, 0]
    qpos = model.jnt_qposadr[joints]
    dofs = model.jnt_dofadr[joints]
    if (np.any(model.actuator_biastype[ids] != mujoco.mjtBias.mjBIAS_NONE)
            or not np.allclose(model.actuator_gainprm[ids, 0], 1)
            or not np.allclose(model.actuator_gear[ids, 0], 1)):
        raise ValueError('STA requires unit-gain direct joint-torque actuators')
    lower = np.maximum(model.actuator_ctrlrange[ids, 0], model.actuator_forcerange[ids, 0])
    upper = np.minimum(model.actuator_ctrlrange[ids, 1], model.actuator_forcerange[ids, 1])
    return qpos, dofs, lower, upper


def compute_arm_torque(model, data, controllers, actuator_ids, position, velocity,
                       measured_position=None, measured_velocity=None):
    qpos, dofs, lower, upper = _mapping(model, tuple(int(i) for i in actuator_ids))
    position, velocity = np.asarray(position), np.asarray(velocity)
    if position.shape != qpos.shape or velocity.shape != qpos.shape:
        raise ValueError('Joint reference dimensions do not match arm actuators')
    actual_position = data.qpos[qpos] if measured_position is None else np.asarray(measured_position)
    actual_velocity = data.qvel[dofs] if measured_velocity is None else np.asarray(measured_velocity)
    if actual_position.shape != qpos.shape or actual_velocity.shape != qpos.shape:
        raise ValueError('Joint measurement dimensions do not match arm actuators')
    feedback = np.array([controller.control(float(e), float(de))
                         for controller, e, de in zip(controllers,
                             position - actual_position, velocity - actual_velocity)])
    requested = data.qfrc_bias[dofs] + feedback
    if not np.all(np.isfinite(requested)):
        raise ValueError('Non-finite arm torque')
    target = np.clip(requested, lower, upper)
    # Fixed actuator-rate constraint prevents the discrete STA loop from flipping
    # between torque limits in a single 2 ms step. Included in saved ctrl state.
    step = 1000. * float(model.opt.timestep)
    previous = np.clip(data.ctrl[np.asarray(actuator_ids, dtype=int)], lower, upper)
    return np.clip(target, previous - step, previous + step)
