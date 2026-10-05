"""Tracking metrics shared by experiment reporting."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrackingMetrics:
    mae: np.ndarray
    iae: float
    ise: float
    f_value: float


def calculate_tracking_metrics(desired, actual, timestep: float) -> TrackingMetrics:
    desired = np.asarray(desired, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if desired.ndim != 2 or actual.shape != desired.shape or desired.shape[0] == 0:
        raise ValueError("Tracking histories must be nonempty arrays of matching shape")
    if timestep <= 0:
        raise ValueError("Timestep must be positive")
    errors = desired - actual
    iae = float(np.sum(np.abs(errors)) * timestep)
    ise = float(np.sum(errors ** 2) * timestep)
    return TrackingMetrics(np.mean(np.abs(errors), axis=0), iae, ise, 0.5 * (iae + ise))
