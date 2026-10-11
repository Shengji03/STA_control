"""Physics-step metrics for skill and transition evaluation, independent of video."""

import numpy as np


def passes_validation(stats):
    """Shared acceptance gate for training results and runtime parameter banks."""
    interval = stats.get("gain_95pct_interval", [0, 0])
    return (
        len(interval) == 2
        and np.all(np.isfinite(interval))
        and stats.get("cases", 0) >= 8
        and stats.get("learned_successes", -1) == stats.get("cases")
        and stats.get("learned_successes", -1) >= stats.get("baseline_successes", 0)
        and stats.get("paired_successes", 0) >= 8
        and stats.get("mean_relative_gain_pct", 0) >= 2
        and interval[0] > 0
        and stats.get("worst_relative_gain_pct", -100) >= -10
    )


class ControlMetrics:
    def __init__(self, force_limit=60.0, error_limit=0.15):
        self.force_limit, self.error_limit = force_limit, error_limit
        self.count = 0
        self.error_sum = self.effort_sum = self.jitter_sum = self.force_sum = 0.0
        self.peak_force = self.peak_error = self.peak_torque = 0.0
        self.previous_torque = None
        self.violation = ""

    def sample(self, reference, actual, torque, wrench, torque_limits):
        error = np.asarray(reference) - actual
        torque, limits = np.asarray(torque), np.asarray(torque_limits)
        force = float(np.linalg.norm(wrench[:3]))
        if not all(np.all(np.isfinite(v)) for v in (error, torque, wrench)):
            self.violation = "nonfinite_control_state"
            return
        self.count += 1
        self.error_sum += float(np.mean(error**2))
        self.effort_sum += float(np.mean((torque / limits) ** 2))
        self.force_sum += force**2
        if self.previous_torque is not None:
            self.jitter_sum += float(
                np.mean(((torque - self.previous_torque) / limits) ** 2)
            )
        self.previous_torque = torque.copy()
        self.peak_force = max(self.peak_force, force)
        self.peak_error = max(self.peak_error, float(np.max(np.abs(error))))
        self.peak_torque = max(self.peak_torque, float(np.max(np.abs(torque))))
        if force > self.force_limit:
            self.violation = "force_limit"
        elif self.peak_error > self.error_limit:
            self.violation = "tracking_limit"

    def report(self):
        n = max(1, self.count)
        error = np.sqrt(self.error_sum / n)
        force = np.sqrt(self.force_sum / n)
        # Fixed normalizers/weights, shared by every candidate and baseline.
        cost = (
            (error / 0.005) ** 2
            + 0.2 * (force / 30.0) ** 2
            + 0.05 * self.effort_sum / n
            + 10 * self.jitter_sum / n
        )
        return {
            "samples": self.count,
            "joint_rmse_rad": float(error),
            "force_rms_N": float(force),
            "peak_force_N": self.peak_force,
            "peak_joint_error_rad": self.peak_error,
            "peak_actual_torque_Nm": self.peak_torque,
            "cost": float(cost),
            "effort_rms_fraction": float(np.sqrt(self.effort_sum / n)),
            "torque_delta_rms_fraction": float(np.sqrt(self.jitter_sum / n)),
            "violation": self.violation,
        }

    def sample_observation(
        self,
        error_mse,
        error_peak,
        effort_mse,
        jitter_mse,
        force,
        torque_peak,
        finite=True,
    ):
        """Reuse one physics-step observation across full, skill and handoff metrics.

        Jitter is always the difference of consecutive physics steps, including
        when a skill type recurs after an intervening skill.
        """
        if not finite:
            self.violation = "nonfinite_control_state"
            return
        self.count += 1
        self.error_sum += error_mse
        self.effort_sum += effort_mse
        self.jitter_sum += jitter_mse
        self.force_sum += force * force
        self.peak_force = max(self.peak_force, force)
        self.peak_error = max(self.peak_error, error_peak)
        self.peak_torque = max(self.peak_torque, torque_peak)
        if force > self.force_limit:
            self.violation = "force_limit"
        elif error_peak > self.error_limit:
            self.violation = "tracking_limit"


def paired_statistics(baseline, learned, seed=503, resamples=4000):
    """Paired scenario bootstrap; failed tasks are counted, not used to inflate gains."""
    if len(baseline) != len(learned) or not baseline:
        raise ValueError("Paired evaluations must have matching nonempty cases")
    pairs = [
        (b, candidate)
        for b, candidate in zip(baseline, learned)
        if b["valid"] and candidate["valid"]
    ]
    relative = np.asarray(
        [
            (b["cost"] - candidate["cost"]) / max(b["cost"], 1e-12)
            for b, candidate in pairs
        ]
    )
    if len(relative):
        rng = np.random.default_rng(seed)
        means = rng.choice(
            relative, size=(resamples, len(relative)), replace=True
        ).mean(axis=1)
        interval = np.percentile(means, [2.5, 97.5]) * 100
    else:
        interval = [0.0, 0.0]
    return {
        "cases": len(baseline),
        "paired_successes": len(pairs),
        "baseline_successes": sum(bool(x["valid"]) for x in baseline),
        "learned_successes": sum(bool(x["valid"]) for x in learned),
        "mean_relative_gain_pct": float(relative.mean() * 100)
        if len(relative)
        else 0.0,
        "gain_95pct_interval": [float(x) for x in interval],
        "worst_relative_gain_pct": float(relative.min() * 100)
        if len(relative)
        else 0.0,
        "relative_gains_pct": (relative * 100).tolist(),
        "method": "paired bootstrap over fixed independently generated evaluation conditions",
    }
