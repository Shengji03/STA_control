"""Numerical CARLA for exactly three STA multipliers.

Fixed grids, trapezoidal PDF integration, and exact inversion of piecewise-linear
densities. This is not the adaptive equal-probability grid in the paper appendix.
"""

from collections import deque

import numpy as np


class CarlaTuner:
    def __init__(
        self,
        bounds=((0.8, 1.5),) * 3,
        points=201,
        memory=50,
        seed=7,
        height=0.3,
        width=0.02,
    ):
        self.bounds = np.asarray(bounds, dtype=float)
        if (
            self.bounds.shape != (3, 2)
            or not np.all(np.isfinite(self.bounds))
            or np.any(self.bounds[:, 0] <= 0)
            or np.any(self.bounds[:, 1] <= self.bounds[:, 0])
        ):
            raise ValueError("CARLA needs three positive ordered search intervals")
        if points < 3 or memory < 2 or height <= 0 or width <= 0:
            raise ValueError("Invalid CARLA discretization or learning settings")
        self.grid = np.array([np.linspace(a, b, points) for a, b in self.bounds])
        self.pdf = np.array([np.ones(points) / (b - a) for a, b in self.bounds])
        self.history = deque(maxlen=memory)
        self.rng = np.random.default_rng(seed)
        self.height, self.width = float(height), float(width)
        self.updates = 0

    def initialize_prior(self, center, uniform_fraction=0.7, width_fraction=0.12):
        """Optional warm-start variant, explicitly mixing uniform exploration and a prior."""
        center = np.asarray(center, dtype=float)
        if (
            self.updates
            or center.shape != (3,)
            or not np.all(np.isfinite(center))
            or np.any(center < self.bounds[:, 0])
            or np.any(center > self.bounds[:, 1])
            or not 0 < uniform_fraction <= 1
            or width_fraction <= 0
        ):
            raise ValueError("Invalid CARLA warm-start prior")
        for i, (grid, (a, b)) in enumerate(zip(self.grid, self.bounds)):
            normal = np.exp(
                -0.5 * ((grid - center[i]) / (width_fraction * (b - a))) ** 2
            )
            normal /= np.trapz(normal, grid)
            self.pdf[i] = uniform_fraction / (b - a) + (1 - uniform_fraction) * normal

    def propose(self):
        actions = []
        for grid, pdf in zip(self.grid, self.pdf):
            dx = np.diff(grid)
            masses = 0.5 * (pdf[:-1] + pdf[1:]) * dx
            cumulative = np.cumsum(masses)
            z = float(self.rng.random()) * cumulative[-1]
            index = min(int(np.searchsorted(cumulative, z, side="right")), len(dx) - 1)
            area = z - (cumulative[index - 1] if index else 0.0)
            # Solve p0*t + slope*t^2/2 = area without cancellation near flat PDFs.
            p0 = pdf[index]
            slope = (pdf[index + 1] - p0) / dx[index]
            root = np.sqrt(max(0.0, p0 * p0 + 2 * slope * area))
            distance = 2 * area / max(p0 + root, np.finfo(float).tiny)
            actions.append(float(grid[index] + np.clip(distance, 0.0, dx[index])))
        return np.asarray(actions)

    def observe(self, ratios, cost, valid=True):
        ratios = np.asarray(ratios, dtype=float)
        if (
            ratios.shape != (3,)
            or np.any(ratios < self.bounds[:, 0])
            or np.any(ratios > self.bounds[:, 1])
            or not np.all(np.isfinite(ratios))
        ):
            raise ValueError("Candidate outside its CARLA search intervals")
        self.updates += 1
        if not valid or not np.isfinite(cost) or cost < 0:
            return 0.0
        # Zero is a valid cost; deque explicitly distinguishes samples from padding.
        self.history.append(float(cost))
        median, minimum = float(np.median(self.history)), min(self.history)
        span = median - minimum
        reward = float(np.clip((median - cost) / span, 0, 1)) if span > 1e-12 else 0.0
        for i, (grid, (a, b)) in enumerate(zip(self.grid, self.bounds)):
            kernel = (
                self.height
                / (b - a)
                * np.exp(-0.5 * ((grid - ratios[i]) / (self.width * (b - a))) ** 2)
            )
            updated = self.pdf[i] + reward * kernel
            updated /= np.trapz(updated, grid)
            self.pdf[i] = updated
        return reward

    def state(self):
        return {
            "bounds": self.bounds.tolist(),
            "grid": self.grid.tolist(),
            "pdf": self.pdf.tolist(),
            "history": list(self.history),
            "memory": self.history.maxlen,
            "height": self.height,
            "width": self.width,
            "updates": self.updates,
            "rng_state": self.rng.bit_generator.state,
        }

    @classmethod
    def from_state(cls, state):
        result = cls(
            state["bounds"],
            len(state["grid"][0]),
            state["memory"],
            height=state["height"],
            width=state["width"],
        )
        result.pdf = np.asarray(state["pdf"], dtype=float)
        if (
            result.pdf.shape != result.grid.shape
            or np.any(result.pdf < 0)
            or not np.all(np.isfinite(result.pdf))
        ):
            raise ValueError("Invalid saved CARLA density")
        if not np.allclose(
            [np.trapz(p, x) for p, x in zip(result.pdf, result.grid)], 1
        ):
            raise ValueError("Saved CARLA densities must integrate to one")
        result.history.extend(state["history"])
        result.updates = state["updates"]
        result.rng.bit_generator.state = state["rng_state"]
        return result
