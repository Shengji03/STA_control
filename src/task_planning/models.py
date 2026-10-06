from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class PlanningConfig:
    """Scores are dimensionless proxies, not probabilities of task success."""

    auxiliary_weight: float = 1.0
    reach_weight: float = 0.5
    posture_weight: float = 0.5
    max_auxiliaries: int | None = None
    exact_auxiliary_limit: int = 12
    max_assignments: int = 4096
    max_policies: int = 50000
    shade_heights: tuple[float, ...] = (0.75, 0.80, 0.85)
    shade_lateral_offsets: tuple[float, ...] = (-0.08, 0.0, 0.08)
    shade_gripper_closed: float = 0.70
    collision_samples: int = 8
    rotation_tolerance: float = 0.35

    def __post_init__(self):
        for name in ("auxiliary_weight", "reach_weight", "posture_weight"):
            value = getattr(self, name)
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.max_auxiliaries is not None and self.max_auxiliaries < 0:
            raise ValueError("max_auxiliaries must be nonnegative or None")
        if self.exact_auxiliary_limit < 0:
            raise ValueError("exact_auxiliary_limit must be nonnegative")
        if min(self.max_assignments, self.max_policies, self.collision_samples) < 1:
            raise ValueError("search limits and collision_samples must be positive")
        if not self.shade_heights or any(not 0.35 < z <= 0.9 for z in self.shade_heights):
            raise ValueError("shade_heights must be in (0.35, 0.9]")
        if not self.shade_lateral_offsets or any(not isfinite(x) or abs(x) > 0.10 for x in self.shade_lateral_offsets):
            raise ValueError("shade_lateral_offsets must be within the 0.10m coverage proxy")
        if not 0 < self.shade_gripper_closed <= 0.8:
            raise ValueError('shade_gripper_closed must be in (0, 0.8]')
        if not isfinite(self.rotation_tolerance) or self.rotation_tolerance <= 0:
            raise ValueError("rotation_tolerance must be positive")


@dataclass(frozen=True)
class Goal:
    id: str
    target: str
    operation: str
    angle: float = 0.0
    shade: str = "none"
    preferred_arm: str = "L"
    steps: tuple[dict, ...] = ()  # Preserved complete legacy bundle, if present.


@dataclass(frozen=True)
class Stage:
    id: int
    nav: dict | None
    goals: tuple[str, ...]
    parallel: bool = False


@dataclass(frozen=True)
class PlanningRequest:
    goals: tuple[Goal, ...]
    stages: tuple[Stage, ...]
    source: str = "semantic"


@dataclass(frozen=True)
class MainCandidate:
    id: str
    goal_id: str
    phase_id: int
    arm: str
    steps: tuple[dict, ...]
    score: float
    preferred: bool = False


@dataclass(frozen=True)
class AuxiliaryCandidate:
    id: str
    phase_id: int
    arm: str
    setup: tuple[dict, ...]
    cleanup: tuple[dict, ...]
    covers: frozenset[str]
    position: tuple[float, float, float]


@dataclass(frozen=True)
class Policy:
    assignments: tuple[MainCandidate, ...]
    auxiliaries: tuple[AuxiliaryCandidate, ...]
    main_score: float
    auxiliary_score: float
    score: float
    auxiliary_method: str

    @property
    def preferred_count(self):
        return sum(a.preferred for a in self.assignments)
