"""Validated per-arm/per-skill STA multipliers; admittance is never tuned here."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .control_metrics import passes_validation

CONTROL_VERSION = "direct_torque_bias_slew_v1"
PARAMETER_NAMES = ("alpha", "beta", "lambda_s")


def scale_parameters(base, ratios):
    ratios = np.asarray(ratios, dtype=float)
    if ratios.shape != (3,) or not np.all(np.isfinite(ratios)) or np.any(ratios <= 0):
        raise ValueError("STA multipliers must be three positive finite values")
    result = {
        name: np.asarray(base[name], dtype=float) * ratio
        for name, ratio in zip(PARAMETER_NAMES, ratios)
    }
    if any(
        v.shape != (6,) or not np.all(np.isfinite(v)) or np.any(v <= 0)
        for v in result.values()
    ):
        raise ValueError("STA baselines must contain six positive values per parameter")
    return result


def baseline_fingerprint(base):
    payload = {
        name: np.asarray(base[name], dtype=float).tolist() for name in PARAMETER_NAMES
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class SkillParameterBank:
    def __init__(self, entries=None):
        self.entries = entries or {}

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("control_version") != CONTROL_VERSION:
            raise ValueError(
                "Learned parameters do not match the torque-control version"
            )
        if not payload.get("accepted_for_runtime"):
            raise ValueError(
                "This parameter bank has not passed composition validation"
            )
        evidence = payload.get("validation", {})
        if (
            payload.get("schema_version") != 2
            or evidence.get("protocol") != "skills_suite_v1"
        ):
            raise ValueError(
                "Expected a validated schema-2 skill/composition parameter bank"
            )
        bank = cls(payload["entries"])
        from ..skills.base_skill import SKILL_STA_PARAMS, SkillType

        for key, entry in bank.entries.items():
            baseline = SKILL_STA_PARAMS[SkillType[key.split(":")[1]]]
            for profile in entry["profiles"]:
                scale_parameters(profile["baseline"], profile["ratios"])
                if baseline_fingerprint(baseline) != baseline_fingerprint(
                    profile["baseline"]
                ):
                    raise ValueError(f"Baseline mismatch for learned parameters: {key}")
                proof = profile.get("validation", {})
                if not proof.get("accepted_for_workflow"):
                    if not np.array_equal(profile["ratios"], [1, 1, 1]):
                        raise ValueError(
                            "Unvalidated workflow must use fixed baseline parameters"
                        )
                elif not passes_validation(proof.get("statistics", {})):
                    raise ValueError(
                        "Workflow parameters lack independent complete-task improvement evidence"
                    )
        return bank

    @staticmethod
    def matches(scope, context):
        context = context or {}
        for field, plural in (
            ("scene", "scenes"),
            ("object", "objects"),
            ("action", "actions"),
            ("workflow", "workflows"),
            ("stage", "stages"),
        ):
            if plural in scope and context.get(field) not in scope[plural]:
                return False
        if "angle_min" in scope:
            angle = context.get("angle") or 0
            if not scope["angle_min"] <= abs(angle) <= scope["angle_max"]:
                return False
            if "angle_sign" in scope and np.sign(angle) != scope["angle_sign"]:
                return False
        if (
            "duration_min" in scope
            and not scope["duration_min"]
            <= (context.get("duration") or 0)
            <= scope["duration_max"]
        ):
            return False
        if "cart_pose" in scope:
            pose = context.get("cart_pose")
            if pose is None or np.any(
                np.abs(np.asarray(pose) - scope["cart_pose"]) > scope["cart_tolerance"]
            ):
                return False
        if "cart_poses" in scope:
            pose = context.get("cart_pose")
            if pose is None or not any(
                np.all(np.abs(np.asarray(pose) - target) <= scope["cart_tolerance"])
                for target in scope["cart_poses"]
            ):
                return False
        if "goal_count" in scope and context.get("goal_count") != scope["goal_count"]:
            return False
        if "goal_sets" in scope and sorted(context.get("goal_objects") or []) not in [
            sorted(group) for group in scope["goal_sets"]
        ]:
            return False
        if "goal_angle_min" in scope:
            for angle in context.get("goal_angles") or []:
                if (
                    not scope["goal_angle_min"] <= abs(angle) <= scope["goal_angle_max"]
                    or np.sign(angle) != scope["goal_angle_sign"]
                ):
                    return False
        return True

    def resolve(self, arm, skill_type, baseline, context=None):
        key = f"{arm}:{skill_type.name}"
        entry = self.entries.get(key)
        for profile in entry.get("profiles", [entry]) if entry else []:
            if not self.matches(profile.get("scope", {}), context):
                continue
            if baseline_fingerprint(baseline) != baseline_fingerprint(
                profile["baseline"]
            ):
                raise ValueError(f"Baseline mismatch for learned parameters: {key}")
            return scale_parameters(baseline, profile["ratios"])
        return scale_parameters(baseline, [1, 1, 1])
