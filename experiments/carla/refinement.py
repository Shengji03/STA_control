"""Recover rejected compositions with physical corrections and revalidated CARLA candidates.

Previously inspected holdout is a diagnostic/regression set. Deployment decisions
use newly registered scenarios, with parameter selection frozen before holdout.
"""

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

from experiments.carla.benchmarks import Scenario, complete_evaluation, make_bank
from experiments.carla.suite import (
    active_profiles,
    fingerprints,
    make_cases,
    passed,
    rotation_prior,
    scenario_from_dict,
)
from experiments.carla.training import save_json
from src.config.paths import DEFAULT_STA_PARAMETERS, OUTPUTS_DIR
from src.controller.control_metrics import paired_statistics
from src.controller.control_parameters import SkillParameterBank

FAMILIES = ("assembly", "repeat", "shade")


def fresh_cases(family, split, count, case_set):
    cases = make_cases(f"{case_set}_{family}_{split}", family, count)
    if family != "assembly":
        angles = [90, 110, 120, 135, 150, 170, 180, 120]
        cases = [
            Scenario(**{**asdict(c), "angle": angles[i % len(angles)] * np.pi / 180})
            for i, c in enumerate(cases)
        ]
    return [asdict(c) for c in cases]


def evaluate_case(case, parameters):
    result = complete_evaluation(scenario_from_dict(case), parameters)
    result["complete_cost"] = result["cost"]
    return result


def batch(pool, cases, parameters):
    futures = [pool.submit(evaluate_case, c, parameters) for c in cases]
    return [f.result() for f in futures]


def eligible(results, baseline):
    return all(r["valid"] for r in results) and all(
        not b["valid"] or r["cost"] <= b["cost"] * 1.1
        for b, r in zip(baseline, results)
    )


def assembly_choices(old, parameters, profile):
    choices = []
    seen = set()
    for path in sorted((old / "profiles" / profile).glob("seed_*/checkpoint.json")):
        cp = json.loads(path.read_text(encoding="utf-8"))
        candidate = cp["best"]
        ratios = tuple(candidate["ratios"])
        if ratios in seen:
            continue
        seen.add(ratios)
        choices.append(
            (
                f"{profile}_{path.parent.name}_candidate_{candidate['iteration']}",
                {**parameters, profile: list(ratios)},
            )
        )
    return choices


def selection(old, family, cases, pool, base_bank):
    baselines = batch(pool, cases, {})
    best_parameters = {}
    best_results = baselines
    best_score = float(np.mean([r["cost"] for r in baselines]))
    records = []
    library = json.loads((old / "skill_library.json").read_text(encoding="utf-8"))[
        "parameters"
    ]
    rotation_old = rotation_prior(base_bank)
    if family == "assembly":
        blocks = [
            "assembly_L_MOVE",
            "assembly_L_INSERT",
            "assembly_L_TRANSLATE",
            "assembly_L_EXTRACT",
        ]
        for profile in blocks:
            current = deepcopy(best_parameters)
            for name, parameters in assembly_choices(old, current, profile):
                results = batch(pool, cases, parameters)
                score = float(np.mean([r["cost"] for r in results]))
                safe = eligible(results, baselines)
                records.append(
                    {
                        "candidate": name,
                        "parameters": parameters,
                        "eligible": safe,
                        "mean_cost": score,
                        "results": results,
                    }
                )
                print("RECOVERY_SELECT", family, name, safe, score, flush=True)
                if safe and score < best_score:
                    best_parameters, best_results, best_score = (
                        parameters,
                        results,
                        score,
                    )
    else:
        if family == "repeat":
            choices = [("validated_skill_library", library)]
            for path in sorted(
                (old / "combinations" / family).glob("seed_*/checkpoint.json")
            ):
                cp = json.loads(path.read_text(encoding="utf-8"))
                ratios = np.asarray(cp["best"]["ratios"])
                parameters = dict(library)
                for profile in active_profiles(family):
                    parameters[profile.id] = (
                        np.asarray(library.get(profile.id, [1.0, 1.0, 1.0])) * ratios
                    ).tolist()
                choices.append((f"previous_joint_CARLA_{path.parent.name}", parameters))
        else:
            choices = []
            for label, rotation in [
                ("deployed_rotation", rotation_old),
                ("library_rotation", library.get("mobile_L_ROTATE", [1.0, 1.0, 1.0])),
            ]:
                choices.append((label, {"mobile_L_ROTATE": rotation}))
                if "mobile_R_MOVE" in library:
                    choices.append(
                        (
                            label + "_cleanup",
                            {
                                "mobile_L_ROTATE": rotation,
                                "mobile_R_MOVE_cleanup": library["mobile_R_MOVE"],
                            },
                        )
                    )
                if "mobile_L_GRASP_open" in library:
                    choices.append(
                        (
                            label + "_release",
                            {
                                "mobile_L_ROTATE": rotation,
                                "mobile_L_GRASP_open": library["mobile_L_GRASP_open"],
                            },
                        )
                    )
        for name, parameters in choices:
            results = batch(pool, cases, parameters)
            score = float(np.mean([r["cost"] for r in results]))
            safe = eligible(results, baselines)
            records.append(
                {
                    "candidate": name,
                    "parameters": parameters,
                    "eligible": safe,
                    "mean_cost": score,
                    "results": results,
                }
            )
            print("RECOVERY_SELECT", family, name, safe, score, flush=True)
            if safe and score < best_score:
                best_parameters, best_results, best_score = parameters, results, score
    return {
        "parameters": best_parameters,
        "baseline": baselines,
        "selected_results": best_results,
        "mean_selection_cost": best_score,
        "candidates": records,
    }


def scopes_for(family, parameters, stats):
    profiles = []
    for profile in active_profiles(family):
        scope = profile.scope()
        scope["workflows"] = [family]
        if family != "assembly":
            scope.update(
                goal_count=2 if family == "repeat" else 1,
                goal_sets=[["valve_1", "valve_1"]]
                if family == "repeat"
                else [["valve_1"]],
                goal_angle_min=np.pi / 2 - 0.02,
                goal_angle_max=np.pi + 0.02,
                goal_angle_sign=1,
            )
        entry = make_bank({profile.id: [1, 1, 1]}).entries[
            f"{profile.arm}:{profile.skill}"
        ]["profiles"][0]
        entry.update(
            id=f"{family}:{profile.id}",
            ratios=parameters.get(profile.id, [1.0, 1.0, 1.0]),
            scope=scope,
            validation={"accepted_for_workflow": True, "statistics": stats},
        )
        profiles.append((f"{profile.arm}:{profile.skill}", entry))
    if "mobile_R_MOVE_cleanup" in parameters:
        entry = make_bank(
            {"mobile_R_MOVE_cleanup": parameters["mobile_R_MOVE_cleanup"]}
        ).entries["R:MOVE"]["profiles"][0]
        entry["scope"].update(
            goal_count=1,
            goal_sets=[["valve_1"]],
            goal_angle_min=np.pi / 2 - 0.02,
            goal_angle_max=np.pi + 0.02,
            goal_angle_sign=1,
        )
        entry.update(validation={"accepted_for_workflow": True, "statistics": stats})
        profiles.insert(0, ("R:MOVE", entry))
    return profiles


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--previous",
        type=Path,
        required=True,
        help="Completed multi-skill study with checkpoints and skill_library.json",
    )
    parser.add_argument("--base-parameters", type=Path, default=DEFAULT_STA_PARAMETERS)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--case-set", help="Unique label for new selection/holdout conditions"
    )
    parser.add_argument(
        "--families", choices=FAMILIES, nargs="+", default=list(FAMILIES)
    )
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    if len(args.families) != len(set(args.families)):
        parser.error("Duplicate families")
    if not (args.previous / "skill_library.json").is_file():
        parser.error("--previous must contain skill_library.json")
    base_bank = SkillParameterBank.load(args.base_parameters)
    bank = json.loads(args.base_parameters.read_text(encoding="utf-8"))
    args.output = args.output or OUTPUTS_DIR / "carla" / datetime.now().strftime(
        "refinement_%Y%m%d_%H%M%S_%f"
    )
    case_set = args.case_set or args.output.name
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError("Recovery output must be a new directory")
    protocol = {
        "kind": "physical_corrections_and_reselection_of_pretrained_CARLA_candidates",
        "previous": str(args.previous.resolve()),
        "case_set": case_set,
        "base_parameters_sha256": hashlib.sha256(
            args.base_parameters.read_bytes()
        ).hexdigest(),
        "provenance": {
            **fingerprints(),
            "experiments/carla/refinement.py": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
        },
        "physical_corrections": {
            "assembly": "pin A closure 0.6 -> 0.575; block/pin B unchanged",
            "repeat": "gripper wrist alignment to current handwheel angle modulo 90 degrees",
            "shade": "main rotation and assist cleanup parameters separated; assist setup held at baseline",
        },
        "cases": {
            f: {
                split: fresh_cases(
                    f, split, 8 if split == "selection" else 32, case_set
                )
                for split in ("selection", "holdout")
            }
            for f in args.families
        },
    }
    save_json(args.output / "protocol.json", protocol)
    frozen = {}
    holdouts = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for family in args.families:
            frozen[family] = selection(
                args.previous,
                family,
                protocol["cases"][family]["selection"],
                pool,
                base_bank,
            )
            save_json(args.output / f"{family}_selection.json", frozen[family])
        save_json(args.output / "frozen_selection.json", frozen)
        print("RECOVERY_SELECTION_FROZEN", flush=True)
        for family in args.families:
            cases = protocol["cases"][family]["holdout"]
            baseline = batch(pool, cases, {})
            learned = batch(pool, cases, frozen[family]["parameters"])
            stats = paired_statistics(baseline, learned)
            holdouts[family] = {
                "baseline": baseline,
                "learned": learned,
                "statistics": stats,
                "accepted": passed(stats),
                "parameters": frozen[family]["parameters"],
            }
            save_json(args.output / f"{family}_holdout.json", holdouts[family])
            print(
                "RECOVERY_HOLDOUT",
                family,
                json.dumps(stats),
                "ACCEPTED",
                passed(stats),
                flush=True,
            )
    for family, result in holdouts.items():
        if not result["accepted"]:
            continue
        for entry in bank["entries"].values():
            entry["profiles"] = [
                p
                for p in entry["profiles"]
                if p.get("scope", {}).get("workflows") != [family]
            ]
        for key, profile in scopes_for(
            family, result["parameters"], result["statistics"]
        ):
            bank["entries"].setdefault(key, {"profiles": []})["profiles"].append(
                profile
            )
        bank["validation"]["compositions"][family] = {
            "accepted": True,
            "statistics": result["statistics"],
            "recovery_evidence": str(args.output.resolve()),
        }
    bank["validation"]["recovery"] = {
        "protocol": protocol["kind"],
        "results": {
            f: {"accepted": r["accepted"], "statistics": r["statistics"]}
            for f, r in holdouts.items()
        },
    }
    save_json(args.output / "best_parameters.json", bank)
    SkillParameterBank.load(args.output / "best_parameters.json")
    save_json(
        args.output / "summary.json",
        {
            "selection": frozen,
            "holdout": holdouts,
            "physical_corrections": protocol["physical_corrections"],
        },
    )
    print("RECOVERY_FINISHED", args.output.resolve(), flush=True)
    print(
        "Deploy explicitly with: python -m experiments.carla install",
        args.output / "best_parameters.json",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
