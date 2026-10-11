"""Multi-skill CARLA study: real prefixes, frozen selection, and trained compositions.

Candidate windows are control evaluations, not claimed complete-task successes.
Runtime parameters require complete-task selection and independently generated holdout.
"""

import argparse
import hashlib
import io
import json
import os
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from datetime import datetime
from math import pi
from pathlib import Path
from time import monotonic, sleep

import mujoco
import numpy as np

from experiments.carla.benchmarks import (
    PROFILES,
    PrefixState,
    Profile,
    Scenario,
    complete_evaluation,
    make_bank,
)
from experiments.carla.training import (
    atomic_replace,
    provenance,
    save_json,
    search_parameters,
)
from src.config.paths import DEFAULT_STA_PARAMETERS, OUTPUTS_DIR, PROJECT_ROOT
from src.controller.carla_tuner import CarlaTuner
from src.controller.control_metrics import paired_statistics
from src.controller.control_metrics import passes_validation as passed
from src.controller.control_parameters import CONTROL_VERSION, SkillParameterBank

FAMILIES = (
    "assembly",
    "valve_L",
    "valve_R",
    "shade",
    "two_valves",
    "repeat",
    "inspect",
    "rotate_inspect",
)
ASSEMBLY_ORDERS = ((0, 1, 2), (1, 0, 2), (0, 2, 1))
_prefixes = OrderedDict()
_full_results = OrderedDict()


def json_value(value):
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def scenario_from_dict(value):
    return Scenario(
        **{
            **value,
            "order": tuple(value.get("order", (0, 1, 2))),
            "arm_order": tuple(value.get("arm_order", ("L", "R"))),
        }
    )


def case_rng(label):
    return np.random.default_rng(
        int.from_bytes(
            hashlib.sha256(("sta_carla:" + label).encode()).digest()[:8], "little"
        )
    )


def make_cases(label, family, count):
    rng, cases = case_rng(label), []
    for i in range(count):
        assembly = family == "assembly"
        speed = float(rng.uniform(0.95, 1.05) if assembly else rng.uniform(0.9, 1.1))
        cases.append(
            Scenario(
                f"{label}_{i:02d}",
                family=family,
                scene="scene3.xml"
                if assembly
                else "scene5_glare.xml"
                if family in ("shade", "inspect", "rotate_inspect")
                else ("scene4_pipeline.xml" if i % 3 == 1 else "scene5_glare.xml"),
                speed=speed,
                payload=float(rng.uniform(0.9, 1.1)),
                resistance=float(rng.uniform(0.85, 1.2)),
                grasp=float(
                    rng.uniform(0.97, 1.03) if assembly else rng.uniform(0.85, 1.15)
                ),
                angle=float(rng.choice([pi / 2, 2 * pi / 3, 3 * pi / 4, pi])),
                target="valve_2"
                if i % 3 == 2 and family in ("valve_L", "valve_R")
                else "valve_1",
                cart_offset=0.0 if assembly else float(rng.uniform(-0.01, 0.01)),
                order=ASSEMBLY_ORDERS[i % 3] if assembly else (0, 1, 2),
                arm_order=(("L", "L"), ("L", "R"), ("R", "L"), ("R", "R"))[i % 4],
            )
        )
    return cases


def profile_cases(profile, split, count, case_set):
    if profile.domain == "assembly":
        # Prefix indexes retain the actual default sequence; alternate valid orders
        # are exercised in complete-task selection, holdout, and composition tuning.
        cases = make_cases(f"{case_set}_{profile.id}_{split}", "assembly", count)
        if split == "train":
            cases = [Scenario(**{**asdict(c), "order": (0, 1, 2)}) for c in cases]
        return cases
    if profile.arm == "L":
        families = ("valve_L", "shade") if profile.skill != "ROTATE" else ("valve_L",)
    else:
        families = (
            ("shade", "valve_R", "inspect")
            if profile.skill == "MOVE"
            else ("shade", "valve_R")
            if profile.skill == "GRASP"
            else ("valve_R",)
        )
    cases = []
    counters = {family: 0 for family in families}
    for i in range(count):
        family = families[i % len(families)]
        case = make_cases(f"{case_set}_{profile.id}_{split}_{i}", family, 1)[0]
        if family in ("valve_L", "valve_R"):
            ordinal = counters[family]
            case = Scenario(
                **{
                    **asdict(case),
                    "target": "valve_2" if ordinal % 2 else "valve_1",
                    "scene": "scene4_pipeline.xml"
                    if ordinal % 2
                    else "scene5_glare.xml",
                }
            )
        counters[family] += 1
        cases.append(case)
    # Explicitly retain both objects/scenes for the main arm, independent of RNG.
    if len(families) == 1:
        cases = make_cases(f"{case_set}_{profile.id}_{split}", families[0], count)
    return cases


def occurrences(profile, cases):
    if profile.domain == "assembly" and profile.skill == "MOVE":
        return [0, 2, 3, 7, 10][: len(cases)]
    if profile.domain == "assembly" and profile.skill == "GRASP":
        return [0, 1, 2, 0, 2][: len(cases)]
    if profile.skill == "MOVE":
        return [
            min(i % 3, 1) if c.family == "inspect" else i % 3
            for i, c in enumerate(cases)
        ]
    return [0] * len(cases)


def fingerprints():
    result = provenance()
    for name in (
        "experiments/carla/benchmarks.py",
        "experiments/carla/suite.py",
        "experiments/long_sequence_task.py",
        "src/assets/scenes/scene3.xml",
        "src/assets/scenes/scene4_pipeline.xml",
        "src/assets/universal_robots_ur5e/ur5e.xml",
        "src/robot/ur5e.py",
        "src/robot/robot.py",
        "src/robot/robot_config.py",
        "src/motion_planning/joint_trajectory.py",
        "src/config/sta_skill_parameters.json",
    ):
        result[name] = hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest()
    return result


def rotation_prior(bank=None):
    """Warm start from the shipped, validated shade rotation profile."""
    bank = bank or SkillParameterBank.load(DEFAULT_STA_PARAMETERS)
    for profile in bank.entries.get("L:ROTATE", {}).get("profiles", []):
        if profile.get("scope", {}).get("workflows") == ["shade"]:
            return list(profile["ratios"])
    return [1.0, 1.0, 1.0]


def profile_config(profile, seed, smoke=False, case_set="default"):
    train = profile_cases(profile, "train", 5, case_set)
    return json_value(
        {
            "protocol": "skills_suite_v1",
            "control_version": CONTROL_VERSION,
            "profile": asdict(profile),
            "seed": seed,
            "case_set": case_set,
            "train_cases": [asdict(c) for c in train],
            "occurrences": occurrences(profile, train),
            "selection_cases": [
                asdict(c)
                for c in profile_cases(
                    profile, "selection", 4 if smoke else 8, case_set
                )
            ],
            "holdout_cases": [
                asdict(c)
                for c in profile_cases(profile, "holdout", 4 if smoke else 16, case_set)
            ],
            "bounds": [[0.7, 1.6], [0.7, 2.4], [0.8, 2.1]],
            "points": 401,
            "memory": 100,
            "training_window": "one actual skill plus 2 s of handoff; whole-task effects required later",
            "force_stop_N": 60.0
            if profile.domain == "mobile" and profile.skill == "ROTATE"
            else 120.0,
            "joint_error_stop_rad": 0.15,
            "provenance": fingerprints(),
            "mujoco_version": mujoco.__version__,
        }
    )


def training_case(case_dict, profile_dict, occurrence, ratios, prefix_dir):
    case, profile = scenario_from_dict(case_dict), Profile(**profile_dict)
    key = json.dumps([case_dict, profile_dict, occurrence], sort_keys=True)
    try:
        if key not in _prefixes:
            with redirect_stdout(io.StringIO()):
                state = PrefixState(case, profile, occurrence)
            _prefixes[key] = state
            _prefixes.move_to_end(key)
            while len(_prefixes) > 5:
                _prefixes.popitem(last=False)
            directory = Path(prefix_dir)
            directory.mkdir(parents=True, exist_ok=True)
            name = hashlib.sha256(key.encode()).hexdigest()[:12]
            path = directory / f"{name}.npz"
            if not path.exists():
                temp = path.with_name(f"{name}.{os.getpid()}.tmp")
                with temp.open("wb") as stream:
                    np.savez_compressed(stream, integration_state=state.physical)
                atomic_replace(temp, path)
                save_json(
                    directory / f"{name}.json",
                    {
                        "case": case_dict,
                        "profile": profile_dict,
                        "occurrence": occurrence,
                        "integration_specification": int(state.specification),
                        "arms": {
                            label: {
                                "controllers": saved["controllers"],
                                "ctx": saved["ctx"]
                                and {
                                    k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                    for k, v in saved["ctx"].items()
                                },
                            }
                            for label, saved in state.arms.items()
                        },
                    },
                )
        state = _prefixes[key]
        with redirect_stdout(io.StringIO()):
            return state.evaluate(ratios)
    except (ValueError, RuntimeError, FloatingPointError) as error:
        return {
            "case": case_dict,
            "profile": profile.id,
            "valid": False,
            "state": "EVALUATION_ERROR",
            "cost": 1e6,
            "error": str(error),
        }


def full_case(case_dict, parameters, profile_dict=None, record=False):
    key = json.dumps([case_dict, parameters, profile_dict], sort_keys=True)
    if not record and key in _full_results:
        return deepcopy_result(_full_results[key])
    result = complete_evaluation(
        scenario_from_dict(case_dict), parameters, record=record
    )
    result["complete_cost"] = result["cost"]
    if profile_dict:
        profile = Profile(**profile_dict)
        matching = [
            metrics
            for name, metrics in result["per_skill"].items()
            if name.split(":")[0] == profile.arm
            and name.split(":")[1] == profile.skill
            and (profile.action is None or name.split(":")[2] == profile.action)
        ]
        if matching:
            targeted = float(np.mean([r["cost"] for r in matching]))
            result["cost"] = targeted + 0.25 * result["complete_cost"]
            result["target_cost"] = targeted
        else:
            result.update(
                valid=False,
                cost=1e6,
                error="Requested skill did not run in this complete task",
            )
    if not record:
        _full_results[key] = result
        while len(_full_results) > 256:
            _full_results.popitem(last=False)
    return deepcopy_result(result)


def deepcopy_result(value):
    # Never attach optimizer labels to a shared cached evaluation.
    return json_value(value)


def full_batch(pool, cases, parameters, profile=None):
    futures = [pool.submit(full_case, c, parameters, profile) for c in cases]
    return [future.result() for future in futures]


class RemoteState:
    def __init__(self, case):
        self.case = scenario_from_dict(case)


def eligible(results, baselines):
    return sum(r["valid"] for r in results) >= sum(
        r["valid"] for r in baselines
    ) and all(
        not b["valid"]
        or (r["valid"] and r["complete_cost"] <= b["complete_cost"] * 1.1)
        for b, r in zip(baselines, results)
    )


def train_profile(
    profile_dict, seed, iterations, case_workers, directory, resume, smoke, case_set
):
    profile, output = Profile(**profile_dict), Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    cfg = profile_config(profile, seed, smoke, case_set)
    cp_path, summary_path = output / "checkpoint.json", output / "seed_summary.json"
    checkpoint = (
        json.loads(cp_path.read_text(encoding="utf-8"))
        if resume and cp_path.exists()
        else None
    )
    if checkpoint and checkpoint["configuration"] != cfg:
        raise ValueError("Skill study checkpoint/source/configuration mismatch")
    if checkpoint and checkpoint["completed"] >= iterations and summary_path.exists():
        result = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            result["completed"] == checkpoint["completed"]
            and result["configuration"] == cfg
        ):
            return result
    with (
        (output / "console.log").open("a", encoding="utf-8", buffering=1) as log,
        redirect_stdout(log),
        redirect_stderr(log),
        ProcessPoolExecutor(max_workers=case_workers) as pool,
    ):

        def evaluate_cases(ratios):
            futures = [
                pool.submit(
                    training_case,
                    c,
                    profile_dict,
                    o,
                    list(ratios),
                    str(output / "prefixes"),
                )
                for c, o in zip(cfg["train_cases"], cfg["occurrences"])
            ]
            return [f.result() for f in futures]

        tuner = CarlaTuner(cfg["bounds"], points=401, memory=100, seed=seed)
        prior = rotation_prior() if profile.id == "mobile_L_ROTATE" else [1.0, 1.0, 1.0]
        tuner.initialize_prior(prior)
        result = search_parameters(
            [RemoteState(c) for c in cfg["train_cases"]],
            tuner,
            iterations,
            output,
            cfg,
            checkpoint,
            initial_candidate=prior,
            label=f"{profile.id} seed {seed}",
            case_evaluator=evaluate_cases,
            require_baseline=False,
        )
        baselines = full_batch(pool, cfg["selection_cases"], {}, profile_dict)
        selected = {"ratios": [1.0, 1.0, 1.0], "iteration": 0, "origin": "baseline"}
        best_score = float(np.mean([r["cost"] for r in baselines]))
        selected_results = baselines
        candidates = []
        seen = set()
        for candidate in result["leaders"][:3]:
            ratios = tuple(candidate["ratios"])
            if ratios in seen or np.allclose(ratios, [1, 1, 1]):
                continue
            seen.add(ratios)
            evaluations = full_batch(
                pool, cfg["selection_cases"], {profile.id: list(ratios)}, profile_dict
            )
            score = float(np.mean([r["cost"] for r in evaluations]))
            safe = eligible(evaluations, baselines)
            candidates.append(
                {
                    "ratios": list(ratios),
                    "iteration": candidate["iteration"],
                    "eligible": safe,
                    "mean_cost": score,
                    "results": evaluations,
                }
            )
            print("SELECT", candidate["iteration"], safe, score, flush=True)
            if safe and score < best_score:
                best_score, selected_results = score, evaluations
                selected = {
                    k: candidate.get(k) for k in ("ratios", "iteration", "origin")
                }
        summary = {
            "profile": profile_dict,
            "seed": seed,
            "completed": result["completed"],
            "configuration": cfg,
            "best_training": result["best"],
            "baseline_training": result["baseline"],
            "selected": selected,
            "mean_selection_cost": best_score,
            "baseline_selection": baselines,
            "selected_results": selected_results,
            "selection_candidates": candidates,
        }
        save_json(summary_path, summary)
        return summary


def holdout_profile(seed_result, case_workers, directory):
    output = Path(directory)
    path = output / "holdout.json"
    cfg = seed_result["configuration"]
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if (
            old["configuration"] == cfg
            and old["ratios"] == seed_result["selected"]["ratios"]
        ):
            return old
    profile = seed_result["profile"]
    pid = profile["id"]
    with ProcessPoolExecutor(max_workers=case_workers) as pool:
        baselines = full_batch(pool, cfg["holdout_cases"], {}, profile)
        learned = full_batch(
            pool,
            cfg["holdout_cases"],
            {pid: seed_result["selected"]["ratios"]},
            profile,
        )
    whole_baseline = [{**r, "cost": r["complete_cost"]} for r in baselines]
    whole_learned = [{**r, "cost": r["complete_cost"]} for r in learned]
    result = {
        "profile": profile,
        "seed": seed_result["seed"],
        "ratios": seed_result["selected"]["ratios"],
        "configuration": cfg,
        "baseline": baselines,
        "learned": learned,
        "statistics": paired_statistics(baselines, learned),
        "complete_statistics": paired_statistics(whole_baseline, whole_learned),
    }
    save_json(path, result)
    return result


def active_profiles(family):
    if family == "assembly":
        return [p for p in PROFILES if p.domain == "assembly"]
    arms = (
        ("R",)
        if family in ("valve_R", "inspect")
        else ("L",)
        if family == "valve_L"
        else ("L", "R")
    )
    return [
        p
        for p in PROFILES
        if p.domain == "mobile"
        and p.arm in arms
        and (family != "inspect" or p.skill == "MOVE")
    ]


def scaled_library(library, family, ratios):
    result = dict(library)
    for profile in active_profiles(family):
        result[profile.id] = (
            np.asarray(library.get(profile.id, [1.0, 1.0, 1.0])) * ratios
        ).tolist()
    return result


def combo_config(family, seed, library, smoke, case_set):
    return json_value(
        {
            "protocol": "skills_composition_v1",
            "control_version": CONTROL_VERSION,
            "family": family,
            "seed": seed,
            "case_set": case_set,
            "library": library,
            "bounds": [[0.85, 1.15]] * 3,
            "train_cases": [
                asdict(c)
                for c in make_cases(f"{case_set}_{family}_combo_train", family, 5)
            ],
            "selection_cases": [
                asdict(c)
                for c in make_cases(
                    f"{case_set}_{family}_combo_selection", family, 4 if smoke else 8
                )
            ],
            "holdout_cases": [
                asdict(c)
                for c in make_cases(
                    f"{case_set}_{family}_combo_holdout", family, 4 if smoke else 16
                )
            ],
            "provenance": fingerprints(),
            "learning": "three shared STA multipliers around independently validated skill library; full tasks every candidate",
        }
    )


def train_combo(
    family, seed, iterations, case_workers, library, directory, resume, smoke, case_set
):
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    cfg = combo_config(family, seed, library, smoke, case_set)
    cp_path, summary_path = output / "checkpoint.json", output / "seed_summary.json"
    checkpoint = (
        json.loads(cp_path.read_text(encoding="utf-8"))
        if resume and cp_path.exists()
        else None
    )
    if checkpoint and checkpoint["configuration"] != cfg:
        raise ValueError("Composition checkpoint mismatch")
    if checkpoint and checkpoint["completed"] >= iterations and summary_path.exists():
        result = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            result["completed"] == checkpoint["completed"]
            and result["configuration"] == cfg
        ):
            return result
    with (
        (output / "console.log").open("a", encoding="utf-8", buffering=1) as log,
        redirect_stdout(log),
        redirect_stderr(log),
        ProcessPoolExecutor(max_workers=case_workers) as pool,
    ):

        def evaluate_cases(ratios):
            return full_batch(
                pool,
                cfg["train_cases"],
                scaled_library(library, family, np.asarray(ratios)),
            )

        tuner = CarlaTuner(cfg["bounds"], points=401, memory=100, seed=seed)
        tuner.initialize_prior([1, 1, 1])
        search = search_parameters(
            [RemoteState(c) for c in cfg["train_cases"]],
            tuner,
            iterations,
            output,
            cfg,
            checkpoint,
            label=f"combo {family} seed {seed}",
            case_evaluator=evaluate_cases,
            require_baseline=False,
        )
        fixed = full_batch(pool, cfg["selection_cases"], {})
        assembled = full_batch(pool, cfg["selection_cases"], library)
        selected = {
            "ratios": [1.0, 1.0, 1.0],
            "parameters": {},
            "iteration": 0,
            "origin": "fixed_baseline",
        }
        best_score = float(np.mean([r["cost"] for r in fixed]))
        best_results = fixed
        candidates = []
        choices = [
            {"ratios": [1.0, 1.0, 1.0], "iteration": -1, "origin": "stitched_library"}
        ] + search["leaders"][:3]
        seen = set()
        for candidate in choices:
            ratios = tuple(candidate["ratios"])
            if ratios in seen:
                continue
            seen.add(ratios)
            parameters = scaled_library(library, family, np.asarray(ratios))
            results = (
                assembled
                if ratios == (1.0, 1.0, 1.0)
                else full_batch(pool, cfg["selection_cases"], parameters)
            )
            score = float(np.mean([r["cost"] for r in results]))
            safe = eligible(results, fixed)
            candidates.append(
                {
                    "ratios": list(ratios),
                    "iteration": candidate["iteration"],
                    "mean_cost": score,
                    "eligible": safe,
                    "results": results,
                }
            )
            if safe and score < best_score:
                best_score, best_results = score, results
                selected = {**candidate, "parameters": parameters}
                selected.pop("evaluation", None)
        result = {
            "family": family,
            "seed": seed,
            "completed": search["completed"],
            "configuration": cfg,
            "best_training": search["best"],
            "baseline_training": search["baseline"],
            "selected": selected,
            "mean_selection_cost": best_score,
            "baseline_selection": fixed,
            "selected_results": best_results,
            "assembled_selection": assembled,
            "selection_candidates": candidates,
        }
        save_json(summary_path, result)
        return result


def holdout_combo(seed_result, case_workers, directory):
    output = Path(directory)
    path = output / "holdout.json"
    cfg = seed_result["configuration"]
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if (
            old["configuration"] == cfg
            and old["parameters"] == seed_result["selected"]["parameters"]
        ):
            return old
    with ProcessPoolExecutor(max_workers=case_workers) as pool:
        baselines = full_batch(pool, cfg["holdout_cases"], {})
        stitched = full_batch(pool, cfg["holdout_cases"], cfg["library"])
        learned = full_batch(
            pool, cfg["holdout_cases"], seed_result["selected"]["parameters"]
        )
    result = {
        "family": seed_result["family"],
        "seed": seed_result["seed"],
        "configuration": cfg,
        "parameters": seed_result["selected"]["parameters"],
        "baseline": baselines,
        "stitched": stitched,
        "learned": learned,
        "statistics": paired_statistics(baselines, learned),
        "stitch_statistics": paired_statistics(baselines, stitched),
        "refinement_statistics": paired_statistics(stitched, learned),
    }
    save_json(path, result)
    return result


def gather(pool, jobs, output, phase):
    futures = {pool.submit(function, *args): label for label, function, args in jobs}
    results = {}
    last = 0.0
    while futures:
        for future in list(futures):
            if future.done():
                label = futures.pop(future)
                try:
                    results[label] = future.result()
                    print("JOB_COMPLETE", phase, label, flush=True)
                except Exception as error:
                    save_json(
                        output / "job_failure.json",
                        {"phase": phase, "job": label, "error": repr(error)},
                    )
                    raise RuntimeError(
                        f"{phase} job failed: {label}; checkpoints retained"
                    ) from error
        if monotonic() - last >= 20:
            progress = {
                "phase": phase,
                "completed_jobs": len(results),
                "remaining_jobs": len(futures),
                "active": [],
            }
            for label in futures.values():
                folder = (
                    output
                    / ("profiles" if phase == "profile_train" else "combinations")
                    / label
                )
                cp = folder / "checkpoint.json"
                if (
                    phase.endswith("train")
                    and cp.exists()
                    and not (folder / "seed_summary.json").exists()
                ):
                    value = json.loads(cp.read_text(encoding="utf-8"))
                    progress["active"].append(
                        {
                            "job": label,
                            "candidate": value["completed"],
                            "best_cost": value["best"]["evaluation"]["cost"],
                        }
                    )
            save_json(output / "progress.json", progress)
            print("SUITE_PROGRESS", json.dumps(progress), flush=True)
            last = monotonic()
        sleep(0.5)
    return results


def runtime_bank(profile_winners, combo_winners, combo_holdouts):
    entries = {}
    for family, winner in combo_winners.items():
        result = combo_holdouts[f"{family}/seed_{winner['seed']}"]
        accepted = passed(result["statistics"])
        values = winner["selected"]["parameters"] if accepted else {}
        for profile in active_profiles(family):
            key = f"{profile.arm}:{profile.skill}"
            scope = profile.scope()
            scope["workflows"] = [family]
            if family != "assembly":
                scope["goal_count"] = (
                    2 if family in ("two_valves", "repeat", "rotate_inspect") else 1
                )
                scope["goal_sets"] = (
                    [["valve_1", "valve_2"]]
                    if family == "two_valves"
                    else [["valve_1", "valve_1"]]
                    if family == "repeat"
                    else [["valve_1", "gauge_1"]]
                    if family == "rotate_inspect"
                    else [["gauge_1"]]
                    if family == "inspect"
                    else [["valve_1"]]
                    if family == "shade"
                    else [["valve_1"], ["valve_2"]]
                )
                scope["goal_angle_min"] = pi / 2 - 0.02
                scope["goal_angle_max"] = pi + 0.02
                scope["goal_angle_sign"] = 1
            entries.setdefault(key, {"profiles": []})["profiles"].append(
                {
                    "id": f"{family}:{profile.id}",
                    "baseline": make_bank({profile.id: [1, 1, 1]}).entries[key][
                        "profiles"
                    ][0]["baseline"],
                    "ratios": values.get(profile.id, [1.0, 1.0, 1.0]),
                    "scope": scope,
                    "validation": {
                        "accepted_for_workflow": accepted,
                        "statistics": result["statistics"],
                    },
                }
            )
    return entries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--combo-iterations", type=int, default=120)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 19, 43])
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--case-workers", type=int, default=3)
    parser.add_argument("--profiles", nargs="+")
    parser.add_argument("--families", nargs="+", choices=FAMILIES)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--case-set",
        help="Unique condition-set label; reuse only for exact resume/reproduction",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="4-case checks; cannot pass runtime acceptance",
    )
    args = parser.parse_args(argv)
    if min(
        args.iterations, args.combo_iterations, args.workers, args.case_workers
    ) < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Positive budgets and unique seeds required")
    profiles = [p for p in PROFILES if not args.profiles or p.id in args.profiles]
    if not profiles or args.profiles and len(profiles) != len(args.profiles):
        parser.error("Unknown or duplicate skill profile")
    families = list(args.families or FAMILIES)
    if len(families) != len(set(families)):
        parser.error("Duplicate families")
    output = args.output or OUTPUTS_DIR / "carla" / datetime.now().strftime(
        "skills_suite_%Y%m%d_%H%M%S"
    )
    case_set = args.case_set or output.name
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()) and not args.resume:
        raise ValueError("Nonempty suite directory; use --resume or another path")
    manifest = json_value(
        {
            "protocol": "skills_suite_v1",
            "profiles": [asdict(p) for p in profiles],
            "families": families,
            "iterations": args.iterations,
            "combo_iterations": args.combo_iterations,
            "seeds": args.seeds,
            "smoke": args.smoke,
            "case_set": case_set,
            "provenance": fingerprints(),
            "skill_configs": {
                p.id: profile_config(p, args.seeds[0], args.smoke, case_set)
                for p in profiles
            },
            "composition_cases": {
                f: {
                    split: [
                        asdict(c)
                        for c in make_cases(
                            f"{case_set}_{f}_combo_{split}",
                            f,
                            5
                            if split == "train"
                            else 4
                            if args.smoke
                            else 8
                            if split == "selection"
                            else 16,
                        )
                    ]
                    for split in ("train", "selection", "holdout")
                }
                for f in families
            },
            "coverage_limits": {
                "arms": "All skills actually present; assembly is single-arm L, mobile uses L and R",
                "navigation": "fixed position-servo navigation; exercised in combinations, no arm STA parameters",
                "assembly_rotation": "wrist trajectory only; no threaded peg mechanism; physical valve rotation verified in mobile scenes",
                "inspection": "verified observation-position arrival; no pressure OCR",
                "invalid_orders": "pin B cannot be placed on occupied pin A pickup zone; only three valid assembly orders",
            },
        }
    )
    manifest_path = output / "protocol.json"
    if (
        args.resume
        and manifest_path.exists()
        and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest
    ):
        raise ValueError("Suite protocol/source mismatch")
    save_json(manifest_path, manifest)
    start = monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [
            (
                f"{p.id}/seed_{seed}",
                train_profile,
                (
                    asdict(p),
                    seed,
                    args.iterations,
                    args.case_workers,
                    str(output / "profiles" / p.id / f"seed_{seed}"),
                    args.resume,
                    args.smoke,
                    case_set,
                ),
            )
            for p in profiles
            for seed in args.seeds
        ]
        seeds = gather(pool, jobs, output, "profile_train")
        winners = {
            p.id: min(
                (seeds[f"{p.id}/seed_{s}"] for s in args.seeds),
                key=lambda r: r["mean_selection_cost"],
            )
            for p in profiles
        }
        save_json(output / "frozen_skill_selection.json", winners)
        jobs = [
            (
                label,
                holdout_profile,
                (result, args.case_workers, str(output / "profiles" / label)),
            )
            for label, result in seeds.items()
        ]
        heldouts = gather(pool, jobs, output, "profile_holdout")
        library = {}
        profile_evidence = {}
        for pid, winner in winners.items():
            result = heldouts[f"{pid}/seed_{winner['seed']}"]
            accepted = (
                passed(result["statistics"])
                and result["complete_statistics"]["mean_relative_gain_pct"] >= -2.0
            )
            profile_evidence[pid] = {
                "accepted": accepted,
                "winner_seed": winner["seed"],
                "ratios": winner["selected"]["ratios"],
                "statistics": result["statistics"],
                "complete_statistics": result["complete_statistics"],
            }
            if accepted:
                library[pid] = winner["selected"]["ratios"]
        save_json(
            output / "skill_library.json",
            {"parameters": library, "evidence": profile_evidence},
        )
        jobs = [
            (
                f"{family}/seed_{seed}",
                train_combo,
                (
                    family,
                    seed,
                    args.combo_iterations,
                    args.case_workers,
                    library,
                    str(output / "combinations" / family / f"seed_{seed}"),
                    args.resume,
                    args.smoke,
                    case_set,
                ),
            )
            for family in families
            for seed in args.seeds
        ]
        combo_seeds = gather(pool, jobs, output, "combo_train")
        combo_winners = {
            f: min(
                (combo_seeds[f"{f}/seed_{s}"] for s in args.seeds),
                key=lambda r: r["mean_selection_cost"],
            )
            for f in families
        }
        save_json(output / "frozen_composition_selection.json", combo_winners)
        jobs = [
            (
                label,
                holdout_combo,
                (result, args.case_workers, str(output / "combinations" / label)),
            )
            for label, result in combo_seeds.items()
        ]
        combo_heldouts = gather(pool, jobs, output, "combo_holdout")
    entries = runtime_bank(winners, combo_winners, combo_heldouts)
    composition_evidence = {
        f: {
            "winner_seed": w["seed"],
            "accepted": passed(combo_heldouts[f"{f}/seed_{w['seed']}"]["statistics"]),
            "statistics": combo_heldouts[f"{f}/seed_{w['seed']}"]["statistics"],
            "stitch_statistics": combo_heldouts[f"{f}/seed_{w['seed']}"][
                "stitch_statistics"
            ],
            "refinement_statistics": combo_heldouts[f"{f}/seed_{w['seed']}"][
                "refinement_statistics"
            ],
        }
        for f, w in combo_winners.items()
    }
    summary = {
        "protocol": manifest["protocol"],
        "profile_evidence": profile_evidence,
        "composition_evidence": composition_evidence,
        "profile_seeds": seeds,
        "profile_holdout": heldouts,
        "combo_seeds": combo_seeds,
        "combo_holdout": combo_heldouts,
        "elapsed_current_invocation_seconds": monotonic() - start,
        "coverage_limits": manifest["coverage_limits"],
    }
    save_json(output / "suite_summary.json", summary)
    accepted = (
        any(e["accepted"] for e in composition_evidence.values()) and not args.smoke
    )
    save_json(
        output / "best_parameters.json",
        {
            "schema_version": 2,
            "control_version": CONTROL_VERSION,
            "accepted_for_runtime": accepted,
            "entries": entries,
            "validation": {
                "protocol": "skills_suite_v1",
                "case_set": case_set,
                "compositions": composition_evidence,
                "profiles": profile_evidence,
            },
        },
    )
    save_json(
        output / "progress.json",
        {
            "phase": "complete",
            "accepted_for_runtime": accepted,
            "profiles": len(profiles),
            "combinations": len(families),
        },
    )
    print("SUITE_FINISHED", output.resolve(), "ACCEPTED", accepted, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
