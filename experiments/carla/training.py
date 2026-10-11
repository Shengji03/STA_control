"""Resumable CARLA search, atomic records and execution-source fingerprints."""

import hashlib
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from time import sleep

import numpy as np

from src.config.paths import DEFAULT_SCENE_PATH
from src.controller.carla_tuner import CarlaTuner


def aggregate(cases):
    costs = [case["cost"] for case in cases]
    return {
        "valid": all(case["valid"] for case in cases),
        "cost": float(np.mean(costs) + 0.2 * max(costs)),
        "cases": cases,
    }


def atomic_replace(temporary, path):
    # Windows readers can briefly hold the destination without delete sharing.
    for attempt in range(8):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            sleep(min(0.025 * 2**attempt, 0.4))


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    atomic_replace(temporary, path)


def reconcile_journal(output, completed):
    """Keep checkpoint-committed trials; preserve every discarded row in an archive."""
    path = output / "trials.jsonl"
    if not path.exists():
        if completed:
            raise ValueError("Completed checkpoint is missing its candidate journal")
        return None
    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
    ]
    committed, discarded = {}, []
    for row in records:
        iteration = row["iteration"]
        if iteration > completed:
            discarded.append(row)
        elif iteration in committed:
            if committed[iteration] != row:
                raise ValueError(
                    f"Conflicting journal records for candidate {iteration}"
                )
            discarded.append(row)
        else:
            committed[iteration] = row
    if sorted(committed) != list(range(1, completed + 1)):
        raise ValueError("Candidate journal does not match the completed checkpoint")
    if not discarded:
        return None
    archive = output / f"journal_recovery_{datetime.now():%Y%m%d_%H%M%S_%f}.json"
    save_json(archive, {"checkpoint_completed": completed, "preserved_rows": discarded})
    temporary = path.with_suffix(".jsonl.tmp")
    temporary.write_text(
        "".join(
            json.dumps(committed[i], ensure_ascii=False, allow_nan=False) + "\n"
            for i in range(1, completed + 1)
        ),
        encoding="utf-8",
    )
    atomic_replace(temporary, path)
    return archive


def provenance():
    root = DEFAULT_SCENE_PATH.parents[2]
    paths = [
        DEFAULT_SCENE_PATH,
        *[
            root / name
            for name in (
                "assets/universal_robots_ur5e/ur5e_classes.xml",
                "assets/universal_robots_ur5e/ur5e_mobile.xml",
                "controller/sta_controller/sta_controller.py",
                "controller/arm_torque.py",
                "controller/carla_tuner.py",
                "controller/control_metrics.py",
                "controller/control_parameters.py",
                "controller/admittance_controller/admittance_controller.py",
                "skills/base_skill.py",
                "skills/skills.py",
                "skills/skill_executor.py",
                "pipeline/task_runner.py",
                "pipeline/plan_executor.py",
                "task_planning/candidates.py",
                "task_planning/effects.py",
            )
        ],
        Path(__file__),
    ]
    return {
        str(path.relative_to(root.parent)): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in paths
    }


def search_parameters(
    states,
    tuner,
    iterations,
    output,
    configuration,
    checkpoint=None,
    initial_candidate=None,
    label="",
    case_evaluator=None,
    require_baseline=True,
):
    """Shared resumable candidate loop for skill and composition training."""
    checkpoint_path = output / "checkpoint.json"
    if checkpoint:
        if checkpoint["configuration"] != configuration:
            raise ValueError("Resume configuration/source fingerprints do not match")
        tuner = CarlaTuner.from_state(checkpoint["tuner"])
        best, baseline, start = (
            checkpoint["best"],
            checkpoint["baseline"],
            checkpoint["completed"],
        )
        leaders = checkpoint["leaders"]
        archive = reconcile_journal(output, start)
        if archive:
            print(f"{label} JOURNAL_RECOVERY {archive}", flush=True)
    else:
        baseline = aggregate(
            case_evaluator([1, 1, 1])
            if case_evaluator
            else [s.evaluate([1, 1, 1]) for s in states]
        )
        if not baseline["valid"] and require_baseline:
            save_json(output / "baseline_failure.json", baseline)
            raise RuntimeError(
                "Fixed baseline failed; parameter learning is not started"
            )
        best = {
            "ratios": [1.0, 1.0, 1.0],
            "evaluation": baseline,
            "iteration": 0,
            "origin": "baseline",
        }
        leaders, start = [best], 0
        if initial_candidate is not None:
            initial = aggregate(
                case_evaluator(initial_candidate)
                if case_evaluator
                else [s.evaluate(initial_candidate) for s in states]
            )
            if initial["valid"]:
                prior = {
                    "ratios": list(initial_candidate),
                    "evaluation": initial,
                    "iteration": -1,
                    "origin": "warm_start",
                }
                leaders.append(prior)
                if initial["cost"] < best["evaluation"]["cost"]:
                    best = prior
        print(
            f"{label} BASELINE cost={baseline['cost']:.6f}; {len(states)} real predecessor states",
            flush=True,
        )

    def checkpoint_now(completed):
        value = {
            "configuration": configuration,
            "tuner": tuner.state(),
            "best": best,
            "baseline": baseline,
            "completed": completed,
            "leaders": leaders,
        }
        save_json(checkpoint_path, value)
        return value

    checkpoint_now(start)
    for iteration in range(start + 1, iterations + 1):
        ratios = tuner.propose()
        if case_evaluator is not None:
            cases = case_evaluator(ratios)
        else:
            cases = []
            for state in states:
                try:
                    cases.append(state.evaluate(ratios))
                except (ValueError, RuntimeError, FloatingPointError) as error:
                    cases.append(
                        {
                            "case": asdict(state.case),
                            "valid": False,
                            "state": "EVALUATION_ERROR",
                            "cost": 1e6,
                            "error": str(error),
                        }
                    )
        evaluation = aggregate(cases)
        reward = tuner.observe(ratios, evaluation["cost"], evaluation["valid"])
        candidate = {
            "ratios": ratios.tolist(),
            "evaluation": evaluation,
            "iteration": iteration,
            "origin": "carla_pdf",
        }
        if evaluation["valid"]:
            if evaluation["cost"] < best["evaluation"]["cost"]:
                best = candidate
            leaders.append(candidate)
            leaders = sorted(leaders, key=lambda item: item["evaluation"]["cost"])[:5]
        row = {
            "iteration": iteration,
            "ratios": ratios.tolist(),
            "reward": reward,
            "evaluation": evaluation,
            "best_cost": best["evaluation"]["cost"],
        }
        with (output / "trials.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        checkpoint_now(iteration)
        print(
            f"{label} ROUND {iteration}/{iterations}: valid={evaluation['valid']} "
            f"cost={evaluation['cost']:.6f} best={best['evaluation']['cost']:.6f}",
            flush=True,
        )
    return checkpoint_now(max(start, iterations))
