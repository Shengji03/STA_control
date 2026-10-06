"""Audit the supplied reproduction before adapting it to robot task planning.

This is an offline research check. It does not call the LLM or change task execution.
Run with --reproduction-dir pointing to the standalone coupled_submodular project.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.paths import OUTPUTS_DIR


def _load_reproduction(directory: Path):
    directory = directory.resolve()
    if not (directory / "coupled_submodular" / "instances.py").is_file():
        raise ValueError("The reproduction directory must contain coupled_submodular")
    sys.path.insert(0, str(directory))
    return (
        importlib.import_module("coupled_submodular.instances"),
        importlib.import_module("coupled_submodular.objectives"),
        importlib.import_module("coupled_submodular.constraints"),
        importlib.import_module("coupled_submodular.solvers"),
    )


def _make_covariance_case(model):
    count = 10
    dimension = count + 1
    epsilon = 1e-4
    first = model.TaskElement(0, 0, 0)
    second = model.TaskElement(1, 0, 1)
    tasks = frozenset((first, second))
    misleading = [model.DeploymentElement(i, i) for i in range(count)]
    informative = [model.DeploymentElement(i, count + i) for i in range(count)]
    measurements = {}
    for index in range(count):
        first_matrix = np.zeros((1, dimension))
        first_matrix[0, 0] = 1
        second_matrix = np.zeros((1, dimension))
        second_matrix[0, index + 1] = 1
        measurements[index] = first_matrix
        measurements[count + index] = second_matrix
    instance = model.CoupledInstance(
        task_ground=tasks,
        deployment_ground=frozenset(misleading + informative),
        task_rewards={first: 0., second: 0.},
        deployment_rewards={element: 0. for element in misleading + informative},
        initial_covariances={
            first: np.diag([100.] + [epsilon] * count),
            second: np.diag([epsilon] + [float(np.expm1(2.))] * count),
        },
        measurement_matrices=measurements,
        noise_covariances={index: np.eye(1) for index in range(2 * count)},
        state_dim=dimension,
        max_tasks_per_robot=1,
        max_robots_per_time=1,
        max_active_times=count,
    )
    return instance, tasks, misleading, informative


def _contextwise_reference(instance, objectives, constraints, solvers):
    """Fixed-context reference, rather than greedily maximizing a max aggregate.

    Each deployment policy is computed for one assignment only. Its score is then
    a constant weight in g(A) + max_{a in A} weight[a]. Guarantees require valid
    fixed matroid constraints and monotone submodular scalar deployment scores.
    This research reference does not enforce completion of robot instructions.
    """
    policies = {
        task: solvers.greedy_deployment_for_tasks(frozenset((task,)), instance)
        for task in sorted(instance.task_ground)
    }
    weights = {
        task: objectives.deployment_score(task, policies[task], instance)
        for task in policies
    }

    def value(tasks):
        return objectives.task_reward(tasks, instance) + max(
            (weights[task] for task in tasks), default=0.,
        )

    selected = frozenset()
    while True:
        candidates = [
            selected | {task}
            for task in sorted(instance.task_ground - selected)
            if constraints.is_task_allocation_feasible(selected | {task}, instance)
        ]
        if not candidates:
            break
        best = max(candidates, key=value)
        if value(best) <= value(selected):
            break
        selected = best
    if not selected:
        return selected, frozenset(), 0.
    context = max(sorted(selected), key=lambda task: weights[task])
    policy = policies[context]
    return selected, policy, objectives.coupled_objective(selected, policy, instance)


def run_audit(directory: Path) -> dict:
    model, objectives, constraints, solvers = _load_reproduction(directory)
    instance, tasks, misleading, informative = _make_covariance_case(model)
    value = lambda policy: objectives.coupled_objective(tasks, policy, instance)
    small = frozenset((misleading[0],))
    large = small | {informative[1], informative[2]}
    extra = informative[3]
    small_gain = value(small | {extra}) - value(small)
    large_gain = value(large | {extra}) - value(large)
    for policy in (small, large, small | {extra}, large | {extra}):
        assert constraints.is_deployment_feasible(policy, instance)

    original_policy = solvers.greedy_deployment_for_tasks(tasks, instance)
    conditional_options = [
        solvers.greedy_deployment_for_tasks(frozenset((task,)), instance)
        for task in sorted(tasks)
    ]
    contextual_policy = max(conditional_options, key=value)
    # In this diagonal, symmetric example the value only depends on how many
    # misleading sensors are selected. Eleven representatives cover every full
    # policy value, avoiding enumeration of 2^20 deployment subsets.
    reference_values = []
    for count in range(len(misleading) + 1):
        policy = frozenset(misleading[:count] + informative[count:])
        assert constraints.is_deployment_feasible(policy, instance)
        reference_values.append(value(policy))
    optimum = max(reference_values)
    assert np.isclose(optimum, 20.)
    assert np.isclose(value(contextual_policy), optimum)
    original_inner_value = float(value(original_policy))
    contextual_inner_value = float(value(contextual_policy))

    instance.task_rewards = {task: .1 for task in tasks}
    original_outer = solvers.greedy_coupled(instance)
    corrected_tasks, corrected_policy, corrected_value = _contextwise_reference(
        instance, objectives, constraints, solvers,
    )
    assert corrected_tasks == tasks
    assert constraints.is_deployment_feasible(corrected_policy, instance)
    assert corrected_value >= original_outer.value

    first = model.DeploymentElement(0, 0)
    second = model.DeploymentElement(0, 1)
    third = model.DeploymentElement(1, 0)
    instance.deployment_ground = frozenset((first, second, third))
    instance.max_robots_per_time = 2
    instance.max_active_times = 1
    larger = frozenset((first, second))
    smaller = frozenset((third,))
    large_feasible = constraints.is_deployment_feasible(larger, instance)
    small_feasible = constraints.is_deployment_feasible(smaller, instance)
    augmentation = any(
        constraints.is_deployment_feasible(smaller | {element}, instance)
        for element in larger - smaller
    )
    assert large_feasible and small_feasible and not augmentation

    zero_budget = model.make_random_instance(
        seed=1, max_tasks_per_robot=0, max_robots_per_time=0, max_active_times=0,
    )
    report = {
        "reproduction_dir": str(directory.resolve()),
        "max_aggregate_diminishing_returns": {
            "marginal_small_set": float(small_gain),
            "marginal_larger_set": float(large_gain),
            "submodularity_violation": bool(small_gain < large_gain),
        },
        "fixed_assignment_deployment": {
            "original_inner_greedy": original_inner_value,
            "contextwise_reference": contextual_inner_value,
            "verified_small_case_optimum": float(optimum),
            "original_ratio": original_inner_value / optimum,
        },
        "outer_allocation": {
            "original_selected_tasks": len(original_outer.tasks),
            "original_value": float(original_outer.value),
            "contextwise_selected_tasks": len(corrected_tasks),
            "contextwise_value": float(corrected_value),
        },
        "active_time_constraint_exchange": {
            "max_robots_per_time": 2, "max_active_times": 1,
            "larger_set_feasible": large_feasible,
            "smaller_set_feasible": small_feasible,
            "can_augment_smaller_set": augmentation,
        },
        "zero_budget_factory": {
            "requested": 0,
            "actual_max_tasks_per_robot": zero_budget.max_tasks_per_robot,
            "actual_max_robots_per_time": zero_budget.max_robots_per_time,
            "actual_max_active_times": zero_budget.max_active_times,
        },
        "scope": "Deterministic offline audit and research reference; not a robot allocator.",
    }
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reproduction-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUTS_DIR / "submodular_audit.json")
    args = parser.parse_args(argv)
    report = run_audit(args.reproduction_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"Audit report: {args.output}")


if __name__ == "__main__":
    main()
