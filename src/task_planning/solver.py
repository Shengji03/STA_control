"""Enumerate complete assignments; solve weighted coverage for each assignment.

The auxiliary objective, with an assignment fixed, is monotone submodular.
Whole-stage resource exclusions and complete assignment constraints are checked
explicitly. No approximation ratio from the original coupled max model is used.
"""

from itertools import combinations, product
from math import isfinite, prod

from .models import AuxiliaryCandidate, MainCandidate, PlanningConfig, PlanningRequest, Policy


def coverage_value(auxiliaries, weights: dict[str, float]) -> float:
    if any(not isfinite(w) or w < 0 for w in weights.values()):
        raise ValueError("Coverage weights must be nonnegative")
    covered = set().union(*(b.covers for b in auxiliaries))
    return sum(weights[q] for q in sorted(covered) if q in weights)


def _feasible_auxiliaries(selected, assignments, config):
    if config.max_auxiliaries is not None and len(selected) > config.max_auxiliaries:
        return False
    phases = [b.phase_id for b in selected]
    if len(phases) != len(set(phases)):  # One auxiliary deployment per stage.
        return False
    occupied = {(a.phase_id, a.arm) for a in assignments}
    return all((b.phase_id, b.arm) not in occupied for b in selected)


def _auxiliary_sets(available, assignments, required, weights, config):
    """Reserve required coverage before greedy optional extension."""
    available = tuple(b for b in available
                      if _feasible_auxiliaries((b,), assignments, config))
    if len(available) <= config.exact_auxiliary_limit:
        for size in range(len(available) + 1):
            for selected in combinations(available, size):
                if (_feasible_auxiliaries(selected, assignments, config)
                        and required <= set().union(*(b.covers for b in selected))):
                    yield selected, "exact"
        return

    # Mandatory variants are enumerated by stage, not greedily discarded.
    phase_ids = sorted({b.phase_id for b in available if b.covers & required})
    pools = [tuple(b for b in available if b.phase_id == phase
                   and b.covers >= {g for g in required
                                    if any(a.goal_id == g and a.phase_id == phase
                                           for a in assignments)})
             for phase in phase_ids]
    if prod(len(pool) for pool in pools) > config.max_policies:
        raise ValueError("必需辅助组合过多，请拆分规划阶段或缩小候选集合")
    for reserved in product(*pools):
        selected = list(reserved)
        if (not _feasible_auxiliaries(selected, assignments, config)
                or not required <= set().union(*(b.covers for b in selected))):
            continue
        yield tuple(selected), 'greedy'  # Validated fallback without optional helpers.
        while True:
            value = coverage_value(selected, weights)
            choices = [(coverage_value(selected + [b], weights) - value, b)
                       for b in available if b not in selected
                       and _feasible_auxiliaries(selected + [b], assignments, config)]
            if not choices:
                break
            gain, best = max(choices, key=lambda item: (item[0], item[1].id))
            if gain <= 1e-12:
                break
            selected.append(best)
        if tuple(selected) != reserved:
            yield tuple(selected), "greedy"


def solve_policies(request: PlanningRequest, candidates: tuple[MainCandidate, ...],
                   auxiliaries: tuple[AuxiliaryCandidate, ...], config: PlanningConfig):
    pools = [tuple(a for a in candidates if a.goal_id == goal.id) for goal in request.goals]
    missing = [g.id for g, pool in zip(request.goals, pools) if not pool]
    if missing:
        raise ValueError(f"目标没有可执行的主臂候选: {', '.join(missing)}")
    count = prod(len(pool) for pool in pools)
    if count > config.max_assignments:
        raise ValueError(f"完整分工组合 {count} 超过限制 {config.max_assignments}，请拆分任务")
    required = {g.id for g in request.goals if g.shade == "required"}
    weights = {g.id: 1.0 for g in request.goals if g.shade != "none"}
    parallel = {s.id for s in request.stages if s.parallel}
    policies = []
    for assignments in product(*pools):
        resources = [(a.phase_id, a.arm) for a in assignments if a.phase_id in parallel]
        if len(resources) != len(set(resources)):
            continue
        main_score = sum(a.score for a in assignments)
        for selected, method in _auxiliary_sets(auxiliaries, assignments, required, weights, config):
            auxiliary_score = coverage_value(selected, weights)
            policies.append(Policy(assignments, selected, main_score, auxiliary_score,
                                   main_score + config.auxiliary_weight * auxiliary_score, method))
            if len(policies) > config.max_policies:
                raise ValueError("联合方案数量超过限制，请拆分任务；没有执行不完整方案")
    if not policies:
        raise ValueError("没有满足全部目标、必需遮光和机械臂资源约束的联合方案")
    # A tie keeps the LLM preference; a second tie uses fewer deployments.
    return sorted(policies, key=lambda p: (-p.score, -p.preferred_count,
                                          len(p.auxiliaries),
                                          tuple(a.id for a in p.assignments),
                                          tuple(b.id for b in p.auxiliaries)))
