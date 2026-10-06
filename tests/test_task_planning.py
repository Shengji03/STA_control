from itertools import combinations
from math import pi

import numpy as np
import pytest

from src.task_planning.models import (AuxiliaryCandidate, Goal, MainCandidate, PlanningConfig,
                                      PlanningRequest, Stage)
from src.task_planning.requests import semantic_request
from src.task_planning.solver import coverage_value, solve_policies
from src.task_planning.effects import ray_intersects_panel


def main(goal, arm, score, phase=0):
    return MainCandidate(f'{goal}:{arm}', goal, phase, arm, (), score)


def helper(name, covers, phase=0):
    return AuxiliaryCandidate(name, phase, 'R', (), (), frozenset(covers), (0, 0, .8))


def test_coupled_selection_changes_main_arm_for_optional_helper_benefit():
    request = PlanningRequest((Goal('g', 'valve_1', 'rotate', pi, 'optional'),), (Stage(0, None, ('g',)),))
    candidates = (main('g', 'L', .4), main('g', 'R', .7))
    auxiliaries = (helper('shade', ['g']),)
    best = solve_policies(request, candidates, auxiliaries, PlanningConfig())[0]
    assert best.assignments[0].arm == 'L'
    assert best.score == pytest.approx(1.4)
    unassisted = solve_policies(request, candidates, auxiliaries, PlanningConfig(auxiliary_weight=0))[0]
    assert unassisted.assignments[0].arm == 'R'
    assert unassisted.auxiliaries == ()


@pytest.mark.parametrize('exact_limit', [0, 12])
def test_required_helpers_are_reserved_even_when_their_score_weight_is_zero(exact_limit):
    request = PlanningRequest((Goal('g', 'valve_1', 'rotate', pi, 'required'),), (Stage(0, None, ('g',)),))
    policies = solve_policies(request, (main('g', 'L', .1), main('g', 'R', 100)),
                              (helper('shade', ['g']),),
                              PlanningConfig(auxiliary_weight=0, exact_auxiliary_limit=exact_limit))
    assert all(p.assignments[0].arm == 'L' and len(p.auxiliaries) == 1 for p in policies)
    assert policies[0].auxiliary_method == ('greedy' if exact_limit == 0 else 'exact')


def test_zero_budget_is_preserved_and_missing_required_helper_is_an_error():
    goal = Goal('g', 'valve_1', 'rotate', pi, 'required')
    request = PlanningRequest((goal,), (Stage(0, None, ('g',)),))
    config = PlanningConfig(max_auxiliaries=0)
    assert config.max_auxiliaries == 0
    with pytest.raises(ValueError, match='全部目标'):
        solve_policies(request, (main('g', 'L', 1),), (helper('shade', ['g']),), config)


def test_every_goal_is_assigned_even_when_its_main_score_is_zero():
    request = PlanningRequest((Goal('a', 'valve_1', 'rotate'), Goal('b', 'valve_2', 'rotate')),
                              (Stage(0, None, ('a',)), Stage(1, None, ('b',))))
    best = solve_policies(request, (main('a', 'R', 1), main('b', 'L', 0, 1)), (), PlanningConfig())[0]
    assert {a.goal_id for a in best.assignments} == {'a', 'b'}


def test_parallel_goals_use_distinct_arms_and_cannot_share_an_auxiliary_arm():
    request = PlanningRequest((Goal('a', 'valve_1', 'rotate'), Goal('b', 'valve_2', 'rotate')),
                              (Stage(0, None, ('a', 'b'), True),))
    candidates = tuple(main(g, arm, 1) for g in ('a', 'b') for arm in ('L', 'R'))
    policies = solve_policies(request, candidates, (), PlanningConfig())
    assert all({a.arm for a in p.assignments} == {'L', 'R'} for p in policies)
    required = PlanningRequest((Goal('a', 'valve_1', 'rotate', pi, 'required'), request.goals[1]), request.stages)
    with pytest.raises(ValueError, match='资源约束'):
        solve_policies(required, candidates, (helper('shade', ['a']),), PlanningConfig())


def test_coverage_has_diminishing_returns_and_duplicate_helpers_add_no_reward():
    ground = (helper('a', ['x']), helper('b', ['x', 'y']), helper('c', ['z']), helper('d', ['x']))
    weights = {'x': 2, 'y': 1, 'z': 3}
    subsets = [set(items) for size in range(5) for items in combinations(range(4), size)]
    value = lambda indices: coverage_value([ground[i] for i in indices], weights)
    for small in subsets:
        for large in subsets:
            if small <= large:
                for element in set(range(4)) - large:
                    assert value(small | {element}) - value(small) >= value(large | {element}) - value(large)
    assert coverage_value((ground[0], ground[3]), weights) == 2
    with pytest.raises(ValueError, match='nonnegative'):
        coverage_value((), {'x': -1})


def test_search_limit_fails_without_dropping_goals():
    request = PlanningRequest((Goal('a', 'valve_1', 'rotate'), Goal('b', 'valve_2', 'rotate')),
                              (Stage(0, None, ('a',)), Stage(1, None, ('b',))))
    candidates = tuple(main(g, arm, 1, phase) for phase, g in enumerate(('a', 'b')) for arm in ('L', 'R'))
    with pytest.raises(ValueError, match='完整分工组合'):
        solve_policies(request, candidates, (), PlanningConfig(max_assignments=2))


OBJECTS = {'valve_1': {'kind': 'valve'}, 'valve_2': {'kind': 'valve'},
           'gauge_1': {'kind': 'gauge'}, 'gauge_2': {'kind': 'gauge'}}


def request_data():
    return {'goals': [{'id': 'a', 'object': 'valve_1', 'operation': 'rotate', 'angle': pi},
                      {'id': 'b', 'object': 'valve_2', 'operation': 'rotate', 'angle': pi}],
            'stages': [{'nav': {'target': [1, 0]}, 'goals': ['a', 'b']}]}


def test_sequential_expansion_keeps_navigation_and_goal_order():
    result = semantic_request(request_data(), {'planner_state': {'objects': OBJECTS}}, '旋拧1、2号阀门180度')
    assert [s.goals for s in result.stages] == [('a',), ('b',)]
    assert result.stages[0].nav['target'] == [1, 0]
    assert result.stages[1].nav is None


def test_duplicate_goal_references_and_missing_requested_objects_are_rejected():
    data = request_data()
    data['stages'][0]['goals'] = ['a', 'a']
    with pytest.raises(ValueError, match='且只能'):
        semantic_request(data, {'planner_state': {'objects': OBJECTS}})
    data = request_data()
    data['goals'].pop()
    data['stages'][0]['goals'].pop()
    with pytest.raises(ValueError, match='遗漏'):
        semantic_request(data, {'planner_state': {'objects': OBJECTS}}, '旋拧1、2号阀门180度')
    with pytest.raises(ValueError, match='required'):
        semantic_request(request_data(), {'planner_state': {'objects': OBJECTS}}, '遮光并旋拧1号阀门180度')


def test_pressure_inspection_requires_all_gauges_and_correct_kind():
    data = {'goals': [{'id': 'g', 'object': 'gauge_1', 'operation': 'inspect'}],
            'stages': [{'nav': {'target': [0, 0]}, 'goals': ['g']}]}
    with pytest.raises(ValueError, match='gauge_2'):
        semantic_request(data, {'planner_state': {'objects': OBJECTS}}, '巡检压力表')
    data['goals'][0]['operation'] = 'rotate'
    with pytest.raises(ValueError, match='不支持'):
        semantic_request(data, {'planner_state': {'objects': OBJECTS}})


def test_actual_panel_intersection_uses_pose_and_rejects_behind_target_or_off_ray():
    light, target = np.array([0., 0., 2.]), np.array([0., 0., 0.])
    assert ray_intersects_panel(light, target, np.array([0., 0., 1.]), np.eye(3), np.array([.15, .12, .005]))
    assert not ray_intersects_panel(light, target, np.array([.4, 0., 1.]), np.eye(3), np.array([.15, .12, .005]))
    assert not ray_intersects_panel(light, target, np.array([0., 0., -1.]), np.eye(3), np.array([.15, .12, .005]))
