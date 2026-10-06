"""Offline Scene5 planning comparison; --simulate additionally checks effects."""

import argparse
from dataclasses import asdict, replace
import json
from math import pi
from pathlib import Path

from src.config.paths import DEFAULT_SCENE_PATH, OUTPUTS_DIR
from src.pipeline.task_runner import TaskRunner
from src.task_planning.models import PlanningConfig


def benchmark_cases():
    def goal(goal_id, target, shade='none'):
        return {'id': goal_id, 'object': target, 'operation': 'rotate', 'angle': pi,
                'shade': shade, 'preferred_arm': 'R'}
    return {
        'valve_arm_choice': {'goals': [goal('v1', 'valve_1')],
                            'stages': [{'nav': {'target': [.45, 0]}, 'goals': ['v1']}]},
        'required_shade': {'goals': [goal('v1', 'valve_1', 'required')],
                           'stages': [{'nav': {'target': [.45, 0]}, 'goals': ['v1']}]},
        'two_required_valves': {'goals': [goal('v1', 'valve_1'), goal('v2', 'valve_2')],
                                'stages': [{'nav': {'target': [.45, 0]}, 'goals': ['v1']},
                                           {'nav': {'target': [3.5, 0]}, 'goals': ['v2']}]},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--simulate', action='store_true', help='Run physical execution as well as planning')
    parser.add_argument('--output', type=Path, default=OUTPUTS_DIR / 'submodular_planning.json')
    args = parser.parse_args(argv)
    config = PlanningConfig()
    results = {'model': 'complete assignments + weighted coverage (adapted formulation)',
               'score_is_success_probability': False, 'original_paper_approximation_bound': None,
               'config': asdict(config), 'cases': []}
    runner = TaskRunner(str(DEFAULT_SCENE_PATH))
    try:
        for name, request in benchmark_cases().items():
            runner.reset_scene()
            runner.planning_service.config = config
            exact = runner.prepare_plan(request, instruction='')
            runner.planning_service.config = replace(config, exact_auxiliary_limit=0)
            greedy = runner.prepare_plan(request, instruction='')
            row = {'name': name, 'exact': exact['optimization'], 'greedy': greedy['optimization'],
                   'greedy_to_exact_objective': greedy['optimization']['objective'] / exact['optimization']['objective']}
            if args.simulate:
                runner.planning_service.config = config
                runner.plan_dict = request
                runner.task_instruction = ''
                runner.run(total_time=150, use_viewer=False, show_trajectory=False)
                row['execution_state'] = runner.state
                row['execution_effects'] = runner.execution_report
            results['cases'].append(row)
            print(f"{name}: exact={exact['optimization']['objective']:.4f}, "
                  f"greedy={greedy['optimization']['objective']:.4f}, goals={len(request['goals'])}")
    finally:
        if runner.perception._renderer is not None:
            runner.perception._renderer.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Report: {args.output.resolve()}')
    return int(args.simulate and any(row.get('execution_state') != 'DONE' for row in results['cases']))


if __name__ == '__main__':
    raise SystemExit(main())
