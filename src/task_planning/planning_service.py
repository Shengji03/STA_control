"""Shared post-LLM planning service for the CLI and web simulation."""

from copy import deepcopy
from dataclasses import asdict
from time import perf_counter

import numpy as np

from ..llm_planner.feedback_manager import FeedbackManager
from ..pipeline.plan_executor import PlanExecutor
from .candidates import generate_candidates
from .models import PlanningConfig
from .requests import make_request
from .solver import solve_policies


class PlanningService:
    def __init__(self, model, data, arm_offsets, arm_yaws, config=None):
        self.config = config or PlanningConfig()
        self.executor = PlanExecutor(arm_offsets, arm_yaws, verbose=False)
        self.validator = FeedbackManager(model, data, arm_offsets, arm_yaws,
                                         collision_samples=self.config.collision_samples)

    def prepare(self, result, snapshot, instruction, cart_pose):
        start = perf_counter()
        try:
            request = make_request(result, snapshot, instruction)
        except (TypeError, KeyError, OverflowError) as exc:
            raise ValueError(f"任务规划结构无效: {exc}") from exc
        if request is None:
            return self._fixed_legacy(result, snapshot, instruction, cart_pose)
        main, auxiliaries, rejected = generate_candidates(request, snapshot, cart_pose,
                                                          self.executor, self.config)
        try:
            policies = solve_policies(request, main, auxiliaries, self.config)
        except ValueError as exc:
            details = '; '.join(f"{r['candidate']}: {r['reason']}" for r in rejected[:6])
            raise ValueError(f'{exc}' + (f'。候选筛除原因: {details}' if details else '')) from exc
        failures = []
        prepared = selected = None
        for policy in policies:
            candidate = self._compile(request, policy, result.get('reasoning', ''))
            ok, feedback = self.validator.give_feedback(candidate, snapshot, instruction)
            if ok:
                prepared, selected = candidate, policy
                break
            failures.append({'assignments': [a.id for a in policy.assignments],
                             'auxiliaries': [b.id for b in policy.auxiliaries], 'reason': feedback})
        if selected is None:
            detail = failures[0]['reason'] if failures else '没有联合方案'
            raise ValueError(f"联合方案均未通过执行校验，需重新规划导航或目标: {detail}")

        # The preference baseline uses the SAME feasible templates/objective.
        # It is not labelled as an exact score for the original LLM trajectories.
        baseline_score = None
        for baseline in (p for p in policies if p.preferred_count == len(request.goals)):
            if baseline == selected:
                baseline_score = baseline.score
                break
            else:
                ok, _ = self.validator.give_feedback(self._compile(request, baseline, ''), snapshot, instruction)
                if ok:
                    baseline_score = baseline.score
                    break
        report = {
            'status': 'optimized', 'source': request.source,
            'algorithm': 'complete_assignment_enumeration+weighted_coverage',
            'auxiliary_method': selected.auxiliary_method,
            'objective': selected.score, 'main_score': selected.main_score,
            'auxiliary_score': selected.auxiliary_score,
            'auxiliary_weight': self.config.auxiliary_weight,
            'config': asdict(self.config),
            'score_definition': 'nonnegative reach/posture proxy + weighted covered goals',
            'main_score_definition': 'reach_weight * min(1-distance/reach) + posture_weight/(1+sum(mean(abs(dq)))/pi)',
            'coverage_proxy': {'type': 'horizontal_ray_disk', 'radius_m': 0.10, 'weight_per_goal': 1.0},
            'preferred_arm_baseline': baseline_score,
            'gain_over_preference': None if baseline_score is None else selected.score - baseline_score,
            'assignments': [{'goal_id': a.goal_id, 'arm': a.arm, 'phase_id': a.phase_id, 'score': a.score}
                            for a in selected.assignments],
            'auxiliaries': [{'id': b.id, 'phase_id': b.phase_id, 'arm': b.arm,
                             'covers': sorted(b.covers), 'position': list(b.position)} for b in selected.auxiliaries],
            'goal_count': len(request.goals), 'covered_goal_count': len(selected.assignments),
            'required_auxiliary_count': sum(g.shade == 'required' for g in request.goals),
            'main_candidate_count': len(main), 'auxiliary_candidate_count': len(auxiliaries),
            'joint_policy_count': len(policies), 'rejected_candidates': rejected,
            'rejected_policy_count': len(failures), 'rejected_policies': failures[:10],
            'validation': {'task_complete': True, 'ik': True, 'collision_samples_per_segment': self.config.collision_samples,
                           'continuous_collision_proof': False},
            'theoretical_ratio': None,
            'elapsed_ms': round(1000 * (perf_counter() - start), 3),
        }
        prepared['optimization'] = report
        print(f"[TaskOptimizer] {len(request.goals)} goals, objective={selected.score:.4f}, "
              f"aux={selected.auxiliary_method}, rejected={len(failures)}")
        return prepared

    @staticmethod
    def _compile(request, policy, reasoning):
        phases, flat = [], []
        for stage in request.stages:
            assignments = [a for a in policy.assignments if a.phase_id == stage.id]
            helper = next((b for b in policy.auxiliaries if b.phase_id == stage.id), None)
            steps = list(helper.setup) if helper else []
            steps.extend(s for a in assignments for s in a.steps)
            arms = {a.arm for a in assignments}
            mode = ('assist_R_then_L' if helper else 'parallel' if len(arms) == 2
                    else 'single_L' if 'L' in arms else 'single_R' if 'R' in arms else 'idle')
            targets = {'L': [], 'R': []}
            goal_map = {g.id: g for g in request.goals}
            for a in assignments:
                if goal_map[a.goal_id].operation == 'rotate':
                    targets[a.arm].append(goal_map[a.goal_id].target)
            if helper:
                targets['R'].append('shade_board')
            phase = {'id': stage.id, 'nav': deepcopy(stage.nav), 'mode': mode,
                     'goal_ids': list(stage.goals), 'steps': deepcopy(steps),
                     'cleanup_R': deepcopy(list(helper.cleanup)) if helper else [], 'contact_targets': targets}
            phases.append(phase)
            if stage.nav:
                flat.append({'arm': 'L', 'skill': 'NavSkill', 'params': deepcopy(stage.nav), 'description': '保留原导航阶段'})
            flat.extend(deepcopy(steps))
            flat.extend(deepcopy(phase['cleanup_R']))
        for index, item in enumerate(flat, 1):
            item['step'] = index
        return {'reasoning': reasoning,
                'goals': [{'id': g.id, 'object': g.target, 'operation': g.operation, 'angle': g.angle,
                           'shade': g.shade, 'preferred_arm': g.preferred_arm} for g in request.goals],
                'stages': [{'nav': s.nav, 'goals': list(s.goals), 'parallel': s.parallel} for s in request.stages],
                'plan': flat, 'execution_phases': phases}

    def _fixed_legacy(self, result, snapshot, instruction, cart_pose):
        # Do not guess semantics of unknown manipulation blocks or silently skip
        # required assistance. Such tasks need explicit goals/stages for optimization.
        if ('遮光' in instruction or '压力表' in instruction
                or any(s.get('skill') == 'RotateSkill' for s in result.get('plan', []))
                or ('阀门' in instruction and any(word in instruction for word in ('旋', '关闭', '打开')))):
            raise ValueError('无法确认旧计划的完整任务块，请使用 goals/stages 表达全部目标与必需辅助')
        fixed = deepcopy(result)
        parsed = self.executor.parse(fixed, np.asarray(cart_pose)[:2], cart_pose[2])
        if not parsed['phases']:
            raise ValueError('预置计划不能为空')
        ok, feedback = self.validator.give_feedback(fixed, snapshot, instruction)
        if not ok:
            raise ValueError(f'固定旧计划校验失败: {feedback}')
        fixed['optimization'] = {'status': 'fixed_legacy',
                                 'reason': '旧技能序列无法可靠关联到语义目标，已保留并校验；输出 goals/stages 可参与优化'}
        return fixed
