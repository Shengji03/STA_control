"""Measure execution effects separately from the optimizer's proxy objective."""

import mujoco
import numpy as np


def ray_intersects_panel(light, target, center, rotation, half_size):
    normal = rotation[:, 2]
    direction = np.asarray(target) - light
    denominator = float(np.dot(normal, direction))
    if abs(denominator) < 1e-9:
        return False
    fraction = float(np.dot(normal, center - light) / denominator)
    if not 0 < fraction < 1:
        return False
    local = rotation.T @ (light + fraction * direction - center)
    return bool(np.all(np.abs(local[:2]) <= half_size[:2]))


class EffectMonitor:
    def __init__(self, model, data, registry, goals, config):
        self.model, self.data, self.registry, self.config = model, data, registry, config
        self.goals = {g['id']: g for g in goals}
        self.records = {}
        self._tick = 0
        self._panel = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'shade_panel')
        self._light = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, 'glare_light')
        self._board = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'shade_board')

    def _joint_position(self, goal):
        spec = self.registry.objects.get(goal['object'])
        if not spec or not spec.joint:
            return None
        joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, spec.joint)
        return float(self.data.qpos[self.model.jnt_qposadr[joint]]) if joint >= 0 else None

    def _point(self, goal, key):
        spec = self.registry.objects.get(goal['object'])
        ref = spec.keypoints.get(key) if spec else None
        if ref is None or ref.ref_type != 'site':
            return None
        site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, ref.name)
        return self.data.site_xpos[site].copy() if site >= 0 else None

    def start_phase(self, goal_ids, auxiliary_deployed=False):
        for goal_id in goal_ids:
            goal = self.goals[goal_id]
            self.records[goal_id] = {'goal_id': goal_id, 'object': goal['object'], 'operation': goal['operation'],
                                     'initial_angle': self._joint_position(goal), 'shade_samples': 0,
                                     'shade_hits': 0, 'best_inspect_distance_m': None, 'verified': False}
            self.records[goal_id]['auxiliary_deployed'] = auxiliary_deployed
            if auxiliary_deployed and self._board >= 0:
                self.records[goal_id]['board_origin'] = self.data.xpos[self._board].tolist()

    def sample(self, arms):
        self._tick += 1
        if self._tick % 10:
            return
        for arm, state in arms.items():
            skill = state.executor.current_skill if state.executor else None
            goal_id = getattr(skill, 'goal_id', None)
            if goal_id not in self.records:
                continue
            goal, record = self.goals[goal_id], self.records[goal_id]
            if goal['operation'] == 'inspect':
                point = self._point(goal, 'inspect_point')
                # The historical "tcp" sites are at the flange; pinch sites
                # represent the actual gripper point used by target_pos.
                site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'pinch' if arm == 'L' else 'pinch_R')
                if point is not None and site >= 0:
                    distance = float(np.linalg.norm(point - self.data.site_xpos[site]))
                    previous = record['best_inspect_distance_m']
                    record['best_inspect_distance_m'] = distance if previous is None else min(previous, distance)
            elif hasattr(skill, 'angle') and goal.get('shade') != 'none':
                record['shade_samples'] += 1
                point = self._point(goal, 'handle_center')
                if self._panel >= 0 and self._light >= 0 and point is not None:
                    record['shade_hits'] += ray_intersects_panel(
                        self.data.light_xpos[self._light], point,
                        self.data.geom_xpos[self._panel], self.data.geom_xmat[self._panel].reshape(3, 3),
                        self.model.geom_size[self._panel],
                    )

    def finish_phase(self, goal_ids):
        for goal_id in goal_ids:
            goal, record = self.goals[goal_id], self.records[goal_id]
            if goal['operation'] == 'rotate':
                final = self._joint_position(goal)
                record['final_angle'] = final
                if final is None or record['initial_angle'] is None:
                    record['reason'] = '没有可用阀门角度传感器'
                    continue
                delta = final - record['initial_angle']
                # This is an incremental rotation request. A zero movement must
                # not satisfy a requested full turn just because angles wrap.
                error = abs(delta - goal['angle'])
                record.update(measured_delta=delta, expected_delta=goal['angle'], angle_error_rad=error)
                record['verified'] = error <= self.config.rotation_tolerance
                record['shade_fraction'] = (record['shade_hits'] / record['shade_samples']
                                             if record['shade_samples'] else None)
                if goal.get('shade') == 'required':
                    record['verified'] &= (record['shade_fraction'] is not None and record['shade_fraction'] >= 0.9)
                if record.get('auxiliary_deployed') and self._board >= 0:
                    error = float(np.linalg.norm(self.data.xpos[self._board] - record['board_origin']))
                    record['board_return_error_m'] = error
                    record['verified'] &= error <= 0.08
                if not record['verified']:
                    record['reason'] = '实际阀门角度、必需遮光或遮光板放回效果未达到要求'
            else:
                distance = record['best_inspect_distance_m']
                record['verified'] = distance is not None and distance <= 0.08
                record['effect'] = '观测位到达；没有压力读数识别'
        return all(self.records[g]['verified'] for g in goal_ids)

    def report(self):
        return {'verified': bool(self.goals) and len(self.records) == len(self.goals)
                and all(r['verified'] for r in self.records.values()),
                'goals': list(self.records.values()),
                'rotation_tolerance_rad': self.config.rotation_tolerance,
                'required_shade_fraction': 0.9,
                'shade_measurement': 'actual simulated board/light/target ray intersection during rotation'}
