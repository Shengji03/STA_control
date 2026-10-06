"""Expand complete skill templates and filter candidates using real scene IK."""

from copy import deepcopy
from math import pi

import numpy as np
from spatialmath import SO3

from ..config.robot import INITIAL_JOINTS
from .models import AuxiliaryCandidate, MainCandidate
from .requests import object_point, scene_objects


def step(arm, skill, params, goal_id=None, role="main"):
    return {"arm": arm, "skill": skill, "params": params,
            "goal_id": goal_id, "role": role, "description": f"{role}: {skill}"}


def _move(arm, position, goal_id=None, role="main", duration=3.0, **params):
    return step(arm, "MoveSkill", {"target_pos": np.asarray(position).tolist(),
                                  "duration": duration, **params}, goal_id, role)


def main_steps(goal, arm, objects):
    if goal.steps:
        steps = deepcopy(goal.steps)
        for item in steps:
            item.update(arm=arm, goal_id=goal.id, role="main")
        return steps
    if goal.operation == "inspect":
        point = object_point(objects, goal.target, "inspect_point")
        return (_move(arm, point, goal.id),
                _move(arm, point + [0, 0, min(0.15, 0.88 - point[2])], goal.id))
    point = object_point(objects, goal.target, "handle_center")
    return (_move(arm, point + [0, 0, 0.12], goal.id),
            _move(arm, point, goal.id, duration=2.5),
            step(arm, "GraspSkill", {"action": "close"}, goal.id),
            # The down-facing wrist axis opposes the valve's world +z axis.
            step(arm, "RotateSkill", {"angle": -goal.angle, "duration": 5.0}, goal.id),
            step(arm, "GraspSkill", {"action": "open"}, goal.id),
            _move(arm, point + [0, 0, 0.15], goal.id, duration=2.5))


def stage_poses(request, cart_pose):
    pose = np.array(cart_pose, dtype=float)
    for stage in request.stages:
        if stage.nav:
            pose[:2] = stage.nav['target']
            if stage.nav.get('yaw') is not None:
                pose[2] = stage.nav['yaw']
        yield stage, pose.copy()


def _check_bundle(executor, steps, pose):
    for item in steps:
        params = item.get("params", {})
        if "target_pos" in params and params["target_pos"][2] > 0.9:
            raise ValueError("技能目标超过 z=0.9m 安全上限")
    return executor.parse({"plan": list(steps)}, pose[:2], pose[2])["phases"][0]


def generate_candidates(request, snapshot, cart_pose, executor, config):
    objects = scene_objects(snapshot)
    goals = {g.id: g for g in request.goals}
    main, auxiliaries, rejected = [], [], []
    for stage, pose in stage_poses(request, cart_pose):
        for goal_id in stage.goals:
            goal = goals[goal_id]
            for arm in (("R",) if goal.operation == "inspect" else ("L", "R")):
                candidate_id = f"{goal_id}:{arm}"
                try:
                    steps = main_steps(goal, arm, objects)
                    parsed = _check_bundle(executor, steps, pose)
                    base, _ = executor.arm_pose(arm, pose[:2], pose[2])
                    positions = [s['params']['target_pos'] for s in steps if 'target_pos' in s['params']]
                    margin = min(1.0 - np.linalg.norm(np.asarray(p) - base) / executor.MAX_REACH for p in positions)
                    joints = [s.target_joints for s in parsed[arm] if getattr(s, 'target_joints', None) is not None]
                    route = [np.asarray(INITIAL_JOINTS), *joints, np.asarray(INITIAL_JOINTS)]
                    effort = sum(float(np.mean(np.abs(b - a))) for a, b in zip(route, route[1:])) / pi
                    score = config.reach_weight * max(0.0, margin) + config.posture_weight / (1 + effort)
                    main.append(MainCandidate(candidate_id, goal_id, stage.id, arm, tuple(steps), float(score),
                                              arm == goal.preferred_arm))
                except ValueError as exc:
                    rejected.append({"candidate": candidate_id, "reason": str(exc)})

        shade_goals = [goals[i] for i in stage.goals if goals[i].shade != 'none']
        if not shade_goals:
            continue
        try:
            grasp = object_point(objects, 'shade_board', 'grasp_center')
            light = object_point(objects, 'glare_light')
        except ValueError as exc:
            rejected.append({"candidate": f"shade:phase{stage.id}", "reason": str(exc)})
            continue
        for goal in shade_goals:
            target = object_point(objects, goal.target, 'handle_center')
            for height, offset in ((z, x) for z in config.shade_heights for x in config.shade_lateral_offsets):
                candidate_id = f"shade:{stage.id}:{goal.id}:{height:.3f}:{offset:+.3f}"
                try:
                    fraction = (height - light[2]) / (target[2] - light[2])
                    if not 0 < fraction < 1:
                        raise ValueError("遮光位置必须在光源与目标之间")
                    position = light + fraction * (target - light)
                    position[0] += offset
                    # Fixed geometric proxy: a ray crosses a disk of radius 0.10m
                    # at this height. Actual board pose is checked during execution.
                    covers = set()
                    for other in shade_goals:
                        point = object_point(objects, other.target, 'handle_center')
                        t = (height - light[2]) / (point[2] - light[2])
                        if 0 < t < 1 and np.linalg.norm((light + t * (point - light) - position)[:2]) <= 0.10:
                            covers.add(other.id)
                    approach = grasp + [0, 0, 0.115]
                    lift = grasp + [0, 0, 0.185]
                    base, _ = executor.arm_pose('R', pose[:2], pose[2])
                    transfer = base + SO3.Rz(pose[2]).R @ np.array([-0.35, -0.3, 0.35])
                    setup = (_move('R', approach, role='shade'), _move('R', grasp, role='shade', duration=2.5),
                             step('R', 'GraspSkill', {'action': 'close', 'gripper_closed': config.shade_gripper_closed,
                                                     'wait_time': 1.2}, role='shade'),
                             _move('R', lift, role='shade', duration=2.5),
                             _move('R', transfer, role='shade'),
                             _move('R', position, role='shade', duration=4.0, tilt_y=0.6))
                    cleanup = (_move('R', transfer, role='shade_cleanup'),
                               _move('R', lift, role='shade_cleanup'), _move('R', grasp, role='shade_cleanup'),
                               step('R', 'GraspSkill', {'action': 'open'}, role='shade_cleanup'),
                               _move('R', approach, role='shade_cleanup'))
                    _check_bundle(executor, setup + cleanup, pose)
                    auxiliaries.append(AuxiliaryCandidate(candidate_id, stage.id, 'R', setup, cleanup,
                                                          frozenset(covers), tuple(position)))
                except (ValueError, ZeroDivisionError) as exc:
                    rejected.append({"candidate": candidate_id, "reason": str(exc)})
    return tuple(main), tuple(auxiliaries), rejected
