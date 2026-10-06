"""Nominal skill profiles and time-aligned measured/reference telemetry."""

import mujoco
import numpy as np

from src.config.robot import INITIAL_JOINTS
from src.motion_planning import JointTrajectory
from src.pipeline.navigation import NavigationTrajectory


def _targets(skills, initial, return_home):
    current = np.asarray(initial).copy()
    segments = []
    for skill in skills:
        target = getattr(skill, 'target_joints', None)
        if target is None and hasattr(skill, 'angle'):
            target = current.copy()
            target[skill.joint_index] += skill.angle
        if target is None:
            target = current.copy()
        duration = float(getattr(skill, 'duration', getattr(skill, 'wait_time', .8)))
        segments.append((current.copy(), np.asarray(target).copy(), duration))
        current = np.asarray(target).copy()
    if return_home and skills:
        segments.append((current.copy(), np.array(INITIAL_JOINTS), 3.))
    return segments


def _sample(segments, time, initial):
    for start, target, duration in segments:
        if time < duration:
            return JointTrajectory(start, target, duration).interpolate(time)
        time -= duration
    return segments[-1][1] if segments else np.asarray(initial)


def nominal_preview(runner, prepared, interval=.1):
    """FK of nominal cubic joint references, including nav/yaw and assistance.

    This does not simulate control or admittance and does not use measured paths.
    Scratch MjData never changes the live robot. Timing is nominal; runtime
    telemetry also records the active skill's reference for aligned error plots.
    """
    parsed = runner.plan_executor.parse(prepared, runner._get_cart_state(),
                                        float(runner.data.qpos[runner._cart_yaw_qpos]))
    data = mujoco.MjData(runner.model)
    data.qpos[:] = runner.data.qpos
    sites = {arm: mujoco.mj_name2id(runner.model, mujoco.mjtObj.mjOBJ_SITE, 'pinch' + ('_R' if arm == 'R' else ''))
             for arm in ('L', 'R')}
    paths = {'L': [], 'R': []}
    timeline = []
    clock = 0.
    def record(time, phase):
        mujoco.mj_forward(runner.model, data)
        row = {'t': round(time, 4), 'phase': phase}
        for arm in paths:
            point = data.site_xpos[sites[arm]].round(6).tolist()
            paths[arm].append(point)
            row[arm] = point
        timeline.append(row)
    record(0., 0)
    for index, phase in enumerate(parsed['phases']):
        nav = phase.get('nav')
        if nav:
            profile = NavigationTrajectory(data.qpos[[runner._cart_x_qpos, runner._cart_y_qpos]],
                                           np.array(nav['target']), clock,
                                           float(data.qpos[runner._cart_yaw_qpos]), nav.get('yaw'))
            for elapsed in np.linspace(0, profile.duration, max(2, int(profile.duration / interval) + 1)):
                pos, yaw = profile.sample(clock + elapsed)
                data.qpos[[runner._cart_x_qpos, runner._cart_y_qpos]] = pos
                if yaw is not None: data.qpos[runner._cart_yaw_qpos] = yaw
                record(clock + elapsed, index)
            clock += profile.duration
        mode = phase['mode']
        if mode == 'assist_R_then_L':
            groups = [({'R': phase['R']}, False), ({'L': phase['L']}, True),
                      ({'R': phase['R_cleanup']}, True)]
        else:
            groups = [({arm: phase[arm] for arm in ('L', 'R')}, True)]
        for group, return_home in groups:
            profiles = {}
            for arm, skills in group.items():
                address = runner._L_qpos_start if arm == 'L' else runner._R_qpos_start
                initial = data.qpos[address:address + 6].copy()
                segments = _targets(skills, initial, return_home)
                profiles[arm] = (address, initial, segments)
            duration = max((sum(s[2] for s in value[2]) for value in profiles.values()), default=0.)
            for elapsed in np.linspace(0, duration, max(2, int(duration / interval) + 1)):
                for address, initial, segments in profiles.values():
                    data.qpos[address:address + 6] = _sample(segments, elapsed, initial)
                record(clock + elapsed, index)
            clock += duration
    return {'point': 'gripper pinch', 'axes': 'world', 'timing': 'nominal',
            'duration': round(clock, 4), 'paths': paths, 'timeline': timeline}


def telemetry_sample(runner, scratch):
    # mj_step integrates qpos after computing its sensor/force buffers. Never
    # call mj_forward on live data here: that would change the next controller
    # sample and alternate its finite-difference velocity between 2x and 0x.
    scratch.qpos[:] = runner.data.qpos
    mujoco.mj_kinematics(runner.model, scratch)
    sites = {arm: mujoco.mj_name2id(runner.model, mujoco.mjtObj.mjOBJ_SITE, 'pinch' + ('_R' if arm == 'R' else ''))
             for arm in ('L', 'R')}
    measured = {arm: scratch.site_xpos[site].copy() for arm, site in sites.items()}
    for qpos, actuator in [(runner._cart_x_qpos, runner._cart_x_ctrl),
                           (runner._cart_y_qpos, runner._cart_y_ctrl),
                           (runner._cart_yaw_qpos, runner._cart_yaw_ctrl)]:
        scratch.qpos[qpos] = runner.data.ctrl[actuator]
    for arm, values in runner.control_references.items():
        address = runner._L_qpos_start if arm == 'L' else runner._R_qpos_start
        scratch.qpos[address:address + 6] = values['position']
    mujoco.mj_kinematics(runner.model, scratch)
    row = {'t': round(float(runner.data.time), 4), 'phase': runner._phase_index, 'arms': {},
           'qpos': runner.data.qpos.tolist(),
           'solver_t': round(max(0., float(runner.data.time - runner.model.opt.timestep)), 4)}
    for label, arm in runner.arms.items():
        site = sites[label]
        address = runner._L_qpos_start if label == 'L' else runner._R_qpos_start
        actual_q = runner.data.qpos[address:address + 6].copy()
        reference = runner.control_references.get(label, {})
        command = np.asarray(reference.get('position', INITIAL_JOINTS))
        actual = measured[label]
        target = scratch.site_xpos[site].copy()
        executor = arm.executor
        skill = executor.current_skill if executor and not executor.is_all_complete else None
        wrench = arm.ctx.get_external_force() if arm.ctx else np.zeros(6)
        row['arms'][label] = {'actual_q': actual_q.round(6).tolist(), 'desired_q': command.round(6).tolist(),
                             'error_q': (command - actual_q).round(6).tolist(),
                             'actual_tcp': actual.round(6).tolist(), 'reference_tcp': target.round(6).tolist(),
                             'tcp_error_m': float(np.linalg.norm(target - actual)),
                             'force': wrench[:3].round(5).tolist(), 'moment': wrench[3:].round(5).tolist(),
                             'torque': np.asarray(reference.get('torque', np.zeros(6))).round(5).tolist(),
                             'skill': skill.name if skill else ('保持' if label == 'R' and runner._R_cleanup_pending else '待命'),
                             'skill_index': executor.current_index if executor else -1,
                             'skill_count': len(executor.skills) if executor else 0}
    return row


def trajectory_metrics(samples):
    if not samples: return {}
    return {arm: {'tcp_rmse_m': float(np.sqrt(np.mean([s['arms'][arm]['tcp_error_m'] ** 2 for s in samples]))),
                  'joint_rmse_rad': float(np.sqrt(np.mean([np.square(s['arms'][arm]['error_q']) for s in samples]))),
                  'peak_force_N': max(float(np.linalg.norm(s['arms'][arm]['force'])) for s in samples),
                  'peak_torque_Nm': max(float(np.max(np.abs(s['arms'][arm]['torque']))) for s in samples)}
            for arm in ('L', 'R')}
