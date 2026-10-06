from copy import deepcopy
from math import pi

import mujoco
import numpy as np
import pytest

from src.config.paths import DEFAULT_SCENE_PATH
from src.config.robot import INITIAL_JOINTS, MOBILE_ROBOT
from src.perception.object_registry import SceneObjectRegistry
from src.perception.world_state import WorldStateBuilder
from src.pipeline.plan_executor import PlanExecutor
from src.task_planning.models import PlanningConfig
from src.task_planning.planning_service import PlanningService


OFFSETS = {'L': np.array(MOBILE_ROBOT.left_offset), 'R': np.array(MOBILE_ROBOT.right_offset)}
YAWS = {'L': MOBILE_ROBOT.left_yaw, 'R': MOBILE_ROBOT.right_yaw}


@pytest.fixture
def environment():
    model = mujoco.MjModel.from_xml_path(str(DEFAULT_SCENE_PATH))
    data = mujoco.MjData(model)
    def address(name):
        return model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
    for arm in ('', '_R'):
        start = address('shoulder_pan_joint' + arm)
        data.qpos[start:start + 6] = INITIAL_JOINTS
    pose = np.array(MOBILE_ROBOT.cart_start)
    for index, name in enumerate(('cart_x', 'cart_y', 'cart_yaw')):
        data.qpos[address(name)] = pose[index]
    mujoco.mj_forward(model, data)
    world = WorldStateBuilder(model, data, SceneObjectRegistry.from_scene_path(str(DEFAULT_SCENE_PATH))).build()
    snapshot = {'world_state': world}
    return model, data, pose, snapshot


def request(shade='none', preferred='R', target='valve_1', nav=(.45, 0)):
    return {'goals': [{'id': 'v', 'object': target, 'operation': 'rotate', 'angle': pi,
                       'shade': shade, 'preferred_arm': preferred}],
            'stages': [{'nav': {'target': list(nav)}, 'goals': ['v']}]}


def service(environment, config=None):
    model, data, _, _ = environment
    return PlanningService(model, data, OFFSETS, YAWS, config)


def test_real_scene_arm_allocation_changes_preference_and_does_not_mutate_input(environment):
    svc = service(environment)
    data = request()
    original = deepcopy(data)
    prepared = svc.prepare(data, environment[3], '旋拧1号阀门180度', environment[2])
    report = prepared['optimization']
    assert data == original
    assert report['assignments'][0]['arm'] == 'L'
    assert report['gain_over_preference'] > 0
    assert report['covered_goal_count'] == report['goal_count'] == 1
    assert report['auxiliary_method'] == 'exact'
    assert prepared['execution_phases'][0]['mode'] == 'single_L'


def test_real_scene_required_shade_has_explicit_mode_cleanup_and_no_resource_conflict(environment):
    svc = service(environment)
    original_qpos = environment[1].qpos.copy()
    prepared = svc.prepare(request('required'), environment[3], '遮光并旋拧1号阀门180度', environment[2])
    report = prepared['optimization']
    assert report['assignments'][0]['arm'] == 'L'
    assert report['auxiliaries'][0]['arm'] == 'R'
    assert report['auxiliaries'][0]['covers'] == ['v']
    phase = prepared['execution_phases'][0]
    assert phase['mode'] == 'assist_R_then_L'
    assert any(s['params'].get('action') == 'open' for s in phase['cleanup_R'])
    assert report['rejected_policy_count'] > 0  # Interfering candidate poses were checked and discarded.
    assert report['preferred_arm_baseline'] is None  # R main cannot also hold the board.
    np.testing.assert_array_equal(environment[1].qpos, original_qpos)
    assert any('shade_board' in p['reason'] for p in report['rejected_policies'])


def test_multigoal_plan_keeps_every_goal_and_the_navigation_order(environment):
    data = request(preferred='L')
    data['goals'].append({'id': 'v2', 'object': 'valve_2', 'operation': 'rotate', 'angle': pi})
    data['stages'].append({'nav': {'target': [3.5, 0]}, 'goals': ['v2']})
    prepared = service(environment).prepare(data, environment[3], '旋拧1、2号阀门180度', environment[2])
    assert {a['goal_id'] for a in prepared['optimization']['assignments']} == {'v', 'v2'}
    assert [p['nav']['target'] for p in prepared['execution_phases']] == [[.45, 0], [3.5, 0]]


def test_inspection_remains_right_arm_only(environment):
    data = {'goals': [{'id': 'g', 'object': 'gauge_1', 'operation': 'inspect', 'preferred_arm': 'L'}],
            'stages': [{'nav': {'target': [-.1, 0]}, 'goals': ['g']}]}
    prepared = service(environment).prepare(data, environment[3], '巡检1号压力表', environment[2])
    assert prepared['optimization']['main_candidate_count'] == 1
    assert prepared['optimization']['assignments'][0]['arm'] == 'R'


def test_failed_optimized_candidate_uses_only_a_validated_alternative(environment, monkeypatch):
    svc = service(environment)
    monkeypatch.setattr(svc.validator, 'give_feedback', lambda plan, *_:
                        (plan['execution_phases'][0]['mode'] == 'single_R', 'injected collision'))
    prepared = svc.prepare(request(), environment[3], '', environment[2])
    assert prepared['optimization']['assignments'][0]['arm'] == 'R'
    assert prepared['optimization']['rejected_policy_count'] == 1
    monkeypatch.setattr(svc.validator, 'give_feedback', lambda *_: (False, 'injected collision'))
    with pytest.raises(ValueError, match='均未通过'):
        svc.prepare(request(), environment[3], '', environment[2])


def test_required_helper_budget_and_unreachable_target_do_not_fall_back_silently(environment):
    with pytest.raises(ValueError, match='全部目标'):
        service(environment, PlanningConfig(max_auxiliaries=0)).prepare(request('required'), environment[3], '', environment[2])
    with pytest.raises(ValueError, match='没有可执行'):
        service(environment).prepare(request(nav=(-3, -.5)), environment[3], '', environment[2])


def test_recognizable_legacy_valve_bundle_preserves_all_steps(environment):
    svc = service(environment)
    from src.task_planning.candidates import main_steps
    from src.task_planning.models import Goal
    steps = list(main_steps(Goal('v', 'valve_1', 'rotate', pi), 'R', environment[3]['world_state']['objects']))
    steps[0]['params']['duration'] = 7.0
    raw = {'plan': [{'arm': 'R', 'skill': 'NavSkill', 'params': {'target': [.45, 0]}}, *steps]}
    prepared = svc.prepare(raw, environment[3], '', environment[2])
    assert prepared['optimization']['source'] == 'legacy_bundles'
    assert len(prepared['execution_phases'][0]['steps']) == 6
    assert prepared['execution_phases'][0]['steps'][0]['params']['duration'] == 7.0


def test_cart_rotation_and_world_orientation_match_mujoco_gripper_point(environment):
    model, data, _, _ = environment
    executor = PlanExecutor(OFFSETS, YAWS)
    pose = np.array([.45, 0, pi / 2])
    point = [.7, .65, .658]
    q = executor._resolve_target_joints('L', point, pose[:2], cart_yaw=pose[2])
    assert q is not None
    for name, value in zip(('cart_x', 'cart_y', 'cart_yaw'), pose):
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[joint]] = value
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'shoulder_pan_joint')
    address = model.jnt_qposadr[joint]
    data.qpos[address:address + 6] = q
    mujoco.mj_forward(model, data)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'pinch')
    np.testing.assert_allclose(data.site_xpos[site], point, atol=.008)


def test_ordinary_right_grasp_in_parallel_does_not_trigger_shade_mode():
    executor = PlanExecutor(OFFSETS, YAWS)
    plan = {'plan': [{'arm': arm, 'skill': 'GraspSkill', 'params': {'action': 'close'}} for arm in ('L', 'R')]}
    parsed = executor.parse(plan, np.zeros(2))
    assert parsed['phases'][0]['mode'] == 'parallel'
    malformed = {'execution_phases': [{'mode': 'assist_R_then_L', 'steps': plan['plan']}]}
    with pytest.raises(ValueError, match='R_cleanup'):
        executor.parse(malformed, np.zeros(2))
