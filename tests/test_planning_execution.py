from math import pi

import mujoco
import pytest

from src.config.paths import DEFAULT_SCENE_PATH
from src.pipeline.task_runner import TaskRunner
from src.web_sim.mujoco_session import MujocoStreamSession, SimulationConfig


def task_request(shade='none', angle=pi):
    return {'goals': [{'id': 'v1', 'object': 'valve_1', 'operation': 'rotate', 'angle': angle,
                       'shade': shade, 'preferred_arm': 'R'}],
            'stages': [{'nav': {'target': [.45, 0]}, 'goals': ['v1']}]}


@pytest.mark.parametrize('shade,angle', [('none', pi / 2), ('required', pi)])
def test_headless_pipeline_verifies_actual_valve_angle_and_shade_board_return(shade, angle):
    runner = TaskRunner(str(DEFAULT_SCENE_PATH), plan_dict=task_request(shade, angle))
    try:
        runner.run(total_time=100, use_viewer=False, show_trajectory=False)
        assert runner.state == 'DONE'
        assert runner.execution_report['verified'] is True
        effect = runner.execution_report['goals'][0]
        assert effect['angle_error_rad'] < .1
        if shade == 'required':
            assert effect['shade_samples'] > 100
            assert effect['shade_fraction'] >= .9
            assert effect['board_return_error_m'] < .08
            assert runner.arm_R.ctx.gripper_target == 0
    finally:
        runner.perception._renderer.close()


def test_right_arm_main_operation_uses_real_jacobian_and_preserves_valve_effect():
    plan = task_request(angle=pi / 2)
    plan['stages'][0]['nav']['target'] = [.9, 0]  # L is out of reach, R must operate.
    runner = TaskRunner(str(DEFAULT_SCENE_PATH), plan_dict=plan)
    try:
        runner.run(total_time=100, use_viewer=False, show_trajectory=False)
        assert runner.planning_report['assignments'][0]['arm'] == 'R'
        assert runner.state == 'DONE'
        assert runner.execution_report['verified'] is True
        assert runner.execution_report['goals'][0]['angle_error_rad'] < .1
    finally:
        runner.perception._renderer.close()


def test_effect_monitor_does_not_claim_required_shade_success_from_valve_motion_alone():
    from src.task_planning.effects import EffectMonitor
    from src.task_planning.models import PlanningConfig
    from src.perception.object_registry import SceneObjectRegistry
    model = mujoco.MjModel.from_xml_path(str(DEFAULT_SCENE_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    goals = task_request('required')['goals']
    monitor = EffectMonitor(model, data, SceneObjectRegistry.from_scene_path(str(DEFAULT_SCENE_PATH)), goals, PlanningConfig())
    monitor.start_phase(['v1'])
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'valve_joint')
    data.qpos[model.jnt_qposadr[joint]] = pi
    assert monitor.finish_phase(['v1']) is False
    assert monitor.report()['goals'][0]['angle_error_rad'] == 0
    assert monitor.report()['goals'][0]['shade_fraction'] is None


def test_a_requested_full_turn_is_not_satisfied_by_zero_motion():
    from src.task_planning.effects import EffectMonitor
    from src.task_planning.models import PlanningConfig
    from src.perception.object_registry import SceneObjectRegistry
    model = mujoco.MjModel.from_xml_path(str(DEFAULT_SCENE_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    monitor = EffectMonitor(model, data, SceneObjectRegistry.from_scene_path(str(DEFAULT_SCENE_PATH)),
                            task_request(angle=2 * pi)['goals'], PlanningConfig())
    monitor.start_phase(['v1'])
    assert monitor.finish_phase(['v1']) is False
    assert monitor.report()['goals'][0]['angle_error_rad'] == pytest.approx(2 * pi)


def test_cli_and_web_llm_paths_share_the_optimizer_and_return_explicit_phases():
    class Planner:
        def __init__(self):
            self.calls = 0
        def plan(self, snapshot, instruction, extra_context=''):
            assert snapshot['images'] == {}
            self.calls += 1
            return task_request()

    planner = Planner()
    runner = TaskRunner(str(DEFAULT_SCENE_PATH), llm_planner=planner, task_instruction='旋拧1号阀门180度')
    try:
        runner._do_planning()
        cli_assignment = runner.planning_report['assignments']
        assert runner._phases[0]['mode'] == 'single_L'
    finally:
        runner.perception._renderer.close()

    session = MujocoStreamSession(SimulationConfig(width=80, height=60))
    try:
        session.start_planning_task(task_id='OPT-TEST', instruction='旋拧1号阀门180度', total_time=100)
        plan = session.create_plan_with_llm('旋拧1号阀门180度', planner)
        assert plan['optimization']['assignments'] == cli_assignment
        session.activate_planned_task('OPT-TEST', plan)
        session.step_and_render()
        status = session.get_status()['active_task']
        assert status['optimization']['status'] == 'optimized'
        assert status['runner_state'] == 'PHASE_NAV'
    finally:
        session.close()
        session.runner.perception._renderer.close()
    assert planner.calls == 2


def test_web_retries_joint_planning_errors_as_llm_feedback():
    class Planner:
        def __init__(self):
            self.feedback = []
        def plan(self, snapshot, instruction, extra_context=''):
            self.feedback.append(extra_context)
            if len(self.feedback) == 1:
                bad = task_request()
                bad['stages'][0]['goals'] = []
                return bad
            return task_request()

    session = MujocoStreamSession(SimulationConfig(width=80, height=60))
    planner = Planner()
    try:
        plan = session.create_plan_with_llm('旋拧1号阀门180度', planner)
        assert plan['optimization']['covered_goal_count'] == 1
        assert len(planner.feedback) == 2
        assert '且只能' in planner.feedback[1]
    finally:
        session.close()
        session.runner.perception._renderer.close()
