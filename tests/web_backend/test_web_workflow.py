from math import pi
from threading import Event
import time

import mujoco
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.web_backend.app import create_app, SimulationRuntime
from src.web_backend.config import BackendSettings
from src.web_backend.storage import TaskRepository


NAV = {'plan': [{'arm': 'L', 'skill': 'NavSkill', 'params': {'target': [-2.8, -.5]}}]}


@pytest.fixture
def settings(tmp_path):
    return BackendSettings(width=160, height=90, fps=20, data_dir=tmp_path / 'records')


def wait_status(client, task_id, target, seconds=5):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        task = next(t for t in client.get('/api/tasks/history').json() if t['id'] == task_id)
        if task['status'] in target: return task
        time.sleep(.02)
    raise AssertionError(f'Task {task_id} did not reach {target}: {task}')


def dispatch(client, **kwargs):
    response = client.post('/api/tasks/dispatch', json={'plan': NAV, 'instruction': '导航演示', 'total_time': 10, **kwargs})
    assert response.status_code == 200, response.text
    return response.json()['id']


def test_planning_preview_never_advances_robot_until_explicit_execute(settings):
    with TestClient(create_app(settings=settings)) as client:
        task_id = dispatch(client, mode='仅规划')
        wait_status(client, task_id, {'planned'})
        first = client.get('/api/simulation/status').json()
        time.sleep(.2)
        second = client.get('/api/simulation/status').json()
        assert first['sim_time'] == second['sim_time'] == 0
        detail = client.get(f'/api/tasks/{task_id}/detail').json()
        assert len(detail['preview']['timeline']) > 10
        assert detail['samples'] == []
        assert client.post(f'/api/tasks/{task_id}/execute').status_code == 200
        result = wait_status(client, task_id, {'completed'}, seconds=7)
        assert result['sample_count'] > 20


def test_task_finishes_and_persists_without_any_websocket_and_replays_after_restart(settings):
    with TestClient(create_app(settings=settings)) as client:
        task_id = dispatch(client)
        result = wait_status(client, task_id, {'completed'}, seconds=7)
        detail = client.get(f'/api/tasks/{task_id}/detail').json()
        assert detail['samples'] and result['metrics']['L']['tcp_rmse_m'] >= 0
        data = client.get(f'/api/tasks/{task_id}/export')
        assert data.status_code == 200 and 'attachment' in data.headers['content-disposition']
    with TestClient(create_app(settings=settings)) as client:
        assert client.get('/api/tasks/history').json()[0]['status'] == 'completed'
        saved = client.get(f'/api/tasks/{task_id}/detail').json()
        assert saved['samples'] == detail['samples']
        assert client.post(f'/api/tasks/{task_id}/replay', json={'index': 10}).status_code == 200
        snapshot = client.get('/api/simulation/status').json()
        assert snapshot['state'] == 'replay' and snapshot['sim_time'] == saved['samples'][10]['t']
        assert snapshot['last_task']['id'] == task_id
        time.sleep(.1)
        assert client.get('/api/simulation/status').json()['sim_time'] == snapshot['sim_time']


def test_pause_resume_stop_and_reject_concurrent_task(settings):
    with TestClient(create_app(settings=settings)) as client:
        task_id = dispatch(client)
        wait_status(client, task_id, {'running'})
        assert client.post('/api/tasks/dispatch', json={'plan': NAV}).status_code == 409
        client.post('/api/simulation/control', json={'action': 'pause'})
        first = client.get('/api/simulation/status').json()['sim_time']
        time.sleep(.2)
        assert client.get('/api/simulation/status').json()['sim_time'] == first
        client.post('/api/simulation/control', json={'action': 'resume'})
        time.sleep(.2)
        assert client.get('/api/simulation/status').json()['sim_time'] > first
        client.post('/api/simulation/control', json={'action': 'stop'})
        assert wait_status(client, task_id, {'cancelled'})['status'] == 'cancelled'
        reset = client.post('/api/simulation/control', json={'action': 'reset'}).json()
        assert reset['telemetry'] is None and reset['last_task'] is None
        assert reset['sim_time'] == 0


def test_two_viewers_consume_the_same_frame_without_stepping_again(settings):
    with TestClient(create_app(settings=settings)) as client:
        task_id = dispatch(client, mode='仅规划')
        wait_status(client, task_id, {'planned'})
        with client.websocket_connect('/ws/simulation') as first, client.websocket_connect('/ws/simulation') as second:
            assert first.receive_json()['payload']['sim_time'] == 0
            assert second.receive_json()['payload']['sim_time'] == 0
            assert first.receive_bytes()[:2] == b'\xff\xd8'
            assert second.receive_bytes()[:2] == b'\xff\xd8'
            assert first.receive_json()['payload']['sim_time'] == 0
            assert second.receive_json()['payload']['sim_time'] == 0


def test_cancelled_planning_result_cannot_start_a_replacement_task(settings, monkeypatch):
    entered, release = Event(), Event()
    class SlowPlanner:
        def plan(self, *args, **kwargs):
            entered.set()
            release.wait(5)
            return NAV
    monkeypatch.setattr(SimulationRuntime, '_build_llm_planner', lambda _: SlowPlanner())
    with TestClient(create_app(settings=settings)) as client:
        response = client.post('/api/tasks/dispatch', json={'instruction': '导航演示'})
        first = response.json()['id']
        assert entered.wait(3)
        client.post('/api/simulation/control', json={'action': 'stop'})
        second = dispatch(client, mode='仅规划')
        release.set()
        wait_status(client, second, {'planned'})
        time.sleep(.1)
        status = client.get('/api/simulation/status').json()
        assert status['active_task']['id'] == second and status['active_task']['status'] == 'planned'
        assert wait_status(client, first, {'cancelled'})['status'] == 'cancelled'


def test_scene_and_recording_options_are_real_and_invalid_scene_fails(settings):
    with TestClient(create_app(settings=settings)) as client:
        assert client.post('/api/tasks/dispatch', json={'plan': NAV, 'scene': 'scene3.xml'}).status_code == 409
        task_id = dispatch(client, scene='scene4_pipeline.xml', record_tcp=False)
        wait_status(client, task_id, {'completed'}, seconds=7)
        assert client.get('/api/simulation/status').json()['scene'] == 'scene4_pipeline.xml'
        detail = client.get(f'/api/tasks/{task_id}/detail').json()
        assert detail['samples'] == []
        assert client.post(f'/api/tasks/{task_id}/replay', json={'index': 0}).status_code == 409


def test_preview_is_nominal_fk_and_does_not_modify_live_scene(settings):
    app = create_app(settings=settings)
    with TestClient(app) as client:
        preset = client.get('/api/scenarios').json()[0]
        task_id = dispatch(client, plan=preset['plan'], instruction=preset['instruction'], mode='仅规划', total_time=120)
        wait_status(client, task_id, {'planned'})
        detail = client.get(f'/api/tasks/{task_id}/detail').json()
        preview = detail['preview']
        assert preview['timing'] == 'nominal' and preview['point'] == 'gripper pinch'
        assert preview['duration'] > 50 and len(preview['timeline']) > 500
        assert not np.allclose(preview['paths']['L'], preview['paths']['R'])
        assert detail['samples'] == []
        assert client.get('/api/simulation/status').json()['sim_time'] == 0


def test_repository_rejects_path_traversal_and_marks_interrupted_tasks(settings):
    repository = TaskRepository(settings.data_dir)
    with pytest.raises(ValueError): repository.detail('../secret')
    repository.save({'id': 'LAB-interrupted', 'instruction': 'test', 'scene': 'scene5_glare.xml',
                     'mode': '实时仿真', 'status': 'running', 'created_at': '2026-10-06', 'message': ''})
    app = create_app(settings=settings, enable_simulation=False)
    task = TestClient(app).get('/api/tasks/history').json()[0]
    assert task['status'] == 'interrupted'


def test_telemetry_does_not_refresh_live_sensor_buffers_or_change_robot_state(settings):
    from src.pipeline.task_runner import TaskRunner
    from src.web_sim.trajectory import telemetry_sample
    runner = TaskRunner(str(settings.scene_path))
    try:
        before_sensors = runner.data.sensordata.copy()
        runner.data.qpos[runner._L_qpos_start] += .01
        before_qpos = runner.data.qpos.copy()
        row = telemetry_sample(runner, mujoco.MjData(runner.model))
        np.testing.assert_array_equal(runner.data.sensordata, before_sensors)
        np.testing.assert_array_equal(runner.data.qpos, before_qpos)
        np.testing.assert_allclose(row['arms']['L']['actual_q'], before_qpos[runner._L_qpos_start:runner._L_qpos_start + 6], atol=1e-6)
    finally:
        runner.perception._renderer.close()


def test_planned_overlay_changes_rendered_pixels(settings):
    from src.web_sim.mujoco_session import MujocoStreamSession, SimulationConfig
    session = MujocoStreamSession(SimulationConfig(width=160, height=90))
    try:
        plan = {'goals': [{'id': 'v1', 'object': 'valve_1', 'operation': 'rotate', 'angle': pi}],
                'stages': [{'nav': {'target': [.45, 0]}, 'goals': ['v1']}]}
        session.start_planning_task(task_id='PIXELS', instruction='旋拧1号阀门180度', total_time=100)
        prepared = session.prepare_preplanned(plan, '旋拧1号阀门180度')
        session.activate_planned_task('PIXELS', prepared, execute=False)
        with_overlay = session._render_rgb_locked()
        session.set_overlay(planned=False)
        without_overlay = session._render_rgb_locked()
        assert np.mean(np.abs(with_overlay.astype(float) - without_overlay.astype(float))) > .001
    finally:
        session.close()
