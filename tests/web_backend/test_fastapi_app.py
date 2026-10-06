import time

from fastapi.testclient import TestClient

from src.web_backend.app import create_app
from src.web_backend.config import BackendSettings
from src.web_sim.mujoco_session import DEFAULT_SCENE_PATH


SAMPLE_NAV_PLAN = {
    "plan": [
        {
            "step": 1,
            "arm": "L",
            "skill": "NavSkill",
            "params": {"target": [-2.8, -0.5]},
            "description": "小车沿原项目 NavSkill 移动",
        }
    ]
}


def test_health_endpoint_reports_running_backend():
    client = TestClient(create_app(enable_simulation=False))

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_status_endpoint_works_without_simulation_session():
    client = TestClient(create_app(enable_simulation=False))

    response = client.get("/api/simulation/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["state"] == "disabled"
    assert payload["available_cameras"] == []


def test_simulation_control_endpoint_reports_disabled_session():
    client = TestClient(create_app(enable_simulation=False))

    response = client.post("/api/simulation/control", json={"action": "pause"})

    assert response.status_code == 200
    assert response.json()["state"] == "disabled"


def test_simulation_websocket_reports_disabled_when_session_is_disabled():
    client = TestClient(create_app(enable_simulation=False))

    with client.websocket_connect("/ws/simulation") as websocket:
        message = websocket.receive_json()

    assert message["type"] == "status"
    assert message["payload"]["state"] == "disabled"


def test_simulation_websocket_streams_fresh_status_after_frames():
    settings = BackendSettings(
        scene_path=DEFAULT_SCENE_PATH,
        width=80,
        height=60,
        fps=5,
        jpeg_quality=70,
    )

    with TestClient(create_app(settings=settings)) as client:
        with client.websocket_connect("/ws/simulation") as websocket:
            initial = websocket.receive_json()
            frame = websocket.receive_bytes()
            refreshed = websocket.receive_json()

    assert initial["type"] == "status"
    assert frame[:2] == b"\xff\xd8"
    assert refreshed["type"] == "status"
    assert refreshed["payload"]["frame_index"] > initial["payload"]["frame_index"]


def test_app_serves_frontend_dist_when_present(tmp_path):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<div id='app'></div>", encoding="utf-8")

    client = TestClient(create_app(enable_simulation=False, frontend_dist=dist))
    response = client.get("/")

    assert response.status_code == 200
    assert "app" in response.text


def test_task_dispatch_records_history_logs_and_active_simulation_task():
    settings = BackendSettings(
        scene_path=DEFAULT_SCENE_PATH,
        width=80,
        height=60,
        fps=5,
        jpeg_quality=70,
    )

    with TestClient(create_app(settings=settings)) as client:
        response = client.post(
            "/api/tasks/dispatch",
            json={
                "instruction": "向前移动小车，执行一次可见的实验室演示动作",
                "scene": "scene5_glare.xml",
                "mode": "实时仿真",
                "total_time": 300,
                "fps": 20,
                "record_tcp": True,
                "plan": SAMPLE_NAV_PLAN,
            },
        )

        assert response.status_code == 200
        task = response.json()
        assert task["status"] == "running"
        assert task["id"].startswith("LAB-")

        status = client.get("/api/simulation/status").json()
        assert status["active_task"]["id"] == task["id"]
        assert status["active_task"]["runner_state"] == "INIT"

        history = client.get("/api/tasks/history").json()
        assert history[0]["id"] == task["id"]

        logs = client.get("/api/logs").json()
        assert any(task["id"] in entry["message"] for entry in logs)


def test_task_dispatch_without_plan_uses_async_llm_planning(monkeypatch, tmp_path):
    monkeypatch.setenv("STA_ENV_FILE", str(tmp_path / "missing.env"))
    monkeypatch.delenv("STA_LLM_API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = BackendSettings(
        scene_path=DEFAULT_SCENE_PATH,
        width=80,
        height=60,
        fps=5,
        jpeg_quality=70,
    )

    with TestClient(create_app(settings=settings)) as client:
        response = client.post(
            "/api/tasks/dispatch",
            json={
                "instruction": "遮光并旋拧1号阀门180度",
                "scene": "scene5_glare.xml",
                "mode": "实时仿真",
                "total_time": 300,
                "fps": 20,
                "record_tcp": True,
            },
        )

        assert response.status_code == 200
        task = response.json()
        assert task["status"] == "planning"

        deadline = time.monotonic() + 2.0
        latest = task
        while time.monotonic() < deadline:
            latest = client.get("/api/tasks/history").json()[0]
            if latest["status"] == "failed":
                break
            time.sleep(0.05)

        assert latest["status"] == "failed"
        assert "LLM 规划失败" in latest["message"]
        assert "No LLM API key configured" in latest["message"]

        logs = client.get("/api/logs").json()
        assert any("LLM 规划队列" in entry["message"] for entry in logs)
        assert any("LLM 规划失败" in entry["message"] for entry in logs)


def test_semantic_task_history_exposes_optimization_report_without_api_calls(monkeypatch):
    from math import pi
    from src.web_backend.app import SimulationRuntime

    class Planner:
        def plan(self, snapshot, instruction, extra_context=''):
            return {'goals': [{'id': 'v', 'object': 'valve_1', 'operation': 'rotate',
                               'angle': pi, 'preferred_arm': 'R'}],
                    'stages': [{'nav': {'target': [.45, 0]}, 'goals': ['v']}]}

    monkeypatch.setattr(SimulationRuntime, '_build_llm_planner', lambda _self: Planner())
    settings = BackendSettings(scene_path=DEFAULT_SCENE_PATH, width=80, height=60, fps=5)
    with TestClient(create_app(settings=settings)) as client:
        dispatched = client.post('/api/tasks/dispatch', json={'instruction': '旋拧1号阀门180度'}).json()
        deadline = time.monotonic() + 3
        history = None
        while time.monotonic() < deadline:
            history = client.get('/api/tasks/history').json()[0]
            if history['status'] != 'planning':
                break
            time.sleep(.01)
        assert history['id'] == dispatched['id']
        assert history['status'] == 'running'
        assert history['optimization']['assignments'][0]['arm'] == 'L'
        assert history['optimization']['covered_goal_count'] == 1


def test_task_dispatch_reports_disabled_simulation_session():
    client = TestClient(create_app(enable_simulation=False))

    response = client.post(
        "/api/tasks/dispatch",
        json={"instruction": "测试任务", "scene": "scene5_glare.xml", "mode": "实时仿真"},
    )

    assert response.status_code == 409
