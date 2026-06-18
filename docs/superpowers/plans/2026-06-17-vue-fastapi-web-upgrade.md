# Vue FastAPI Web Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the prototype static HTML/standard-library web server with a Vue 3 + Vite + TypeScript frontend and a FastAPI backend while preserving the working MuJoCo realtime render stream.

**Architecture:** FastAPI owns the local lab web application, serves health/status APIs, exposes a WebSocket stream, and can serve the built Vue app for one-command production use. Vue handles the enterprise-style dashboard shell, simulation viewport, controls, and typed WebSocket/HTTP client code. The existing `src/web_sim.mujoco_session.MujocoStreamSession` remains the MuJoCo render/session engine.

**Tech Stack:** Python 3.12, FastAPI, Uvicorn, Pydantic, MuJoCo, Pillow, Vue 3, Vite, TypeScript, Pinia, Vitest.

---

### Task 1: FastAPI Backend Shell

**Files:**
- Create: `src/web_backend/__init__.py`
- Create: `src/web_backend/config.py`
- Create: `src/web_backend/schemas.py`
- Create: `src/web_backend/app.py`
- Create: `src/web_backend/main.py`
- Create: `run_enterprise_web.py`
- Create: `requirements-web.txt`
- Test: `tests/web_backend/test_fastapi_app.py`

- [ ] **Step 1: Write failing tests**

```python
from fastapi.testclient import TestClient

from src.web_backend.app import create_app


def test_health_endpoint_reports_running_backend():
    client = TestClient(create_app(enable_simulation=False))
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_status_endpoint_works_without_simulation_session():
    client = TestClient(create_app(enable_simulation=False))
    response = client.get("/api/simulation/status")
    assert response.status_code == 200
    assert response.json()["state"] == "disabled"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/web_backend/test_fastapi_app.py -q`
Expected: fail because `src.web_backend.app` does not exist.

- [ ] **Step 3: Implement minimal FastAPI app**

Create `create_app(enable_simulation: bool = True)` with:
- `GET /api/health`
- `GET /api/simulation/status`
- `POST /api/simulation/control`
- `POST /api/simulation/camera`
- `WS /ws/simulation`

When `enable_simulation=False`, status returns a disabled payload without creating MuJoCo state.

- [ ] **Step 4: Run backend tests**

Run: `pytest tests/web_backend/test_fastapi_app.py -q`
Expected: pass.

### Task 2: WebSocket Streaming Adapter

**Files:**
- Modify: `src/web_backend/app.py`
- Test: `tests/web_backend/test_fastapi_app.py`

- [ ] **Step 1: Write failing WebSocket test**

```python
def test_simulation_websocket_reports_disabled_when_session_is_disabled():
    client = TestClient(create_app(enable_simulation=False))
    with client.websocket_connect("/ws/simulation") as websocket:
        message = websocket.receive_json()
    assert message["type"] == "status"
    assert message["payload"]["state"] == "disabled"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/web_backend/test_fastapi_app.py -q`
Expected: fail until the WebSocket route sends a disabled status.

- [ ] **Step 3: Implement route**

The route accepts the WebSocket, sends current status, and closes immediately when simulation is disabled. When enabled, it streams binary JPEG frames from `MujocoStreamSession.step_and_render()` and accepts JSON control/camera messages.

- [ ] **Step 4: Run tests**

Run: `pytest tests/web_backend/test_fastapi_app.py -q`
Expected: pass.

### Task 3: Vue Frontend Project

**Files:**
- Create: `web_frontend/package.json`
- Create: `web_frontend/index.html`
- Create: `web_frontend/vite.config.ts`
- Create: `web_frontend/tsconfig.json`
- Create: `web_frontend/tsconfig.node.json`
- Create: `web_frontend/src/main.ts`
- Create: `web_frontend/src/App.vue`
- Create: `web_frontend/src/styles/main.css`
- Create: `web_frontend/src/api/http.ts`
- Create: `web_frontend/src/stores/simulation.ts`
- Create: `web_frontend/src/components/SimulationViewport.vue`
- Create: `web_frontend/src/components/ControlPanel.vue`
- Create: `web_frontend/src/components/StatusPanel.vue`
- Create: `web_frontend/src/vite-env.d.ts`

- [ ] **Step 1: Scaffold typed Vue app files**

Use Vue 3 composition API and Pinia. The app has a work-focused dashboard layout: live render viewport, operation controls, camera controls, and telemetry.

- [ ] **Step 2: Add dev proxy**

Configure Vite proxy:
- `/api` -> `http://127.0.0.1:8000`
- `/ws` -> `ws://127.0.0.1:8000`

- [ ] **Step 3: Install dependencies**

Run: `npm install` in `web_frontend`.

- [ ] **Step 4: Build frontend**

Run: `npm run build` in `web_frontend`.
Expected: TypeScript and Vite build pass and produce `web_frontend/dist`.

### Task 4: Production Static Serving

**Files:**
- Modify: `src/web_backend/app.py`
- Test: `tests/web_backend/test_fastapi_app.py`

- [ ] **Step 1: Write failing static-serving test**

```python
def test_app_serves_frontend_dist_when_present(tmp_path):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<div id='app'></div>", encoding="utf-8")
    client = TestClient(create_app(enable_simulation=False, frontend_dist=dist))
    response = client.get("/")
    assert response.status_code == 200
    assert "app" in response.text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/web_backend/test_fastapi_app.py -q`
Expected: fail until static serving is mounted.

- [ ] **Step 3: Implement static serving**

Serve built assets from `web_frontend/dist/assets` and return `index.html` for `/` and unknown frontend routes.

- [ ] **Step 4: Run tests**

Run: `pytest tests/web_backend/test_fastapi_app.py -q`
Expected: pass.

### Task 5: End-to-End Verification

**Files:**
- Modify only if verification finds defects.

- [ ] **Step 1: Run backend tests**

Run: `pytest tests/web_backend tests/web_sim -q`
Expected: all tests pass.

- [ ] **Step 2: Compile Python modules**

Run: `python -m compileall -q src/web_backend src/web_sim run_enterprise_web.py`
Expected: exit code 0.

- [ ] **Step 3: Build Vue app**

Run: `npm run build` in `web_frontend`.
Expected: exit code 0.

- [ ] **Step 4: Start enterprise server**

Run: `python run_enterprise_web.py --http-port 8020 --width 640 --height 360 --fps 10`.
Expected: `GET http://127.0.0.1:8020/api/health` returns `{"status":"ok"}` and browser can load `http://127.0.0.1:8020/`.

- [ ] **Step 5: Verify WebSocket frame**

Connect to `ws://127.0.0.1:8020/ws/simulation`.
Expected: receive a JSON status message followed by binary JPEG bytes beginning with `0xff 0xd8`.
