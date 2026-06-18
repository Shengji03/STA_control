from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager, suppress
from datetime import datetime
from pathlib import Path
from threading import RLock, Thread
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.llm_planner.dialog_planner import DialogPlanner
from src.web_sim.mujoco_session import MujocoStreamSession, SimulationConfig

from .config import BackendSettings
from .schemas import (
    CameraRequest,
    ControlRequest,
    HealthResponse,
    LogRecord,
    SimulationStatus,
    TaskDispatchRequest,
    TaskRecord,
)


class SimulationRuntime:
    def __init__(self, settings: BackendSettings, enabled: bool):
        self.settings = settings
        self.enabled = enabled
        self.session: MujocoStreamSession | None = None
        self._lock = RLock()
        self._task_counter = 0
        self._log_counter = 0
        self._tasks: list[TaskRecord] = []
        self._logs: list[LogRecord] = []
        self._add_log("INFO", "FastAPI 服务已启动")
        if enabled:
            config = SimulationConfig(
                scene_path=settings.scene_path,
                width=settings.width,
                height=settings.height,
                fps=settings.fps,
                jpeg_quality=settings.jpeg_quality,
            )
            self.session = MujocoStreamSession(config)
            self._add_log("INFO", "MuJoCo 渲染流等待连接")
        else:
            self._add_log("WARN", "MuJoCo 仿真会话已禁用")

    def status(self) -> SimulationStatus:
        self._sync_completed_tasks()
        if self.session is None:
            return SimulationStatus(state="disabled")
        return SimulationStatus(**self.session.get_status())

    def control(self, request: ControlRequest) -> SimulationStatus:
        if self.session is None:
            return self.status()
        if request.action == "pause":
            self.session.pause()
        elif request.action == "resume":
            self.session.resume()
        elif request.action == "reset":
            self.session.reset()
        return self.status()

    def camera(self, request: CameraRequest) -> SimulationStatus:
        if self.session is None:
            return self.status()
        self.session.apply_camera_command(request.model_dump())
        return self.status()

    def dispatch_task(self, request: TaskDispatchRequest) -> TaskRecord:
        if self.session is None:
            self._add_log("ERROR", "任务下发失败：MuJoCo 仿真会话未启用")
            raise HTTPException(status_code=409, detail="Simulation session is disabled")

        instruction = request.instruction.strip()
        if not instruction and request.plan is None:
            raise HTTPException(status_code=400, detail="Task instruction or plan is required")
        if not instruction:
            instruction = "执行预置计划"
        task_id = self._next_task_id()
        is_preplanned = request.plan is not None
        task = TaskRecord(
            id=task_id,
            instruction=instruction,
            scene=request.scene,
            mode=request.mode,
            status="running" if is_preplanned else "planning",
            created_at=self._now(),
            message="任务已下发到 TaskRunner 执行队列" if is_preplanned else "任务已创建，等待 LLM 规划",
        )
        total_time = max(request.total_time, 1.0)

        try:
            if request.plan is not None:
                self.session.dispatch_pipeline_task(
                    task_id=task_id,
                    instruction=instruction,
                    total_time=total_time,
                    plan_dict=request.plan,
                )
            else:
                self.session.start_planning_task(
                    task_id=task_id,
                    instruction=instruction,
                    total_time=total_time,
                )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        self._insert_task(task)
        self._add_log("INFO", f"任务 {task_id} 已下发：{instruction}")
        if request.plan is not None:
            self._add_log("INFO", f"任务 {task_id} 已进入 TaskRunner 执行队列")
        else:
            self._add_log("INFO", f"任务 {task_id} 已进入 LLM 规划队列")
            self._start_planning_thread(task_id, instruction)
        return task

    def task_history(self) -> list[TaskRecord]:
        self._sync_completed_tasks()
        with self._lock:
            return list(self._tasks)

    def logs(self) -> list[LogRecord]:
        self._sync_completed_tasks()
        with self._lock:
            return list(self._logs)

    def close(self) -> None:
        if self.session is not None:
            self.session.close()

    def _sync_completed_tasks(self) -> None:
        if self.session is None:
            return
        for event in self.session.consume_task_events():
            task_id = event["task_id"]
            changed = self._update_task(task_id, status=event["status"], message=event["message"])
            if changed:
                level = "INFO" if event["status"] == "completed" else "ERROR"
                self._add_log(level, f"任务 {task_id} {event['message']}")

    def _start_planning_thread(self, task_id: str, instruction: str) -> None:
        thread = Thread(
            target=self._run_planning_job,
            args=(task_id, instruction),
            daemon=True,
            name=f"sta-llm-planner-{task_id}",
        )
        thread.start()

    def _run_planning_job(self, task_id: str, instruction: str) -> None:
        if self.session is None:
            return
        self._add_log("INFO", f"任务 {task_id} 开始 LLM 规划")
        try:
            llm_planner = self._build_llm_planner()
            plan_dict = self.session.create_plan_with_llm(instruction, llm_planner)
            self.session.activate_planned_task(task_id, plan_dict)
            self._update_task(task_id, status="running", message="LLM 规划完成，任务执行中")
            self._add_log("INFO", f"任务 {task_id} LLM 规划完成，已进入 TaskRunner 执行队列")
        except Exception as exc:
            message = self._format_planning_error(exc)
            self.session.fail_planning_task(task_id, message)
            self._update_task(task_id, status="failed", message=message)
            self._add_log("ERROR", f"任务 {task_id} {message}")

    def _build_llm_planner(self) -> DialogPlanner:
        api_key = (
            os.environ.get("STA_LLM_API_KEY")
            or os.environ.get("DASHSCOPE_API_KEY")
            or os.environ.get("DEEPSEEK_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        if not api_key:
            raise HTTPException(
                status_code=400,
                detail="No LLM API key configured. Set STA_LLM_API_KEY before dispatching natural-language tasks.",
            )
        return DialogPlanner(
            api_key=api_key,
            base_url=os.environ.get(
                "STA_LLM_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
            model_name=os.environ.get("STA_LLM_MODEL", "qwen3.5-plus"),
        )

    def _add_log(self, level: str, message: str) -> None:
        with self._lock:
            self._log_counter += 1
            self._logs.insert(
                0,
                LogRecord(
                    id=f"LOG-{self._log_counter:04d}",
                    timestamp=self._now(),
                    level=level,
                    message=message,
                ),
            )

    def _next_task_id(self) -> str:
        with self._lock:
            self._task_counter += 1
            return f"LAB-{self._task_counter:04d}"

    def _insert_task(self, task: TaskRecord) -> None:
        with self._lock:
            self._tasks.insert(0, task)

    def _update_task(self, task_id: str, *, status: str, message: str) -> bool:
        with self._lock:
            for index, task in enumerate(self._tasks):
                if task.id == task_id:
                    if task.status == status and task.message == message:
                        return False
                    self._tasks[index] = task.model_copy(
                        update={"status": status, "message": message}
                    )
                    return True
        return False

    @staticmethod
    def _format_planning_error(exc: Exception) -> str:
        if isinstance(exc, HTTPException):
            return f"LLM 规划失败：{exc.detail}"
        return f"LLM 规划失败：{exc}"

    @staticmethod
    def _now() -> str:
        return datetime.now().isoformat(timespec="seconds")


def create_app(
    *,
    enable_simulation: bool = True,
    settings: BackendSettings | None = None,
    frontend_dist: Path | None = None,
) -> FastAPI:
    active_settings = settings or BackendSettings()
    if frontend_dist is not None:
        active_settings = BackendSettings(
            scene_path=active_settings.scene_path,
            frontend_dist=Path(frontend_dist),
            width=active_settings.width,
            height=active_settings.height,
            fps=active_settings.fps,
            jpeg_quality=active_settings.jpeg_quality,
        )

    runtime = SimulationRuntime(active_settings, enabled=enable_simulation)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            yield
        finally:
            runtime.close()

    app = FastAPI(title="STA MuJoCo Web", version="0.2.0", lifespan=lifespan)
    app.state.simulation_runtime = runtime

    @app.get("/api/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get("/api/simulation/status", response_model=SimulationStatus)
    def simulation_status() -> SimulationStatus:
        return runtime.status()

    @app.post("/api/simulation/control", response_model=SimulationStatus)
    def simulation_control(request: ControlRequest) -> SimulationStatus:
        return runtime.control(request)

    @app.post("/api/simulation/camera", response_model=SimulationStatus)
    def simulation_camera(request: CameraRequest) -> SimulationStatus:
        return runtime.camera(request)

    @app.post("/api/tasks/dispatch", response_model=TaskRecord)
    def task_dispatch(request: TaskDispatchRequest) -> TaskRecord:
        return runtime.dispatch_task(request)

    @app.get("/api/tasks/history", response_model=list[TaskRecord])
    def task_history() -> list[TaskRecord]:
        return runtime.task_history()

    @app.get("/api/logs", response_model=list[LogRecord])
    def system_logs() -> list[LogRecord]:
        return runtime.logs()

    @app.websocket("/ws/simulation")
    async def simulation_websocket(websocket: WebSocket) -> None:
        await websocket.accept()
        if runtime.session is None:
            await websocket.send_json({"type": "status", "payload": runtime.status().model_dump()})
            await websocket.close()
            return

        send_lock = asyncio.Lock()

        async def send_json(payload: dict[str, Any]) -> None:
            async with send_lock:
                try:
                    await websocket.send_json(payload)
                except RuntimeError as exc:
                    raise WebSocketDisconnect(code=1000, reason=str(exc)) from exc

        async def receive_commands() -> None:
            while True:
                payload = await websocket.receive_json()
                message_type = payload.get("type")
                if message_type == "control":
                    status = runtime.control(ControlRequest(action=payload.get("action")))
                elif message_type == "camera":
                    status = runtime.camera(CameraRequest(**payload))
                else:
                    await send_json({"type": "error", "message": f"Unknown message type: {message_type}"})
                    continue
                await send_json({"type": "status", "payload": status.model_dump()})

        receiver = asyncio.create_task(receive_commands())
        try:
            await send_json({"type": "status", "payload": runtime.status().model_dump()})
            while True:
                frame = runtime.session.step_and_render()
                runtime._sync_completed_tasks()
                async with send_lock:
                    try:
                        await websocket.send_bytes(frame)
                    except RuntimeError as exc:
                        raise WebSocketDisconnect(code=1000, reason=str(exc)) from exc
                await send_json({"type": "status", "payload": runtime.status().model_dump()})
                await asyncio.sleep(runtime.session.frame_interval)
        except (WebSocketDisconnect, asyncio.CancelledError):
            pass
        finally:
            receiver.cancel()
            with suppress(asyncio.CancelledError, WebSocketDisconnect):
                await receiver

    _mount_frontend(app, active_settings.frontend_dist)
    return app


def _mount_frontend(app: FastAPI, frontend_dist: Path) -> None:
    dist = Path(frontend_dist)
    assets = dist / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="frontend-assets")

    index_file = dist / "index.html"
    if not index_file.is_file():
        return

    @app.get("/")
    def frontend_index() -> FileResponse:
        return FileResponse(index_file)

    @app.get("/{frontend_path:path}")
    def frontend_fallback(frontend_path: str) -> FileResponse:
        requested = (dist / frontend_path).resolve()
        try:
            requested.relative_to(dist.resolve())
        except ValueError:
            return FileResponse(index_file)
        if requested.is_file():
            return FileResponse(requested)
        return FileResponse(index_file)
