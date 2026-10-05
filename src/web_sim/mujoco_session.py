from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any

import mujoco
import numpy as np

from src.pipeline.task_runner import TaskRunner
from src.config.paths import DEFAULT_SCENE_PATH, PROJECT_ROOT

from .camera import CameraCommand, CameraState
from .frame_codec import encode_jpeg


@dataclass
class _PipelineTask:
    task_id: str
    instruction: str
    total_time: float
    status: str = "planning"
    message: str = "LLM 规划中"
    plan_dict: dict[str, Any] | None = None


@dataclass
class _TaskEvent:
    task_id: str
    status: str
    message: str


@dataclass
class SimulationConfig:
    scene_path: Path = DEFAULT_SCENE_PATH
    width: int = 960
    height: int = 540
    fps: int = 20
    jpeg_quality: int = 80
    camera: CameraState = field(default_factory=CameraState)

    @property
    def frame_interval(self) -> float:
        return 1.0 / max(1, int(self.fps))

    def normalized(self) -> "SimulationConfig":
        return SimulationConfig(
            scene_path=Path(self.scene_path).resolve(),
            width=max(64, int(self.width)),
            height=max(48, int(self.height)),
            fps=max(1, int(self.fps)),
            jpeg_quality=max(1, min(int(self.jpeg_quality), 95)),
            camera=self.camera,
        )


class MujocoStreamSession:
    """Streams frames from the project's TaskRunner model/data."""

    def __init__(self, config: SimulationConfig):
        self.config = config.normalized()
        self._lock = RLock()
        self.runner = TaskRunner(scene_path=str(self.config.scene_path))
        self.model = self.runner.model
        self.data = self.runner.data
        self.renderer = mujoco.Renderer(
            self.model,
            height=self.config.height,
            width=self.config.width,
        )
        self.camera = self.config.camera
        self._free_camera = mujoco.MjvCamera()
        self._state = "running"
        self._frame_index = 0
        self._physics_steps_per_frame = self._calculate_physics_steps_per_frame()
        self._available_cameras = self._collect_camera_names()
        self._active_task: _PipelineTask | None = None
        self._task_events: list[_TaskEvent] = []

    @property
    def available_cameras(self) -> list[str]:
        return list(self._available_cameras)

    @property
    def frame_interval(self) -> float:
        return self.config.frame_interval

    def apply_camera_command(self, payload: dict[str, Any] | CameraCommand) -> None:
        command = payload if isinstance(payload, CameraCommand) else CameraCommand.from_payload(payload)
        if command.action == "set_fixed" and command.camera not in self._available_cameras:
            return
        with self._lock:
            self.camera.apply(command)

    def pause(self) -> None:
        with self._lock:
            self._state = "paused"

    def resume(self) -> None:
        with self._lock:
            self._state = "running"

    def reset(self) -> None:
        with self._lock:
            self.runner.reset_scene()
            self._frame_index = 0
            self._state = "running"
            self._active_task = None
            self._task_events.clear()

    def start_planning_task(
        self,
        *,
        task_id: str,
        instruction: str,
        total_time: float,
    ) -> dict[str, Any]:
        with self._lock:
            self._ensure_idle_locked()
            self.runner.reset_scene()
            self._active_task = _PipelineTask(
                task_id=task_id,
                instruction=instruction,
                total_time=float(total_time),
                status="planning",
                message="LLM 规划中",
            )
            self._state = "running"
            return self._active_task_payload()

    def dispatch_pipeline_task(
        self,
        *,
        task_id: str,
        instruction: str,
        total_time: float,
        plan_dict: dict[str, Any],
    ) -> dict[str, Any]:
        with self._lock:
            self._ensure_idle_locked()
            self._start_runner_locked(
                task_id=task_id,
                instruction=instruction,
                total_time=total_time,
                plan_dict=plan_dict,
            )
            return self._active_task_payload()

    def activate_planned_task(self, task_id: str, plan_dict: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self._active_task is None or self._active_task.task_id != task_id:
                raise ValueError(f"任务 {task_id} 已不再处于可启动状态")
            if self._active_task.status != "planning":
                raise ValueError(f"任务 {task_id} 当前状态为 {self._active_task.status}")
            self._start_runner_locked(
                task_id=task_id,
                instruction=self._active_task.instruction,
                total_time=self._active_task.total_time,
                plan_dict=plan_dict,
            )
            return self._active_task_payload()

    def fail_planning_task(self, task_id: str, message: str) -> None:
        with self._lock:
            if self._active_task is None or self._active_task.task_id != task_id:
                return
            self._finish_active_task_locked("failed", message)

    def create_plan_with_llm(self, instruction: str, llm_planner: Any) -> dict[str, Any]:
        with self._lock:
            self.runner.set_llm_planner(llm_planner)
            snapshot = self.runner.perception.get_scene_snapshot()

        plan_dict = llm_planner.plan(snapshot, instruction)
        if not plan_dict.get("plan"):
            raise ValueError("LLM 未返回可执行 plan")

        with self._lock:
            self.runner.plan_executor.parse(plan_dict, self.runner._get_cart_state()[:2])
        return plan_dict

    def consume_task_events(self) -> list[dict[str, str]]:
        with self._lock:
            events = [
                {
                    "task_id": event.task_id,
                    "status": event.status,
                    "message": event.message,
                }
                for event in self._task_events
            ]
            self._task_events.clear()
            return events

    def step_and_render(self) -> bytes:
        with self._lock:
            if self._state == "running":
                for _ in range(self._physics_steps_per_frame):
                    if self._active_task is not None and self._active_task.status == "running":
                        self._step_runner_task_locked()
                    elif self._active_task is None:
                        mujoco.mj_step(self.model, self.data)
            rgb = self._render_rgb_locked()
            self._frame_index += 1
            return encode_jpeg(rgb, quality=self.config.jpeg_quality)

    def render_current_frame(self) -> bytes:
        with self._lock:
            rgb = self._render_rgb_locked()
            return encode_jpeg(rgb, quality=self.config.jpeg_quality)

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state": self._state,
                "sim_time": round(float(self.data.time), 4),
                "frame_index": self._frame_index,
                "resolution": [self.config.width, self.config.height],
                "fps": self.config.fps,
                "camera": self.camera.to_payload(),
                "available_cameras": self.available_cameras,
                "active_task": self._active_task_payload(),
            }

    def close(self) -> None:
        with self._lock:
            close = getattr(self.renderer, "close", None)
            if callable(close):
                close()
            try:
                import glfw

                glfw.terminate()
            except Exception:
                pass

    def _step_runner_task_locked(self) -> None:
        assert self._active_task is not None
        try:
            still_running = self.runner.step_once()
        except Exception as exc:
            self._finish_active_task_locked("failed", f"任务执行失败：{exc}")
            self._state = "paused"
            return

        if not still_running:
            if self.runner.state == "DONE":
                self._finish_active_task_locked("completed", "任务执行完成")
            elif self.runner.state == "TIMEOUT":
                self._finish_active_task_locked("failed", "任务执行超时")
            else:
                self._finish_active_task_locked("failed", f"任务停止：{self.runner.state}")

    def _finish_active_task_locked(self, status: str, message: str) -> None:
        if self._active_task is None:
            return
        self._active_task.status = status
        self._active_task.message = message
        self._task_events.append(
            _TaskEvent(
                task_id=self._active_task.task_id,
                status=status,
                message=message,
            )
        )
        self._active_task = None

    def _start_runner_locked(
        self,
        *,
        task_id: str,
        instruction: str,
        total_time: float,
        plan_dict: dict[str, Any],
    ) -> None:
        self.runner.start(
            total_time=total_time,
            show_trajectory=False,
            task_instruction=instruction,
            plan_dict=plan_dict,
            llm_planner=None,
        )
        self._active_task = _PipelineTask(
            task_id=task_id,
            instruction=instruction,
            total_time=float(total_time),
            status="running",
            message="任务正在执行",
            plan_dict=plan_dict,
        )
        self._state = "running"

    def _ensure_idle_locked(self) -> None:
        if self._active_task is not None:
            raise ValueError(f"当前已有任务 {self._active_task.task_id} 正在{self._active_task.status}")

    def _active_task_payload(self) -> dict[str, Any] | None:
        if self._active_task is None:
            return None
        phase_count = len(getattr(self.runner, "_phases", []))
        phase_index = getattr(self.runner, "_phase_index", 0)
        return {
            "id": self._active_task.task_id,
            "instruction": self._active_task.instruction,
            "status": self._active_task.status,
            "message": self._active_task.message,
            "progress": round(self.runner.progress(), 4),
            "runner_state": self.runner.state,
            "phase_index": phase_index,
            "phase_count": phase_count,
        }

    def _collect_camera_names(self) -> list[str]:
        names: list[str] = []
        for cam_id in range(self.model.ncam):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_id)
            if name:
                names.append(name)
        return names

    def _calculate_physics_steps_per_frame(self) -> int:
        timestep = float(self.model.opt.timestep)
        if timestep <= 0:
            return 1
        return max(1, int(round(self.config.frame_interval / timestep)))

    def _render_rgb_locked(self) -> np.ndarray:
        if self.camera.mode == "fixed" and self.camera.fixed_camera in self._available_cameras:
            self.renderer.update_scene(self.data, camera=self.camera.fixed_camera)
        else:
            self._free_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            self._free_camera.azimuth = self.camera.azimuth
            self._free_camera.elevation = self.camera.elevation
            self._free_camera.distance = self.camera.distance
            self._free_camera.lookat[:] = self.camera.lookat
            self.renderer.update_scene(self.data, camera=self._free_camera)
        return self.renderer.render().copy()
