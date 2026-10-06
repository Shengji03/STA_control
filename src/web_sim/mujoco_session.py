from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any
from copy import deepcopy

import mujoco
import numpy as np

from src.pipeline.task_runner import TaskRunner
from src.config.paths import DEFAULT_SCENE_PATH, PROJECT_ROOT

from .camera import CameraCommand, CameraState
from .frame_codec import encode_jpeg
from .trajectory import nominal_preview, telemetry_sample, trajectory_metrics


@dataclass
class _PipelineTask:
    task_id: str
    instruction: str
    total_time: float
    status: str = "planning"
    message: str = "LLM 规划中"
    plan_dict: dict[str, Any] | None = None
    preview: dict | None = None


@dataclass
class _TaskEvent:
    task_id: str
    status: str
    message: str
    optimization: dict = field(default_factory=dict)
    execution_effects: dict = field(default_factory=dict)
    detail: dict = field(default_factory=dict)


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
        if 'cam_global' in self._available_cameras:
            self.camera.apply(CameraCommand(action='set_fixed', camera='cam_global'))
        self._active_task: _PipelineTask | None = None
        self._task_events: list[_TaskEvent] = []
        self._last_task = None
        self._samples = []
        self._scratch = mujoco.MjData(self.model)
        self._last_sample_time = -1.
        self._record_tcp = True
        self._overlay = {'planned': True, 'actual': True, 'L': True, 'R': True}
        self._replay_data = None
        self._replay_index = 0
        self._latest_sample = None

    def configure_task(self, scene, fps=20, record_tcp=True):
        if self._active_task is not None:
            raise ValueError('当前任务尚未结束，请先停止或完成任务')
        if scene not in ('scene5_glare.xml', 'scene4_pipeline.xml'):
            raise ValueError('当前网页支持遮光与管道双臂场景；装配实验使用独立入口')
        if self.config.scene_path.name != scene:
            from src.config.paths import SCENES_DIR
            self.runner.perception._renderer.close()
            self.runner = TaskRunner(str(SCENES_DIR / scene))
            self.model, self.data = self.runner.model, self.runner.data
            self.renderer.close()
            self.renderer = mujoco.Renderer(self.model, height=self.config.height, width=self.config.width)
            self._scratch = mujoco.MjData(self.model)
            self.config.scene_path = SCENES_DIR / scene
            self._available_cameras = self._collect_camera_names()
            self.camera.apply(CameraCommand(action='set_fixed', camera='cam_global' if 'cam_global' in self._available_cameras else 'cam_overview'))
        self.config.fps = max(5, min(int(fps), 30))
        self._physics_steps_per_frame = self._calculate_physics_steps_per_frame()
        self._record_tcp = bool(record_tcp)
        self._replay_data = None
        self._last_task = None
        self._latest_sample = None

    def set_overlay(self, **values):
        for key in self._overlay:
            if key in values: self._overlay[key] = bool(values[key])

    def planning_input(self):
        with self._lock:
            data = mujoco.MjData(self.model)
            data.qpos[:] = self.data.qpos
            data.qvel[:] = self.data.qvel
            data.ctrl[:] = self.data.ctrl
            mujoco.mj_forward(self.model, data)
            return self.model, data, self.runner.perception.get_scene_snapshot(include_images=False)

    def prepare_preplanned(self, plan, instruction):
        return self.runner.prepare_plan(plan, instruction=instruction)

    def detail(self):
        if self._replay_data is not None:
            return deepcopy(self._replay_data)
        task = self._active_task or self._last_task
        return {'task_id': task.task_id if task else None, 'scene': self.config.scene_path.name,
                'plan': task.plan_dict if task else None, 'preview': task.preview if task else None,
                'samples': deepcopy(self._samples), 'optimization': self.runner.planning_report,
                'execution_effects': self.runner.execution_report, 'metrics': trajectory_metrics(self._samples)}

    def replay(self, detail, index):
        if self._active_task:
            raise ValueError('运行或规划期间不能切换到历史回放')
        if detail['scene'] != self.config.scene_path.name:
            self.configure_task(detail['scene'], self.config.fps, True)
        samples = detail.get('samples', [])
        if not samples: raise ValueError('该任务没有记录可回放的仿真数据')
        index = max(0, min(int(index), len(samples) - 1))
        row = samples[index]
        self.data.qpos[:] = row['qpos']
        self.data.time = row['t']
        mujoco.mj_forward(self.model, self.data)
        self._replay_data = detail
        self._replay_index = index
        self._state = 'replay'
        return {'index': index, 'count': len(samples), 'time': row['t']}

    def stop(self):
        if self._active_task:
            self.runner._is_started = False
            self._finish_active_task_locked('cancelled', '任务已停止')
        self._state = 'paused'

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
            self.stop()
            self.runner.reset_scene()
            self._frame_index = 0
            self._state = "running"
            self._active_task = None
            self._last_task = None
            self._samples = []
            self._replay_data = None
            self._latest_sample = None
            self._last_sample_time = -1.

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
            self._samples = []
            self._last_sample_time = -1.
            self._latest_sample = None
            self._last_task = None
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

    def activate_planned_task(self, task_id: str, plan_dict: dict[str, Any], execute=True) -> dict[str, Any]:
        with self._lock:
            if self._active_task is None or self._active_task.task_id != task_id:
                raise ValueError(f"任务 {task_id} 已不再处于可启动状态")
            if self._active_task.status != "planning":
                raise ValueError(f"任务 {task_id} 当前状态为 {self._active_task.status}")
            self._active_task.plan_dict = plan_dict
            self._active_task.preview = nominal_preview(self.runner, plan_dict)
            self.runner.prepared_plan = plan_dict
            self.runner.planning_report = plan_dict.get('optimization', {})
            self._active_task.status = 'planned'
            self._active_task.message = '规划完成，等待执行'
            self._state = 'paused'
            if not execute:
                return self._active_task_payload()
            return self.execute_planned(task_id)

    def execute_planned(self, task_id):
        with self._lock:
            if not self._active_task or self._active_task.task_id != task_id or self._active_task.status != 'planned':
                raise ValueError('任务已失效或不处于待执行状态')
            preview = self._active_task.preview
            self._start_runner_locked(
                task_id=task_id,
                instruction=self._active_task.instruction,
                total_time=self._active_task.total_time,
                plan_dict=self._active_task.plan_dict,
            )
            self._active_task.preview = preview
            return self._active_task_payload()

    def fail_planning_task(self, task_id: str, message: str) -> None:
        with self._lock:
            if self._active_task is None or self._active_task.task_id != task_id:
                return
            self._finish_active_task_locked("failed", message)

    def create_plan_with_llm(self, instruction: str, llm_planner: Any) -> dict[str, Any]:
        return self.plan_from_input(self.planning_input(), instruction, llm_planner)

    @staticmethod
    def plan_from_input(planning_input, instruction, llm_planner):
        from src.pipeline.task_runner import ARM_L_OFFSET, ARM_R_OFFSET, ARM_L_YAW, ARM_R_YAW
        from src.task_planning.planning_service import PlanningService
        model, data, snapshot = planning_input
        offsets = {'L': ARM_L_OFFSET, 'R': ARM_R_OFFSET}
        yaws = {'L': ARM_L_YAW, 'R': ARM_R_YAW}
        if hasattr(llm_planner, 'attach_env'):
            llm_planner.attach_env(model, data, offsets, yaws)
        service = PlanningService(model, data, offsets, yaws)
        cart = snapshot['planner_state']['cart']
        pose = [*cart['position'][:2], cart.get('yaw', 0.)]

        feedback = ''
        for attempt in range(1 + TaskRunner._MAX_PLAN_RETRIES):
            plan_dict = llm_planner.plan(snapshot, instruction, extra_context=feedback)
            try:
                return service.prepare(plan_dict, snapshot, instruction, pose)
            except ValueError as exc:
                if attempt == TaskRunner._MAX_PLAN_RETRIES:
                    raise
                feedback = f'上一次完整分工与辅助安排校验失败: {exc}。请修正 goals/stages 和导航位置。'

    def consume_task_events(self) -> list[dict[str, Any]]:
        with self._lock:
            events = [
                {
                    "task_id": event.task_id,
                    "status": event.status,
                    "message": event.message,
                    "optimization": event.optimization,
                    "execution_effects": event.execution_effects,
                    'detail': event.detail,
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
                        # Idle scenes keep their initialized posture. Uncontrolled
                        # gravity stepping used to make the standby robot collapse.
                        pass
            rgb = self._render_rgb_locked()
            self._frame_index += 1
            return encode_jpeg(rgb, quality=self.config.jpeg_quality)

    def render_current_frame(self) -> bytes:
        with self._lock:
            rgb = self._render_rgb_locked()
            return encode_jpeg(rgb, quality=self.config.jpeg_quality)

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            sample = (self._replay_data['samples'][self._replay_index]
                      if self._replay_data else self._latest_sample)
            replay_task = None
            if self._replay_data and self._replay_data.get('task'):
                meta = self._replay_data['task']
                replay_task = {'id': meta['id'], 'instruction': meta['instruction'], 'status': 'replay',
                               'message': '历史仿真状态回放', 'progress': self._replay_index / max(1, len(self._replay_data['samples']) - 1),
                               'phase_index': sample['phase'], 'phase_count': len(self._replay_data.get('plan', {}).get('execution_phases', [])),
                               'runner_state': 'REPLAY', 'optimization': meta.get('optimization')}
            return {
                "state": self._state,
                "sim_time": round(float(self.data.time), 4),
                "frame_index": self._frame_index,
                "resolution": [self.config.width, self.config.height],
                "fps": self.config.fps,
                "camera": self.camera.to_payload(),
                "available_cameras": self.available_cameras,
                "active_task": self._active_task_payload(),
                'last_task': replay_task or (self._task_payload(self._last_task) if self._last_task else None),
                'scene': self.config.scene_path.name,
                'telemetry': {key: value for key, value in sample.items() if key != 'qpos'} if sample else None,
                'overlay': dict(self._overlay),
                'replay': self._replay_data is not None,
            }

    def close(self) -> None:
        with self._lock:
            close = getattr(self.renderer, "close", None)
            if callable(close):
                close()
            if self.runner.perception._renderer is not None:
                self.runner.perception._renderer.close()

    def _step_runner_task_locked(self) -> None:
        assert self._active_task is not None
        try:
            still_running = self.runner.step_once()
            if self.data.time - self._last_sample_time >= .049:
                self._latest_sample = telemetry_sample(self.runner, self._scratch)
                if self._record_tcp:
                    self._samples.append(self._latest_sample)
                self._last_sample_time = float(self.data.time)
        except Exception as exc:
            self._finish_active_task_locked("failed", f"任务执行失败：{exc}")
            self._state = "paused"
            return

        if not still_running:
            if self.runner.state == "DONE":
                self._finish_active_task_locked("completed", "任务执行完成")
            elif self.runner.state == "TIMEOUT":
                self._finish_active_task_locked("failed", "任务执行超时")
            elif self.runner.state == 'FAILED':
                self._finish_active_task_locked('failed', '阀门角度或必需遮光效果未通过传感器校验')
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
                optimization=self.runner.planning_report,
                execution_effects=self.runner.execution_report,
                detail=self.detail(),
            )
        )
        self._last_task = self._active_task
        self._active_task = None
        self._state = 'paused'

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
        self._samples = []
        self._last_sample_time = -1.
        self._latest_sample = None
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
        return self._task_payload(self._active_task)

    def _task_payload(self, task):
        phase_count = len(getattr(self.runner, "_phases", []))
        phase_index = getattr(self.runner, "_phase_index", 0)
        return {
            "id": task.task_id,
            "instruction": task.instruction,
            "status": ('paused' if task.status == 'running' and self._state == 'paused' else task.status),
            "message": task.message,
            "progress": round(self._task_progress(task), 4),
            "runner_state": self.runner.state,
            "phase_index": phase_index,
            "phase_count": phase_count,
            "optimization": {key: value for key, value in self.runner.planning_report.items()
                             if key in ('status', 'objective', 'main_score', 'auxiliary_score',
                                        'algorithm', 'auxiliary_method', 'elapsed_ms', 'goal_count',
                                        'covered_goal_count', 'assignments', 'auxiliaries')},
            "execution_effects": self.runner.execution_report,
        }

    def _task_progress(self, task):
        if task.status == 'completed': return 1.
        if task.status in ('planning', 'planned'): return 0.
        count = len(self.runner._phases)
        if not count: return 0.
        stage = self.runner._phase_index
        if stage >= count: return 1.
        if self.runner.state == 'PHASE_NAV':
            nav = self.runner._navigation
            partial = .15 * min(max((self.data.time - nav.start_time) / nav.duration, 0), 1) if nav else 0
        else:
            fractions = [(max(0, arm.executor.current_index) / max(len(arm.executor.skills), 1))
                         for arm in self.runner.arms.values() if arm.executor]
            fraction = min(fractions, default=0.)
            if self.runner._L_waiting_for_R: fraction *= .25
            elif self.runner._R_cleanup_pending: fraction = .25 + .5 * fraction
            elif self.runner._phases[stage]['mode'] == 'assist_R_then_L': fraction = .75 + .25 * fraction
            partial = .15 + .85 * fraction
        return min(.99, (stage + partial) / count)

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
        detail = self._replay_data
        task = self._active_task or self._last_task
        preview = detail.get('preview') if detail else task.preview if task else None
        samples = detail.get('samples', []) if detail else self._samples
        if detail:
            samples = samples[:self._replay_index + 1]
        if preview and self._overlay['planned']:
            self._draw_paths(preview['paths'], True)
        if samples and self._overlay['actual']:
            self._draw_paths({arm: [s['arms'][arm]['actual_tcp'] for s in samples] for arm in ('L', 'R')}, False)
        return self.renderer.render().copy()

    def _draw_paths(self, paths, planned):
        from src.visualization.trajectories import draw_trajectory
        colors = {'L': [0.22, .65, 1., .8], 'R': [1., .62, .18, .8]}
        for arm, points in paths.items():
            if not self._overlay[arm] or len(points) < 2: continue
            points = np.asarray(points, dtype=float)
            stride = max(1, len(points) // 250)
            points = points[::stride]
            # Stationary references and gripper holds generate repeated points.
            # Preserve turns, but avoid hundreds of overlapping marker geoms.
            selected = [points[0]]
            for point in points[1:]:
                if np.linalg.norm(point - selected[-1]) >= .004:
                    selected.append(point)
            if len(selected) < 2: continue
            points = np.asarray(selected)
            if planned:
                # Separate short segments form a dashed planned path.
                for i in range(0, len(points) - 1, 4):
                    draw_trajectory(self.renderer.scene, points[i:i + 2], colors[arm], .003)
            else:
                draw_trajectory(self.renderer.scene, points, colors[arm], .004)
