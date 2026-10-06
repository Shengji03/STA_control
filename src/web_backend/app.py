from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
from dataclasses import replace
from datetime import datetime
import json
from math import pi
from pathlib import Path
from threading import RLock, Thread
from concurrent.futures import TimeoutError as FutureTimeoutError
from uuid import uuid4

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from src.config.llm import LLMSettings
from src.llm_planner.factory import create_planner
from src.web_sim.mujoco_session import MujocoStreamSession, SimulationConfig
from src.web_sim.worker import SimulationWorker
from .config import BackendSettings
from .schemas import (CameraRequest, ControlRequest, HealthResponse, LogRecord, OverlayRequest,
                      ReplayRequest, SimulationStatus, TaskDispatchRequest, TaskRecord)
from .storage import TaskRepository


def scenarios():
    def goal(gid, obj, shade='none'):
        return {'id': gid, 'object': obj, 'operation': 'rotate', 'angle': pi,
                'shade': shade, 'preferred_arm': 'L'}
    return [
        {'id': 'shade', 'name': '遮光协作', 'description': 'R 臂持板，L 臂旋拧，操作完成后放回',
         'instruction': '遮光并旋拧1号阀门180度', 'scene': 'scene5_glare.xml',
         'plan': {'goals': [goal('v1', 'valve_1', 'required')],
                  'stages': [{'nav': {'target': [.45, 0]}, 'goals': ['v1']}]}},
        {'id': 'valve', 'name': '单阀门操作', 'description': '自动选择主操作臂，旋转 180°',
         'instruction': '旋拧1号阀门180度', 'scene': 'scene5_glare.xml',
         'plan': {'goals': [goal('v1', 'valve_1')],
                  'stages': [{'nav': {'target': [.45, 0]}, 'goals': ['v1']}]}},
        {'id': 'two-valves', 'name': '双目标作业', 'description': '保留两个必做目标，依次导航与操作',
         'instruction': '依次旋拧1号和2号阀门180度', 'scene': 'scene5_glare.xml',
         'plan': {'goals': [goal('v1', 'valve_1'), goal('v2', 'valve_2')],
                  'stages': [{'nav': {'target': [.45, 0]}, 'goals': ['v1']},
                             {'nav': {'target': [3.5, 0]}, 'goals': ['v2']}]}},
    ]


class SimulationRuntime:
    def __init__(self, settings, enabled):
        self.settings, self.enabled = settings, enabled
        self.worker = None
        self.session = None
        self._lock = RLock()
        self._dispatch_lock = RLock()
        self._replay_cache = None
        self.repository = TaskRepository(settings.data_dir)
        self._tasks = [TaskRecord(**item) for item in self.repository.tasks()]
        for index, task in enumerate(self._tasks):
            if task.status in ('planning', 'planned', 'running', 'paused'):
                self._tasks[index] = task.model_copy(update={'status': 'interrupted', 'message': '服务重启，原任务未继续执行'})
                self.repository.save(self._tasks[index].model_dump())
        self._logs = []
        self._add_log('INFO', '服务就绪，仿真由独立线程驱动')

    def start(self):
        if self.enabled:
            config = SimulationConfig(scene_path=self.settings.scene_path, width=self.settings.width,
                                      height=self.settings.height, fps=self.settings.fps,
                                      jpeg_quality=self.settings.jpeg_quality)
            self.worker = SimulationWorker(config, self._receive_event)
            self.worker.start()
            self.session = self.worker.session
            self._add_log('INFO', 'MuJoCo 画面与任务运行已就绪')

    def _call(self, method, *args, **kwargs):
        if not self.worker: raise HTTPException(409, 'MuJoCo 仿真会话未启用')
        try: return self.worker.call(method, *args, **kwargs)
        except ValueError as exc: raise HTTPException(409, str(exc)) from exc
        except (RuntimeError, FutureTimeoutError) as exc:
            raise HTTPException(503, str(exc) or '仿真服务响应超时，请检查运行日志') from exc

    def status(self):
        if not self.worker: return SimulationStatus(state='disabled')
        return SimulationStatus(**self.worker.snapshot()[1])

    def control(self, request):
        if not self.worker: return self.status()
        with self._dispatch_lock:
            self._call({'pause': 'pause', 'resume': 'resume', 'reset': 'reset', 'stop': 'stop'}[request.action])
            status = self.status()
            if status.active_task and status.active_task['status'] in ('running', 'paused'):
                self._update_task(status.active_task['id'], status=status.active_task['status'])
            return status

    def camera(self, request):
        if self.worker: self._call('apply_camera_command', request.model_dump())
        return self.status()

    def dispatch_task(self, request):
        if not self.worker: raise HTTPException(409, 'Simulation session is disabled')
        instruction = request.instruction.strip()
        if not instruction and request.plan is None: raise HTTPException(400, 'Task instruction or plan is required')
        instruction = instruction or '执行预置计划'
        task_id = 'LAB-' + datetime.now().strftime('%Y%m%d-%H%M%S-') + uuid4().hex[:6]
        # Reserve on the single simulation owner before storing/starting a job.
        with self._dispatch_lock:
            self._call('configure_task', request.scene, request.fps, request.record_tcp)
            self._call('start_planning_task', task_id=task_id, instruction=instruction, total_time=request.total_time)
            task = TaskRecord(id=task_id, instruction=instruction, scene=request.scene, mode=request.mode,
                              status='planning', created_at=self._now(), message='正在生成并校验任务计划')
            with self._lock:
                self._tasks.insert(0, task)
                self.repository.save(task.model_dump())
            planning_input = self._call('planning_input')
            self._replay_cache = None
        self._add_log('INFO', f'任务 {task_id} 已进入 LLM 规划队列' if request.plan is None else f'任务 {task_id} 开始预置计划校验')
        Thread(target=self._plan, args=(task_id, request, planning_input), name='sta-task-planner', daemon=True).start()
        return task

    def _plan(self, task_id, request, planning_input):
        try:
            if request.plan is None:
                plan = MujocoStreamSession.plan_from_input(planning_input, request.instruction, self._build_llm_planner())
            else:
                class FixedPlanner:
                    def plan(self, *_args, **_kwargs): return request.plan
                plan = MujocoStreamSession.plan_from_input(planning_input, request.instruction, FixedPlanner())
            # The task ID check rejects stale planning results after stop/reset.
            status = self._call('activate_planned_task', task_id, plan, execute=request.mode == '实时仿真')
            detail = self._call('detail')
            self._update_task(task_id, status=status['status'], message=status['message'], optimization=plan.get('optimization'))
            if detail.get('task_id') == task_id:
                self.repository.save(self._find(task_id).model_dump(), detail)
            self._add_log('INFO', f'任务 {task_id} 分工和可执行性校验通过')
        except Exception as exc:
            task = self._find(task_id)
            if task and task.status not in ('cancelled', 'completed', 'interrupted'):
                message = 'LLM 规划失败：' + (str(exc.detail) if isinstance(exc, HTTPException) else str(exc))
                try: self._call('fail_planning_task', task_id, message)
                except Exception: pass
                self._update_task(task_id, status='failed', message=message)
                self._add_log('ERROR', f'任务 {task_id} {message}')

    def execute(self, task_id):
        with self._dispatch_lock:
            payload = self._call('execute_planned', task_id)
            self._update_task(task_id, status='running', mode='实时仿真', message='任务正在执行')
            return payload

    def _receive_event(self, event):
        detail = event.get('detail', {})
        self._update_task(event['task_id'], status=event['status'], message=event['message'],
                          optimization=event.get('optimization'), execution_effects=event.get('execution_effects'),
                          metrics=detail.get('metrics'), sample_count=len(detail.get('samples', [])), completed_at=self._now())
        task = self._find(event['task_id'])
        if task: self.repository.save(task.model_dump(), detail)
        self._add_log('INFO' if event['status'] == 'completed' else 'WARN', f"{event['task_id']} {event['message']}")

    def _find(self, task_id):
        return next((task for task in self._tasks if task.id == task_id), None)

    def detail(self, task_id):
        task = self._find(task_id)
        if not task: raise HTTPException(404, '任务不存在')
        status = self.status()
        current = status.active_task or status.last_task
        detail = self._call('detail') if current and current['id'] == task_id else self.repository.detail(task_id)
        return {'task': task.model_dump(), **(detail or {'samples': [], 'preview': None, 'plan': None})}

    def task_history(self):
        with self._lock: return list(self._tasks)

    def replay(self, task_id, index):
        if self.status().active_task:
            raise HTTPException(409, '运行或规划期间不能回放历史任务')
        if not self._replay_cache or self._replay_cache['task']['id'] != task_id:
            self._replay_cache = self.detail(task_id)
        return self._call('replay', self._replay_cache, index)

    def logs(self):
        with self._lock: return list(self._logs)

    def _update_task(self, task_id, **updates):
        with self._lock:
            for index, task in enumerate(self._tasks):
                if task.id == task_id:
                    if task.status in ('completed', 'failed', 'cancelled', 'interrupted') and updates.get('status') in ('planning', 'planned', 'running', 'paused'):
                        return False
                    updates = {key: value for key, value in updates.items() if value is not None}
                    self._tasks[index] = task.model_copy(update=updates)
                    self.repository.save(self._tasks[index].model_dump())
                    return True
        return False

    def _add_log(self, level, message):
        with self._lock:
            self._logs.insert(0, LogRecord(id=uuid4().hex[:10], timestamp=self._now(), level=level, message=message))
            del self._logs[500:]

    def _build_llm_planner(self): return create_planner(LLMSettings.from_env())

    @staticmethod
    def _now(): return datetime.now().astimezone().isoformat(timespec='seconds')

    def close(self):
        if self.worker:
            try:
                self._call('stop')
            except HTTPException:
                pass
            finally:
                self.worker.close()


def create_app(*, enable_simulation=True, settings=None, frontend_dist=None):
    settings = settings or BackendSettings()
    if frontend_dist is not None: settings = replace(settings, frontend_dist=Path(frontend_dist))
    runtime = SimulationRuntime(settings, enable_simulation)

    @asynccontextmanager
    async def lifespan(_app):
        await asyncio.to_thread(runtime.start)
        try: yield
        finally: await asyncio.to_thread(runtime.close)

    app = FastAPI(title='STA Control', version='0.3.0', lifespan=lifespan)
    app.state.simulation_runtime = runtime

    @app.get('/api/health', response_model=HealthResponse)
    def health(): return HealthResponse(status='ok')

    @app.get('/api/scenarios')
    def presets(): return scenarios()

    @app.get('/api/simulation/status', response_model=SimulationStatus)
    def status(): return runtime.status()

    @app.post('/api/simulation/control', response_model=SimulationStatus)
    def control(request: ControlRequest): return runtime.control(request)

    @app.post('/api/simulation/camera', response_model=SimulationStatus)
    def camera(request: CameraRequest): return runtime.camera(request)

    @app.post('/api/simulation/overlay')
    def overlay(request: OverlayRequest):
        runtime._call('set_overlay', **request.model_dump())
        return runtime.status()

    @app.post('/api/tasks/dispatch', response_model=TaskRecord)
    def dispatch(request: TaskDispatchRequest): return runtime.dispatch_task(request)

    @app.get('/api/tasks/history', response_model=list[TaskRecord])
    def history(): return runtime.task_history()

    @app.get('/api/tasks/{task_id}/detail')
    def detail(task_id: str): return runtime.detail(task_id)

    @app.post('/api/tasks/{task_id}/execute')
    def execute(task_id: str): return runtime.execute(task_id)

    @app.post('/api/tasks/{task_id}/replay')
    def replay(task_id: str, request: ReplayRequest):
        return runtime.replay(task_id, request.index)

    @app.get('/api/tasks/{task_id}/export')
    def export(task_id: str):
        content = json.dumps(runtime.detail(task_id), ensure_ascii=False)
        return Response(content, media_type='application/json',
                        headers={'Content-Disposition': f'attachment; filename="{task_id}.json"'})

    @app.get('/api/logs', response_model=list[LogRecord])
    def logs(): return runtime.logs()

    @app.websocket('/ws/simulation')
    async def stream(websocket: WebSocket):
        await websocket.accept()
        if not runtime.worker:
            await websocket.send_json({'type': 'status', 'payload': runtime.status().model_dump()})
            await websocket.close()
            return
        lock = asyncio.Lock()
        async def send(payload):
            async with lock: await websocket.send_json(payload)
        async def receive():
            while True:
                payload = await websocket.receive_json()
                try:
                    if payload.get('type') == 'control':
                        value = await asyncio.to_thread(runtime.control, ControlRequest(action=payload.get('action')))
                    elif payload.get('type') == 'camera':
                        value = await asyncio.to_thread(runtime.camera, CameraRequest(**payload))
                    else: raise ValueError('不支持的操作')
                    await send({'type': 'status', 'payload': value.model_dump()})
                except Exception as exc: await send({'type': 'error', 'message': str(exc)})
        receiver = asyncio.create_task(receive())
        last_frame = runtime.status().frame_index
        try:
            await send({'type': 'status', 'payload': runtime.status().model_dump()})
            while True:
                frame, snapshot = runtime.worker.snapshot()
                if snapshot.get('frame_index') != last_frame:
                    async with lock:
                        await websocket.send_bytes(frame)
                        await websocket.send_json({'type': 'status', 'payload': snapshot})
                    last_frame = snapshot.get('frame_index')
                await asyncio.sleep(.025)
        except (WebSocketDisconnect, asyncio.CancelledError, RuntimeError): pass
        finally:
            receiver.cancel()
            with suppress(asyncio.CancelledError, WebSocketDisconnect): await receiver

    _mount_frontend(app, settings.frontend_dist)
    return app


def _mount_frontend(app, directory):
    dist = Path(directory).resolve()
    if (dist / 'assets').is_dir(): app.mount('/assets', StaticFiles(directory=dist / 'assets'), name='frontend-assets')
    if not (dist / 'index.html').is_file(): return
    @app.get('/')
    @app.get('/{frontend_path:path}')
    def frontend(frontend_path=''):
        if frontend_path.startswith(('api/', 'ws/')):
            raise HTTPException(404, '接口不存在')
        path = (dist / frontend_path).resolve()
        if path.is_relative_to(dist) and path.is_file(): return FileResponse(path)
        return FileResponse(dist / 'index.html')
