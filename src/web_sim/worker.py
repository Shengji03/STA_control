"""One owner thread for MuJoCo, OpenGL and the simulation clock.

HTTP commands use a queue. WebSockets consume cached frames and never step the
robot. All renderers are created, used and closed on this same thread.
"""

from concurrent.futures import Future
from copy import deepcopy
from queue import Empty, Queue
from threading import Event, RLock, Thread
from time import monotonic

from .mujoco_session import MujocoStreamSession


class SimulationWorker:
    def __init__(self, config, on_event):
        self.config, self.on_event = config, on_event
        self._commands = Queue()
        self._stop = Event()
        self._ready = Future()
        self._lock = RLock()
        self._frame = b''
        self._status = {}
        self._error = None
        self.session = None
        self.thread = Thread(target=self._run, name='sta-simulation-owner', daemon=True)

    def start(self):
        self.thread.start()
        self._ready.result(timeout=45)

    def call(self, method, *args, **kwargs):
        if self._error:
            raise RuntimeError(f'仿真线程异常: {self._error}')
        if not self.thread.is_alive():
            raise RuntimeError('仿真尚未启动或已经关闭')
        future = Future()
        self._commands.put((method, args, kwargs, future))
        return future.result(timeout=45)

    def snapshot(self):
        with self._lock:
            return self._frame, deepcopy(self._status)

    def close(self):
        self._stop.set()
        self.thread.join(timeout=15)

    def _publish(self, frame):
        with self._lock:
            self._frame = frame
            self._status = self.session.get_status()
        for event in self.session.consume_task_events():
            self.on_event(event)

    def _run(self):
        try:
            self.session = MujocoStreamSession(self.config)
            self._publish(self.session.render_current_frame())
            self._ready.set_result(True)
            deadline = monotonic()
            while not self._stop.is_set():
                timeout = max(0., deadline - monotonic())
                try:
                    method, args, kwargs, future = self._commands.get(timeout=min(timeout, .05))
                    try:
                        result = getattr(self.session, method)(*args, **kwargs)
                        self._publish(self.session.render_current_frame())
                        future.set_result(result)
                    except Exception as exc:
                        future.set_exception(exc)
                except Empty:
                    pass
                if monotonic() >= deadline:
                    frame_started = monotonic()
                    self._publish(self.session.step_and_render())
                    # Fixed physics dt; render load slows wall time rather than
                    # adding duplicate steps for each connected viewer.
                    deadline = max(frame_started + self.session.frame_interval, monotonic())
        except Exception as exc:
            self._error = str(exc)
            if not self._ready.done(): self._ready.set_exception(exc)
            with self._lock:
                self._status.update(state='error', message=str(exc))
        finally:
            if self.session:
                self.session.close()
