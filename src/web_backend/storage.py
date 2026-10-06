"""Atomic local experiment records; generated data stays in ignored outputs."""

import gzip
import json
from pathlib import Path
from threading import RLock


class TaskRepository:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()

    def _path(self, task_id, suffix):
        if not task_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in task_id):
            raise ValueError('无效任务编号')
        return self.directory / (task_id + suffix)

    def save(self, task, detail=None):
        with self._lock:
            path = self._path(task['id'], '.json')
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(task, ensure_ascii=False), encoding='utf-8')
            temporary.replace(path)
            if detail is not None:
                path = self._path(task['id'], '.data.json.gz')
                temporary = path.with_suffix('.tmp')
                with gzip.open(temporary, 'wt', encoding='utf-8') as stream:
                    json.dump(detail, stream, ensure_ascii=False)
                temporary.replace(path)

    def tasks(self):
        with self._lock:
            tasks = []
            for path in self.directory.glob('*.json'):
                try:
                    tasks.append(json.loads(path.read_text(encoding='utf-8')))
                except (OSError, ValueError):
                    continue
            return sorted(tasks, key=lambda item: item.get('created_at', ''), reverse=True)

    def detail(self, task_id):
        path = self._path(task_id, '.data.json.gz')
        if not path.exists():
            return None
        with self._lock, gzip.open(path, 'rt', encoding='utf-8') as stream:
            return json.load(stream)
