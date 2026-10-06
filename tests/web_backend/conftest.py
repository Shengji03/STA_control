"""API tests never write their records into the user's live experiment history."""
from pathlib import Path
import pytest
from src.config.paths import OUTPUTS_DIR
from src.web_backend.storage import TaskRepository


@pytest.fixture(autouse=True)
def isolate_default_task_storage(monkeypatch, tmp_path):
    def repository(directory):
        if Path(directory).resolve() == (OUTPUTS_DIR / 'web').resolve():
            directory = tmp_path / 'web-records'
        return TaskRepository(directory)
    monkeypatch.setattr('src.web_backend.app.TaskRepository', repository)
