import pytest

from src.llm_planner.dialog_planner import DialogPlanner


class _FailingCompletions:
    def __init__(self):
        self.calls = 0

    def create(self, **_kwargs):
        self.calls += 1
        raise Exception("Error code: 401 - {'error': {'code': 'invalid_api_key'}}")


class _FakeChat:
    def __init__(self, completions):
        self.completions = completions


class _FakeClient:
    def __init__(self, completions):
        self.chat = _FakeChat(completions)


def test_dialog_planner_does_not_retry_authentication_errors(monkeypatch):
    completions = _FailingCompletions()
    planner = object.__new__(DialogPlanner)
    planner.client = _FakeClient(completions)
    planner.model = "qwen3.5-plus"
    planner.temperature = 0.2
    planner.max_completion_tokens = 128
    planner.verbose = False
    monkeypatch.setattr("src.llm_planner.dialog_planner.time.sleep", lambda _seconds: None)

    with pytest.raises(RuntimeError, match="LLM authentication failed"):
        planner._query_llm("system", "user")

    assert completions.calls == 1
