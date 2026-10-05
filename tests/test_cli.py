import json

from src.cli.dialog import main
from src.config.llm import API_KEY_ENV_NAMES


def test_preplanned_cli_runs_without_credentials_or_planning(tmp_path, monkeypatch):
    for name in API_KEY_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    plan = {"reasoning": "离线验证", "plan": [
        {"arm": "L", "skill": "NavSkill", "params": {"target": [-2.99, -0.5]}}
    ]}
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
    calls = {}

    class Runner:
        def __init__(self, **kwargs):
            calls["init"] = kwargs

        def run(self, **kwargs):
            calls["run"] = kwargs

    def fail_if_planning_is_called(_settings):
        raise AssertionError("An offline plan must not call the LLM")

    monkeypatch.setattr("src.pipeline.task_runner.TaskRunner", Runner)
    monkeypatch.setattr("src.llm_planner.factory.create_planner", fail_if_planning_is_called)
    monkeypatch.chdir(tmp_path)
    assert main(["--plan", str(plan_file), "--no-viewer", "--no-trajectory"]) == 0
    assert calls["init"]["plan_dict"] == plan
    assert calls["init"]["llm_planner"] is None
    assert calls["run"]["use_viewer"] is False
    assert calls["run"]["show_trajectory"] is False
