import pytest

import src.config.llm as llm_config

from src.config.llm import LLMSettings
from src.config.paths import DEFAULT_SCENE_PATH, SCENES_DIR, scene_path
from src.cli.dialog import parse_args


def test_explicit_llm_settings_override_shared_environment():
    settings = LLMSettings.from_env(
        api_key="explicit", base_url="https://explicit.example/v1", model_name="explicit-model",
        environ={"STA_LLM_API_KEY": "environment", "STA_LLM_MODEL": "environment-model"},
    )
    assert settings.api_key == "explicit"
    assert settings.base_url == "https://explicit.example/v1"
    assert settings.model_name == "explicit-model"
    assert "explicit" not in repr(settings).split("base_url=")[0]


def test_cli_and_web_share_provider_key_priority():
    settings = LLMSettings.from_env(environ={
        "DASHSCOPE_API_KEY": "provider", "OPENAI_API_KEY": "fallback",
        "STA_LLM_BASE_URL": "https://provider.example/v1", "STA_LLM_MODEL": "provider-model",
    })
    assert settings.api_key == "provider"
    assert settings.model_name == "provider-model"
    assert settings.base_url == "https://provider.example/v1"


def test_missing_llm_key_has_actionable_error():
    with pytest.raises(ValueError, match="STA_LLM_API_KEY"):
        LLMSettings.from_env(environ={})


def test_scene_defaults_do_not_depend_on_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = parse_args(["--task", "test"])
    assert args.scene == DEFAULT_SCENE_PATH
    assert args.scene.is_file()
    assert args.api_key is None
    assert scene_path("scene3") == SCENES_DIR / "scene3.xml"


@pytest.mark.parametrize("name", ["../scene3", "folder/scene3", "folder\\scene3"])
def test_bundled_scene_names_cannot_escape_assets(name):
    with pytest.raises(ValueError, match="filename"):
        scene_path(name)


def test_project_env_loads_independently_of_working_directory(tmp_path, monkeypatch):
    env_file = tmp_path / "project.env"
    env_file.write_text(
        "STA_LLM_API_KEY=local-test-key\nSTA_LLM_MODEL=qwen3.8-flash\n",
        encoding="utf-8",
    )
    for name in llm_config.API_KEY_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("STA_ENV_FILE", raising=False)
    monkeypatch.setattr(llm_config, "DEFAULT_ENV_FILE", env_file)
    other_directory = tmp_path / "other"
    other_directory.mkdir()
    monkeypatch.chdir(other_directory)
    settings = LLMSettings.from_env()
    assert settings.api_key == "local-test-key"
    assert settings.model_name == "qwen3.8-flash"
    assert "STA_LLM_API_KEY" not in llm_config.os.environ


def test_process_environment_overrides_project_env(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("STA_LLM_API_KEY=file-key\nSTA_LLM_MODEL=file-model\n", encoding="utf-8")
    settings = LLMSettings.from_env(
        environ={"STA_LLM_API_KEY": "process-key", "STA_LLM_MODEL": "process-model"},
        env_file=env_file,
    )
    assert settings.api_key == "process-key"
    assert settings.model_name == "process-model"


@pytest.mark.parametrize("name", ["DEEPSEEK_API_KEY", "OPENAI_API_KEY"])
def test_qwen_endpoint_does_not_use_another_provider_key(name):
    with pytest.raises(ValueError, match="STA_LLM_API_KEY"):
        LLMSettings.from_env(environ={name: "another-provider-key"})


def test_provider_specific_key_is_used_with_its_own_endpoint():
    settings = LLMSettings.from_env(environ={
        "DEEPSEEK_API_KEY": "deepseek-test-key",
        "STA_LLM_BASE_URL": "https://api.deepseek.com/v1",
        "STA_LLM_MODEL": "deepseek-chat",
    })
    assert settings.api_key == "deepseek-test-key"
    assert settings.model_name == "deepseek-chat"
