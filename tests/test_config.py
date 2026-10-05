import pytest

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
