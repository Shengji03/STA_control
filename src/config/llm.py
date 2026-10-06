"""Use the same model configuration for command-line and web tasks."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping
from urllib.parse import urlsplit

from dotenv import dotenv_values

from .paths import PROJECT_ROOT


DEFAULT_BASE_URL = "https://maas.qianwenaiapi.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.8-flash"
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"
API_KEY_ENV_NAMES = (
    "STA_LLM_API_KEY",
    "DASHSCOPE_API_KEY",
    "DEEPSEEK_API_KEY",
    "OPENAI_API_KEY",
)


def _provider_key_names(base_url: str) -> tuple[str, ...]:
    host = (urlsplit(base_url).hostname or "").lower()
    if host in {"dashscope.aliyuncs.com", "dashscope-intl.aliyuncs.com"} or host.endswith(
        (".qianwenaiapi.com", ".qwencloudapi.com", ".maas.aliyuncs.com", ".dashscope.aliyuncs.com")
    ):
        return ("STA_LLM_API_KEY", "DASHSCOPE_API_KEY")
    if host == "api.deepseek.com":
        return ("STA_LLM_API_KEY", "DEEPSEEK_API_KEY")
    if host == "api.openai.com":
        return ("STA_LLM_API_KEY", "OPENAI_API_KEY")
    return API_KEY_ENV_NAMES


@dataclass(frozen=True)
class LLMSettings:
    api_key: str = field(repr=False)
    base_url: str = DEFAULT_BASE_URL
    model_name: str = DEFAULT_MODEL

    @classmethod
    def from_env(
        cls,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model_name: str | None = None,
        environ: Mapping[str, str] | None = None,
        env_file: str | Path | None = None,
    ) -> "LLMSettings":
        process_env = dict(os.environ if environ is None else environ)
        file_env = {}
        if environ is None or env_file is not None:
            path = Path(env_file) if env_file is not None else Path(
                process_env.get("STA_ENV_FILE") or DEFAULT_ENV_FILE
            )
            if path.is_file():
                file_env = {
                    name: value for name, value in dotenv_values(path, encoding="utf-8").items()
                    if value is not None
                }
        env = {**file_env, **process_env}
        resolved_url = base_url or env.get("STA_LLM_BASE_URL") or DEFAULT_BASE_URL
        names = _provider_key_names(resolved_url)
        key = api_key or next((env[name] for name in names if env.get(name)), "")
        if not key:
            raise ValueError(
                "No LLM API key configured. Set STA_LLM_API_KEY in the environment "
                "or project .env before dispatching natural-language tasks."
            )
        return cls(
            api_key=key,
            base_url=resolved_url,
            model_name=model_name or env.get("STA_LLM_MODEL") or DEFAULT_MODEL,
        )
