"""Use the same model configuration for command-line and web tasks."""

import os
from dataclasses import dataclass, field
from typing import Mapping


DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.5-plus"
API_KEY_ENV_NAMES = (
    "STA_LLM_API_KEY",
    "DASHSCOPE_API_KEY",
    "DEEPSEEK_API_KEY",
    "OPENAI_API_KEY",
)


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
    ) -> "LLMSettings":
        env = os.environ if environ is None else environ
        key = api_key or next((env[name] for name in API_KEY_ENV_NAMES if env.get(name)), "")
        if not key:
            raise ValueError(
                "No LLM API key configured. Set STA_LLM_API_KEY before dispatching "
                "natural-language tasks."
            )
        return cls(
            api_key=key,
            base_url=base_url or env.get("STA_LLM_BASE_URL") or DEFAULT_BASE_URL,
            model_name=model_name or env.get("STA_LLM_MODEL") or DEFAULT_MODEL,
        )
