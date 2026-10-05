"""Create planners from shared settings without coupling configuration to runtime."""

from src.config.llm import LLMSettings

from .dialog_planner import DialogPlanner


def create_planner(settings: LLMSettings) -> DialogPlanner:
    return DialogPlanner(
        api_key=settings.api_key,
        base_url=settings.base_url,
        model_name=settings.model_name,
    )
