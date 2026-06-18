from dataclasses import dataclass
from pathlib import Path

from src.web_sim.mujoco_session import DEFAULT_SCENE_PATH


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FRONTEND_DIST = PROJECT_ROOT / "web_frontend" / "dist"


@dataclass(frozen=True)
class BackendSettings:
    scene_path: Path = DEFAULT_SCENE_PATH
    frontend_dist: Path = DEFAULT_FRONTEND_DIST
    width: int = 960
    height: int = 540
    fps: int = 20
    jpeg_quality: int = 80

