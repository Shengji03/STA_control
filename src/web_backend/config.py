from dataclasses import dataclass
from pathlib import Path

from src.config.paths import DEFAULT_FRONTEND_DIST, DEFAULT_SCENE_PATH, OUTPUTS_DIR


@dataclass(frozen=True)
class BackendSettings:
    scene_path: Path = DEFAULT_SCENE_PATH
    frontend_dist: Path = DEFAULT_FRONTEND_DIST
    width: int = 960
    height: int = 540
    fps: int = 20
    jpeg_quality: int = 80
    data_dir: Path = OUTPUTS_DIR / 'web'

