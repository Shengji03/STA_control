"""Resolve project resources independently of the process working directory."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = PROJECT_ROOT / "src" / "assets"
SCENES_DIR = ASSETS_DIR / "scenes"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
DEFAULT_SCENE_PATH = SCENES_DIR / "scene5_glare.xml"
DEFAULT_FRONTEND_DIST = PROJECT_ROOT / "web_frontend" / "dist"


def scene_path(name: str) -> Path:
    """Return a bundled scene by filename or stem."""
    filename = name if name.endswith(".xml") else f"{name}.xml"
    if Path(filename).name != filename or "\\" in filename:
        raise ValueError("Scene name must be a filename or stem")
    return SCENES_DIR / filename
