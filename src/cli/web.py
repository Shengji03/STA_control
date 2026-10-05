"""Run the existing Vue and FastAPI application."""

import argparse
from pathlib import Path

from src.config.paths import DEFAULT_FRONTEND_DIST, DEFAULT_SCENE_PATH


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the STA Vue + FastAPI web application.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--http-port", type=int, default=8000)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE_PATH)
    parser.add_argument("--frontend-dist", type=Path, default=DEFAULT_FRONTEND_DIST)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--quality", type=int, default=80)
    parser.add_argument("--no-simulation", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    import uvicorn

    from src.web_backend.app import create_app
    from src.web_backend.config import BackendSettings

    settings = BackendSettings(
        scene_path=args.scene.resolve(), frontend_dist=args.frontend_dist.resolve(),
        width=args.width, height=args.height, fps=args.fps, jpeg_quality=args.quality,
    )
    app = create_app(enable_simulation=not args.no_simulation, settings=settings)
    uvicorn.run(app, host=args.host, port=args.http_port)


if __name__ == "__main__":
    main()
