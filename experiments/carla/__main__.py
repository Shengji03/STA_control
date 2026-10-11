"""One entry point for CARLA training, refinement, evaluation and deployment."""

import argparse
import json
import sys
from importlib import import_module
from pathlib import Path

from src.config.paths import DEFAULT_STA_PARAMETERS

COMMANDS = {
    "train": "suite",
    "refine": "refinement",
    "evaluate": "evaluate",
    "export": "exports",
}


def install(argv):
    """Explicitly install a validated bank at the requested destination."""
    from src.controller.control_parameters import SkillParameterBank

    from .training import save_json

    parser = argparse.ArgumentParser(description=install.__doc__)
    parser.add_argument("parameters", type=Path)
    parser.add_argument("--target", type=Path, default=DEFAULT_STA_PARAMETERS)
    args = parser.parse_args(argv)
    SkillParameterBank.load(args.parameters)
    payload = json.loads(args.parameters.read_text(encoding="utf-8"))
    target = args.target.resolve()
    if target.exists() and json.loads(target.read_text(encoding="utf-8")) == payload:
        print(f"Parameter bank already installed: {target}")
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    save_json(target, payload)
    print(f"Validated parameter bank installed: {target}")
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "install":
        return install(argv[1:])
    if argv and argv[0] in COMMANDS:
        module = import_module(f"experiments.carla.{COMMANDS[argv[0]]}")
        return module.main(argv[1:]) or 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=[*COMMANDS, "install"], nargs="?")
    parser.parse_args(argv)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
