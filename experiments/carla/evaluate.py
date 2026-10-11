"""Run and record a complete trained skill composition without LLM or DDPG."""

import argparse
import io
import json
from contextlib import redirect_stdout
from math import pi
from pathlib import Path

import numpy as np

from experiments.carla.benchmarks import Scenario, evaluate, prepare
from experiments.carla.exports import metrics_summary
from experiments.carla.suite import FAMILIES
from experiments.carla.training import save_json
from src.controller.control_parameters import SkillParameterBank


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=FAMILIES, default="assembly")
    parser.add_argument("--sta-parameters", type=Path)
    parser.add_argument(
        "--scene",
        choices=["scene4_pipeline.xml", "scene5_glare.xml"],
        default="scene5_glare.xml",
    )
    parser.add_argument("--angle-deg", type=float, default=180.0)
    parser.add_argument("--arm-order", choices=["LL", "LR", "RL", "RR"], default="LR")
    parser.add_argument(
        "--assembly-order", choices=["012", "102", "021"], default="012"
    )
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/carla/manual_evaluation")
    )
    args = parser.parse_args(argv)
    bank = SkillParameterBank.load(args.sta_parameters) if args.sta_parameters else None
    case = Scenario(
        "manual_" + args.family,
        family=args.family,
        scene="scene3.xml" if args.family == "assembly" else args.scene,
        angle=args.angle_deg * pi / 180,
        arm_order=tuple(args.arm_order),
        order=tuple(map(int, args.assembly_order)),
    )
    with redirect_stdout(io.StringIO()):
        runner = prepare(case, bank)
    if args.viewer:
        import time

        import mujoco.viewer

        with mujoco.viewer.launch_passive(runner.model, runner.data) as viewer:
            original = runner.step_once

            def displayed_step():
                if not viewer.is_running():
                    runner.state = "CANCELLED"
                    return False
                start = time.monotonic()
                continuing = original()
                if round(runner.data.time / runner.model.opt.timestep) % 10 == 0:
                    viewer.sync()
                wait = runner.model.opt.timestep - (time.monotonic() - start)
                if wait > 0:
                    time.sleep(wait)
                return continuing

            runner.step_once = displayed_step
            result = evaluate(runner, case, record=True)
    else:
        result = evaluate(runner, case, record=True)
    rows = result.pop("traces")
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / "motion_traces.npz",
        time=np.asarray([r[0] for r in rows]),
        arm=np.asarray([r[1] for r in rows]),
        skill=np.asarray([r[2] for r in rows]),
        desired_q=np.asarray([r[3] for r in rows]),
        actual_q=np.asarray([r[4] for r in rows]),
        actual_torque_Nm=np.asarray([r[5] for r in rows]),
        external_wrench=np.asarray([r[6] for r in rows]),
    )
    save_json(
        args.output / "evaluation.json",
        {
            "parameter_bank": str(args.sta_parameters) if args.sta_parameters else None,
            "control_parameters": bank.entries if bank else {},
            "result": result,
            "metrics": metrics_summary(result),
            "physics_rate_Hz": 500,
            "trace_rate_Hz": 100,
        },
    )
    print(
        json.dumps(
            {
                "valid": result["valid"],
                "state": result["state"],
                "effects": result["execution_effects"],
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
