"""CSV, scientific figures and observed time-series exports for a completed suite."""

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metrics_summary(result):
    metrics = list(result["per_skill"].values())
    count = sum(m["samples"] for m in metrics)

    def rms(key):
        return float(
            np.sqrt(sum(m[key] ** 2 * m["samples"] for m in metrics) / max(1, count))
        )

    goals = result["execution_effects"].get("goals", [])
    angles = [g["angle_error_rad"] for g in goals if "angle_error_rad" in g]
    positions = [g["position_error_m"] for g in goals if "position_error_m" in g]
    shade = [g["shade_fraction"] for g in goals if g.get("shade_fraction") is not None]
    return {
        "samples_500Hz": count,
        "joint_rmse_rad": rms("joint_rmse_rad"),
        "force_rms_N": rms("force_rms_N"),
        "peak_force_N": max((m["peak_force_N"] for m in metrics), default=0),
        "actual_torque_peak_Nm": max(
            (m["peak_actual_torque_Nm"] for m in metrics), default=0
        ),
        "torque_effort_rms_fraction": rms("effort_rms_fraction"),
        "torque_delta_rms_fraction": rms("torque_delta_rms_fraction"),
        "maximum_angle_error_deg": float(np.degrees(max(angles))) if angles else "",
        "maximum_object_position_error_m": max(positions) if positions else "",
        "minimum_shade_fraction": min(shade) if shade else "",
        "transition_torque_jump_Nm": result["transition_torque_jump_Nm"],
    }


def export_tables(directory, output=None):
    source = Path(directory)
    directory = Path(output) if output else source
    directory.mkdir(parents=True, exist_ok=True)
    summary = read_json(source / "suite_summary.json")
    profiles = []
    for pid, evidence in summary["profile_evidence"].items():
        stats = evidence["statistics"]
        whole = evidence["complete_statistics"]
        profiles.append(
            {
                "profile": pid,
                "frozen_seed": evidence["winner_seed"],
                "accepted_as_library_candidate": evidence["accepted"],
                "alpha_ratio": evidence["ratios"][0],
                "beta_ratio": evidence["ratios"][1],
                "lambda_ratio": evidence["ratios"][2],
                "cases": stats["cases"],
                "baseline_successes": stats["baseline_successes"],
                "learned_successes": stats["learned_successes"],
                "paired_successes": stats["paired_successes"],
                "skill_objective_gain_pct": stats["mean_relative_gain_pct"],
                "skill_gain_CI_low_pct": stats["gain_95pct_interval"][0],
                "skill_gain_CI_high_pct": stats["gain_95pct_interval"][1],
                "complete_task_gain_pct": whole["mean_relative_gain_pct"],
                "worst_skill_gain_pct": stats["worst_relative_gain_pct"],
            }
        )
    combos = []
    for family, evidence in summary["composition_evidence"].items():
        stats = evidence["statistics"]
        combos.append(
            {
                "family": family,
                "frozen_seed": evidence["winner_seed"],
                "accepted_for_runtime": evidence["accepted"],
                "cases": stats["cases"],
                "baseline_successes": stats["baseline_successes"],
                "learned_successes": stats["learned_successes"],
                "paired_successes": stats["paired_successes"],
                "complete_task_gain_pct": stats["mean_relative_gain_pct"],
                "gain_CI_low_pct": stats["gain_95pct_interval"][0],
                "gain_CI_high_pct": stats["gain_95pct_interval"][1],
                "worst_gain_pct": stats["worst_relative_gain_pct"],
                "stitched_library_gain_pct": evidence["stitch_statistics"][
                    "mean_relative_gain_pct"
                ],
                "refinement_gain_vs_stitched_pct": evidence["refinement_statistics"][
                    "mean_relative_gain_pct"
                ],
            }
        )
    rows = []
    for category, field in [
        ("skill", "profile_holdout"),
        ("composition", "combo_holdout"),
    ]:
        for label, result in summary[field].items():
            reference = result["baseline"]
            variants = (
                ["baseline", "learned"]
                if category == "skill"
                else ["baseline", "stitched", "learned"]
            )
            for variant in variants:
                for baseline, evaluation in zip(reference, result[variant]):
                    case = evaluation["case"]
                    valid_pair = baseline["valid"] and evaluation["valid"]
                    rows.append(
                        {
                            "category": category,
                            "job": label,
                            "seed": result["seed"],
                            "variant": variant,
                            "case": case["name"],
                            "family": case["family"],
                            "scene": case["scene"],
                            "speed_duration_multiplier": case["speed"],
                            "payload_mass_multiplier": case["payload"],
                            "resistance_multiplier": case["resistance"],
                            "grasp_duration_multiplier": case["grasp"],
                            "requested_angle_deg": float(np.degrees(case["angle"])),
                            "target": case["target"],
                            "cart_offset_m": case["cart_offset"],
                            "assembly_order": ",".join(map(str, case["order"])),
                            "arm_order": ",".join(case.get("arm_order", ("L", "R"))),
                            "valid": evaluation["valid"],
                            "state": evaluation["state"],
                            "scored_cost": evaluation["cost"],
                            "complete_task_cost": evaluation["complete_cost"],
                            "gain_vs_baseline_pct": (
                                baseline["cost"] - evaluation["cost"]
                            )
                            / baseline["cost"]
                            * 100
                            if valid_pair
                            else "",
                            "violation": ",".join(evaluation["violations"]),
                            **metrics_summary(evaluation),
                        }
                    )
    write_csv(directory / "skills_results.csv", profiles)
    write_csv(directory / "combinations_results.csv", combos)
    write_csv(directory / "holdout_cases.csv", rows)
    return {
        "skills": len(profiles),
        "combinations": len(combos),
        "case_rows": len(rows),
    }


def export_refinement_tables(directory, output=None):
    source = Path(directory)
    directory = Path(output) if output else source
    directory.mkdir(parents=True, exist_ok=True)
    summary = read_json(source / "summary.json")
    combinations, cases = [], []
    for family, evidence in summary["holdout"].items():
        stats = evidence["statistics"]
        combinations.append(
            {
                "family": family,
                "accepted_for_runtime": evidence["accepted"],
                "cases": stats["cases"],
                "baseline_successes": stats["baseline_successes"],
                "learned_successes": stats["learned_successes"],
                "complete_task_gain_pct": stats["mean_relative_gain_pct"],
                "gain_CI_low_pct": stats["gain_95pct_interval"][0],
                "gain_CI_high_pct": stats["gain_95pct_interval"][1],
                "worst_gain_pct": stats["worst_relative_gain_pct"],
                "parameters": json.dumps(evidence["parameters"], sort_keys=True),
                "physical_correction": summary["physical_corrections"][family],
                "reference": "corrected fixed STA baseline; same physical corrections in both variants",
            }
        )
        for variant in ("baseline", "learned"):
            for baseline, result in zip(evidence["baseline"], evidence[variant]):
                case = result["case"]
                valid_pair = baseline["valid"] and result["valid"]
                cases.append(
                    {
                        "family": family,
                        "variant": variant,
                        "case": case["name"],
                        "scene": case["scene"],
                        "speed_duration_multiplier": case["speed"],
                        "payload_mass_multiplier": case["payload"],
                        "resistance_multiplier": case["resistance"],
                        "grasp_duration_multiplier": case["grasp"],
                        "requested_angle_deg": float(np.degrees(case["angle"])),
                        "cart_offset_m": case["cart_offset"],
                        "assembly_order": ",".join(map(str, case["order"])),
                        "arm_order": "".join(case["arm_order"]),
                        "valid": result["valid"],
                        "state": result["state"],
                        "complete_task_cost": result["cost"],
                        "gain_vs_corrected_baseline_pct": (
                            baseline["cost"] - result["cost"]
                        )
                        / baseline["cost"]
                        * 100
                        if valid_pair
                        else "",
                        "violations": ",".join(result["violations"]),
                        **metrics_summary(result),
                    }
                )
    write_csv(directory / "combinations_results.csv", combinations)
    write_csv(directory / "holdout_cases.csv", cases)
    return {"combinations": len(combinations), "case_rows": len(cases)}


def plot_refinement(directory, output=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    source = Path(directory)
    directory = Path(output) if output else source
    directory.mkdir(parents=True, exist_ok=True)
    evidence = read_json(source / "summary.json")["holdout"]
    families = list(evidence)
    colors = ["#2563a6", "#16806d", "#9b5aaf"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for index, (family, color) in enumerate(zip(families, colors)):
        stats = evidence[family]["statistics"]
        mean = stats["mean_relative_gain_pct"]
        low, high = stats["gain_95pct_interval"]
        axes[0].errorbar(
            mean,
            index,
            xerr=[[mean - low], [high - mean]],
            fmt="o",
            color=color,
            capsize=5,
            markersize=7,
        )
        axes[0].annotate(
            f"{mean:.2f}% | {stats['learned_successes']}/{stats['cases']} successful",
            (high, index),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
        )
        axes[1].plot(
            range(1, stats["cases"] + 1),
            stats["relative_gains_pct"],
            color=color,
            marker="o",
            markersize=3,
            lw=1,
            label=family,
        )
    axes[0].set(
        yticks=range(len(families)),
        yticklabels=families,
        ylim=(-0.7, 2.7),
        xlim=(-1, 36),
        title="Mean and scenario bootstrap 95% CI",
        xlabel="Complete-task cost reduction (%)",
    )
    axes[0].invert_yaxis()
    axes[1].set(
        title="Paired gains on all untouched holdout conditions",
        xlabel="Condition index within each family",
        ylabel="Complete-task cost reduction (%)",
    )
    axes[1].legend()
    for ax in axes:
        ax.axvline(0, color="#8994a4", lw=0.8) if ax is axes[0] else ax.axhline(
            0, color="#8994a4", lw=0.8
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.17)
    fig.suptitle("Frozen CARLA candidates vs corrected fixed STA baseline", fontsize=13)
    path = directory / "refinement_validation.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_suite(directory, output=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    source = Path(directory)
    directory = Path(output) if output else source
    directory.mkdir(parents=True, exist_ok=True)
    summary = read_json(source / "suite_summary.json")
    colors = ["#2563a6", "#16806d", "#9b5aaf"]
    pids = list(summary["profile_evidence"])
    fig, axes = plt.subplots(
        int(np.ceil(len(pids) / 3)),
        3,
        figsize=(16, 3.1 * np.ceil(len(pids) / 3)),
        constrained_layout=True,
    )
    for ax, pid in zip(axes.flat, pids):
        for index, (label, seed) in enumerate(
            (key, result)
            for key, result in summary["profile_seeds"].items()
            if key.startswith(pid + "/")
        ):
            rows = [
                json.loads(line)
                for line in (source / "profiles" / label / "trials.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            ax.plot(
                [r["iteration"] for r in rows],
                [r["best_cost"] for r in rows],
                color=colors[index % 3],
                label=f"Seed {seed['seed']}",
            )
        ax.set(title=pid, xlabel="CARLA candidate", ylabel="Best training-window cost")
        ax.legend(fontsize=7)
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.17)
    fig.suptitle(
        "Real-prefix skill training: one skill plus a 2-second handoff; whole-task validation is separate",
        fontsize=13,
    )
    fig.savefig(directory / "skills_learning.png", dpi=160)
    plt.close(fig)
    families = list(summary["composition_evidence"])
    fig, axes = plt.subplots(
        int(np.ceil(len(families) / 2)),
        2,
        figsize=(14, 3 * np.ceil(len(families) / 2)),
        constrained_layout=True,
    )
    for ax, family in zip(axes.flat, families):
        for index, (label, seed) in enumerate(
            (key, result)
            for key, result in summary["combo_seeds"].items()
            if key.startswith(family + "/")
        ):
            rows = [
                json.loads(line)
                for line in (source / "combinations" / label / "trials.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            ax.plot(
                [r["iteration"] for r in rows],
                [r["best_cost"] for r in rows],
                color=colors[index % 3],
                label=f"Seed {seed['seed']}",
            )
        ax.set(
            title=family,
            xlabel="Joint-refinement candidate",
            ylabel="Best complete-task cost",
        )
        ax.legend(fontsize=7)
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.17)
    fig.suptitle(
        "Skill composition training: all preceding actions, transitions and physical effects are simulated",
        fontsize=13,
    )
    fig.savefig(directory / "combinations_learning.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    for ax, items, title in (
        (
            axes[0],
            summary["profile_evidence"],
            "Individual skill objective on complete-task holdout",
        ),
        (
            axes[1],
            summary["composition_evidence"],
            "Complete composition objective on untouched holdout",
        ),
    ):
        labels = list(items)
        for index, label in enumerate(labels):
            row = items[label]
            stats = row["statistics"]
            mean = stats["mean_relative_gain_pct"]
            lo, hi = stats["gain_95pct_interval"]
            ax.errorbar(
                mean,
                index,
                xerr=[[max(0, mean - lo)], [max(0, hi - mean)]],
                fmt="o",
                capsize=4,
                color="#16806d" if row["accepted"] else "#8994a4",
            )
            ax.annotate(
                f"{stats['learned_successes']}/{stats['cases']}",
                (hi, index),
                xytext=(5, 0),
                textcoords="offset points",
                fontsize=8,
                va="center",
            )
        ax.set_yticks(range(len(labels)), labels)
        ax.invert_yaxis()
        ax.axvline(0, color="#8994a4", lw=1)
        ax.set(
            title=title,
            xlabel="Paired cost reduction (%) with scenario bootstrap 95% CI",
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.17)
    fig.suptitle(
        "Frozen winners; gray outcomes are recorded but not deployed as improved parameters",
        fontsize=13,
    )
    fig.savefig(directory / "suite_validation.png", dpi=170)
    plt.close(fig)
    return directory / "suite_validation.png"


def trace_case(case_dict, parameters, expected, path):
    import io
    from contextlib import redirect_stdout

    from experiments.carla.benchmarks import evaluate, make_bank, prepare
    from experiments.carla.suite import scenario_from_dict
    from experiments.carla.training import save_json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with redirect_stdout(io.StringIO()):
        runner = prepare(scenario_from_dict(case_dict), make_bank(parameters))
        result = evaluate(runner, scenario_from_dict(case_dict), record=True)
    np.testing.assert_allclose(result["cost"], expected["cost"], rtol=0, atol=1e-12)
    assert result["execution_effects"] == expected["execution_effects"]
    rows = result.pop("traces")
    np.savez_compressed(
        path,
        time=np.asarray([r[0] for r in rows]),
        arm=np.asarray([r[1] for r in rows]),
        skill=np.asarray([r[2] for r in rows]),
        desired_q=np.asarray([r[3] for r in rows]),
        actual_q=np.asarray([r[4] for r in rows]),
        actual_torque_Nm=np.asarray([r[5] for r in rows]),
        external_wrench=np.asarray([r[6] for r in rows]),
    )
    save_json(
        path.with_suffix(".json"),
        {
            "case": case_dict,
            "parameters": parameters,
            "evaluation": result,
            "physics_rate_Hz": 500,
            "trace_rate_Hz": 100,
            "torque_source": "actuator_force",
            "wrench_order": "world Fx,Fy,Fz,Mx,My,Mz",
            "observer_matched_scored_cost": True,
        },
    )
    return str(path)


def export_traces(directory, workers=3, trace_limit=None, output=None):
    source = Path(directory)
    directory = Path(output) if output else source
    directory.mkdir(parents=True, exist_ok=True)
    jobs = []
    if (source / "suite_summary.json").exists():
        summary = read_json(source / "suite_summary.json")
        holdouts = {
            family: summary["combo_holdout"][f"{family}/seed_{evidence['winner_seed']}"]
            for family, evidence in summary["composition_evidence"].items()
        }
    else:
        holdouts = read_json(source / "summary.json")["holdout"]
    for family, holdout in holdouts.items():
        for variant, parameters in [
            ("baseline", {}),
            ("learned", holdout["parameters"]),
        ]:
            for index, expected in enumerate(holdout[variant][:trace_limit]):
                jobs.append(
                    (
                        expected["case"],
                        parameters,
                        expected,
                        str(
                            directory / "traces" / family / f"{variant}_{index:02d}.npz"
                        ),
                    )
                )
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(trace_case, *args) for args in jobs]
        for i, future in enumerate(futures):
            print("TRACE_EXPORTED", i + 1, len(futures), future.result(), flush=True)
    return len(jobs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="Write exports into a new directory while preserving the study",
    )
    parser.add_argument("--traces", action="store_true")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--trace-limit",
        type=int,
        help="Cases per family and variant; omit to export every case",
    )
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.trace_limit is not None and args.trace_limit < 1:
        parser.error("--trace-limit must be positive")
    refinement = not (args.directory / "suite_summary.json").exists()
    print(
        "TABLES",
        (export_refinement_tables if refinement else export_tables)(
            args.directory, args.output
        ),
        flush=True,
    )
    print(
        "FIGURE",
        (plot_refinement if refinement else plot_suite)(args.directory, args.output),
        flush=True,
    )
    if args.traces:
        print(
            "TRACE_FILES",
            export_traces(args.directory, args.workers, args.trace_limit, args.output),
            flush=True,
        )


if __name__ == "__main__":
    main()
