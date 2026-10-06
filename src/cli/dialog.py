"""Run a natural-language or preplanned robot task."""

import argparse
import json
from pathlib import Path

from src.config.llm import LLMSettings
from src.config.paths import DEFAULT_SCENE_PATH


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="双臂协商 LLM 规划")
    parser.add_argument("--task", default="", help="任务指令")
    parser.add_argument("--plan", type=Path, help="预置 plan JSON 文件")
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE_PATH, help="MuJoCo 场景")
    parser.add_argument("--api-key", default=None, help="LLM API key，默认读取环境变量")
    parser.add_argument("--base-url", default=None, help="覆盖 STA_LLM_BASE_URL")
    parser.add_argument("--model", default=None, help="覆盖 STA_LLM_MODEL")
    parser.add_argument("--time", type=float, default=300.0, help="仿真时长（秒）")
    parser.add_argument("--no-viewer", action="store_true", help="无界面运行")
    parser.add_argument("--no-trajectory", action="store_true", help="关闭轨迹绘图")
    parser.add_argument('--plan-only', action='store_true', help='完成 LLM、分工优化和校验，打印结果，不运行仿真')
    parser.add_argument('--report', type=Path, help='保存优化计划与实际执行效果 JSON')
    parser.add_argument('--aux-budget', type=int, default=None, help='最多部署几次辅助；0 表示禁用辅助')
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    planner = None
    plan = None
    if args.plan is not None:
        with args.plan.open(encoding="utf-8") as stream:
            plan = json.load(stream)
    else:
        if not args.task:
            args.task = input("任务: ").strip()
        if not args.task:
            print("Error: 任务不能为空")
            return 1
        try:
            settings = LLMSettings.from_env(
                api_key=args.api_key, base_url=args.base_url, model_name=args.model,
            )
        except ValueError as exc:
            print(f"Error: {exc}")
            return 1
        from src.llm_planner.factory import create_planner

        planner = create_planner(settings)

    from src.pipeline.task_runner import TaskRunner

    runner = TaskRunner(
        scene_path=str(args.scene.resolve()), llm_planner=planner,
        task_instruction=args.task, plan_dict=plan,
    )
    if planner is not None:
        print(f"[LLM] model={settings.model_name}, base_url={settings.base_url}")
    if args.aux_budget is not None:
        from dataclasses import replace
        runner.planning_service.config = replace(runner.planning_service.config, max_auxiliaries=args.aux_budget)
    if args.plan_only:
        runner._do_planning()
        print(json.dumps(runner.prepared_plan, ensure_ascii=False, indent=2))
    else:
        runner.run(
            total_time=args.time, use_viewer=not args.no_viewer,
            show_trajectory=not args.no_trajectory,
        )
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        report = {'plan': runner.prepared_plan, 'optimization': runner.planning_report,
                  'execution_effects': runner.execution_report, 'state': runner.state}
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'规划与执行报告: {args.report.resolve()}')
    return 1 if getattr(runner, 'state', None) in ('FAILED', 'TIMEOUT') else 0


if __name__ == "__main__":
    raise SystemExit(main())
