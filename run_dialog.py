"""
双臂协商 (Dialectic) 规划入口
"""

import argparse
import json
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.pipeline.task_runner import TaskRunner
from src.llm_planner.dialog_planner import DialogPlanner


def main():
    parser = argparse.ArgumentParser(description='双臂协商 LLM 规划')
    parser.add_argument('--task', type=str, default="", help='任务指令')
    parser.add_argument('--plan', type=str, default=None, help='预置 plan JSON 文件')
    parser.add_argument('--api-key', type=str,
                        default='sk-3c750115309648feb24ffa4118c1d85d',
                        help='LLM API key')
    parser.add_argument('--base-url', type=str,
                        default='https://dashscope.aliyuncs.com/compatible-mode/v1',
                        help='API base URL')
    parser.add_argument('--model', type=str, default='qwen3.5-plus',
                        help='LLM 模型名')
    parser.add_argument('--time', type=float, default=300.0,
                        help='仿真时长 (秒)')
    parser.add_argument('--no-viewer', action='store_true',
                        help='无界面运行')
    args = parser.parse_args()

    scene_path = "src/assets/scenes/scene5_glare.xml"

    llm_planner = None
    plan_dict = None

    if args.plan:
        print(f"[Main] 从文件加载 plan: {args.plan}")
        with open(args.plan) as f:
            plan_dict = json.load(f)
    else:
        if not args.task:
            print("\n=== 双臂协商规划 - 交互式任务输入 ===")
            print("请输入您的任务指令")
            args.task = input("任务: ").strip()
            if not args.task:
                print("Error: 任务不能为空")
                return

        api_key = (args.api_key
                   or os.environ.get('DEEPSEEK_API_KEY')
                   or os.environ.get('OPENAI_API_KEY'))
        if not api_key:
            print("Error: 未提供 API key (使用 --api-key 或设置环境变量)")
            return

        llm_planner = DialogPlanner(
            api_key=api_key,
            base_url=args.base_url,
            model_name=args.model,
        )
        print('[Main] 使用双臂协商 (Dialectic) 规划器')

    runner = TaskRunner(
        scene_path=scene_path,
        llm_planner=llm_planner,
        task_instruction=args.task,
        plan_dict=plan_dict,
    )

    # ========== 轨迹可视化开关 ==========
    SHOW_TRAJECTORY = True
    # ====================================

    runner.run(total_time=args.time, use_viewer=not args.no_viewer,
               show_trajectory=SHOW_TRAJECTORY)


if __name__ == '__main__':
    main()
