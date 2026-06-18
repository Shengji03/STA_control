"""
DialogPlanner —— 双臂 Dialectic 协商 LLM 规划器。

对外接口与 LLMPlanner 保持一致 (plan / history / print_plan), 可以无缝替换。
额外提供:
  - attach_env(model, data, arm_offsets, arm_yaws): 注入 MuJoCo 环境,
    激活 FeedbackManager 做 reach / task / collision 校验;
    未注入时退化为纯对话 (只做 task 级规则检查)。

内部流程 (参考 RoCo DialogPrompter):
  replan_idx ∈ [0, num_replans):
      chat 轮转: [L] -> [R] -> [L] -> [R] -> ...
          每个 agent 查询一次 LLM, 回应进入 current_chat
          若响应含 "EXECUTE" 且两边都已发过言, 结束本轮
          若达到 max_calls_per_round, 强制结束
      把最后一条 EXECUTE 响应里的 JSON 解析出来
      用 FeedbackManager 校验:
          通过 -> 返回 plan_dict
          失败 -> 把 feedback 加入 feedback_history, 进入下一轮 replan
  全部失败则返回最后一次的 plan_dict (允许上层继续尝试或 raise)。
"""

from __future__ import annotations

import json
import time
from typing import Dict, List, Optional, Tuple

from .dialog_prompts import build_agent_system_prompt
from .feedback_manager import FeedbackManager
from .prompts import SYSTEM_PROMPT, SCENE_PROMPT_TEMPLATE, PLAN_JSON_SCHEMA


AGENT_ORDER = ["L", "R"]


class DialogPlanner:
    """
    双臂 Dialectic 协商规划器。

    Args:
        api_key:              LLM API key
        base_url:             OpenAI 兼容接口 base url
        model_name:           模型名
        temperature:          采样温度
        max_completion_tokens:单次响应最大 token 数
        num_replans:          plan 被 feedback 拒绝后最多重新协商多少轮
        max_calls_per_round:  一轮协商内最多调 LLM 多少次 (总计, 跨 agent)
        verbose:              是否打印每条发言
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model_name: str = "gpt-4",
        temperature: float = 0.2,
        max_completion_tokens: int = 4096,
        num_replans: int = 3,
        max_calls_per_round: int = 10,
        verbose: bool = True,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("请安装 openai 库: pip install openai>=1.0")

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model_name
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.num_replans = num_replans
        self.max_calls_per_round = max_calls_per_round
        self.verbose = verbose

        self._history: List[Dict] = []
        self._chat_history: List[str] = []  # 跨 replan 的历史聊天 (仅成功 EXECUTE 的)

        # 环境相关, 由 attach_env 注入
        self._fbm: Optional[FeedbackManager] = None

    # ------------------------------------------------------------------
    # 环境注入
    # ------------------------------------------------------------------

    def attach_env(self, model, data, arm_offsets, arm_yaws, dof=6):
        """在 TaskRunner 初始化完成后调用, 激活反馈管理器。"""
        self._fbm = FeedbackManager(
            model=model, data=data,
            arm_offsets=arm_offsets, arm_yaws=arm_yaws,
            dof=dof,
        )
        if self.verbose:
            print("[DialogPlanner] FeedbackManager 已启用 "
                  "(task + reach + collision 校验)")

    # ------------------------------------------------------------------
    # 对外主接口 (与 LLMPlanner.plan 对齐)
    # ------------------------------------------------------------------

    def plan(
        self,
        snapshot: Dict,
        task_instruction: str,
        extra_context: str = "",
    ) -> Dict:
        scene_text = self._build_scene_text(snapshot, task_instruction,
                                            extra_context)

        feedback_history: List[str] = []
        last_plan: Optional[Dict] = None
        last_agent_responses: List[str] = []

        for replan_idx in range(self.num_replans):
            if self.verbose:
                print(f"\n[DialogPlanner] === Replan round {replan_idx} ===")

            final_response, agent_responses = self._run_dialog_round(
                scene_text=scene_text,
                chat_history=self._chat_history,
                feedback_history=feedback_history,
            )
            last_agent_responses = agent_responses

            plan_dict = self._extract_plan_json(final_response)
            last_plan = plan_dict

            if not plan_dict.get("plan"):
                feedback_history.append(
                    "上一次响应里未能提取到有效 JSON plan, "
                    "请确认 EXECUTE 后立刻输出可被 json.loads 解析的 JSON 对象, "
                    "字段必须包含 reasoning 和 plan 数组。"
                )
                continue

            # 校验
            if self._fbm is None:
                if self.verbose:
                    print("[DialogPlanner] 未 attach_env, 跳过环境校验。")
                self._commit_history(agent_responses, plan_dict)
                self._record_usage(replan_idx, task_instruction, plan_dict)
                return plan_dict

            ok, feedback = self._fbm.give_feedback(
                plan_dict, snapshot, task_instruction
            )
            if ok:
                if self.verbose:
                    print(f"[DialogPlanner] Replan {replan_idx} 通过校验")
                self._commit_history(agent_responses, plan_dict)
                self._record_usage(replan_idx, task_instruction, plan_dict)
                return plan_dict

            if self.verbose:
                print(f"[DialogPlanner] Replan {replan_idx} 校验失败:\n"
                      f"{feedback}")
            feedback_history.append(feedback)

        # 所有 replan 都没通过, 仍返回最后一次 plan (让上层决定如何处理)
        if self.verbose:
            print(f"[DialogPlanner] ⚠️  {self.num_replans} 轮协商后仍未通过校验, "
                  f"返回最后一次 plan。")
        if last_plan is None:
            last_plan = {"reasoning": "协商失败", "plan": []}
        self._commit_history(last_agent_responses, last_plan)
        self._record_usage(self.num_replans - 1, task_instruction, last_plan)
        return last_plan

    # ------------------------------------------------------------------
    # 单轮对话 (L -> R -> L -> R -> ... 直到 EXECUTE 或超限)
    # ------------------------------------------------------------------

    def _run_dialog_round(
        self,
        scene_text: str,
        chat_history: List[str],
        feedback_history: List[str],
    ) -> Tuple[str, List[str]]:
        """
        返回 (final_response, agent_responses)。
        final_response 是最后一个 agent 的完整响应 (通常包含 EXECUTE+JSON)。
        """
        agent_responses: List[str] = []
        num_responses = {name: 0 for name in AGENT_ORDER}
        n_calls = 0
        final_response = ""

        while n_calls < self.max_calls_per_round:
            for agent in AGENT_ORDER:
                is_last_call = (n_calls == self.max_calls_per_round - 1)

                system_prompt = build_agent_system_prompt(
                    base_system_prompt=SYSTEM_PROMPT,
                    agent=agent,
                    scene_text=scene_text,
                    chat_history=chat_history,
                    current_chat=agent_responses,
                    feedback_history=feedback_history,
                    is_last_call=is_last_call,
                )
                user_prompt = f"你是 {agent} 臂 Agent, 请发言:"

                response = self._query_llm(
                    system_prompt, user_prompt,
                    force_json=is_last_call,
                )
                final_response = response or ""

                msg = f"[{agent}]:\n{final_response.strip()}"
                agent_responses.append(msg)
                num_responses[agent] += 1
                n_calls += 1

                if self.verbose:
                    print(f"\n--- [{agent}] round call #{n_calls} ---")
                    print(final_response.strip()[:500]
                          + ("..." if len(final_response) > 500 else ""))

                # EXECUTE 收束: 两边都至少发过一次
                if "EXECUTE" in final_response and all(
                    v > 0 for v in num_responses.values()
                ):
                    return final_response, agent_responses

                if n_calls >= self.max_calls_per_round:
                    return final_response, agent_responses

        return final_response, agent_responses

    # ------------------------------------------------------------------
    # LLM 调用
    # ------------------------------------------------------------------

    _MAX_QUERY_RETRIES = 3

    def _query_llm(
        self, system_prompt: str, user_prompt: str, force_json: bool = False
    ) -> str:
        """
        调用 OpenAI 兼容接口, 失败重试几次。

        force_json=True 时启用 json_object 响应格式, 用于最后一次强制 EXECUTE
        的发言 (JSON 结构化更可靠)。
        """
        api_kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_completion_tokens": self.max_completion_tokens,
        }

        # GPT-5.x: structured output
        if self.model.startswith("gpt-5") and force_json:
            api_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": PLAN_JSON_SCHEMA,
            }
        elif force_json:
            # Qwen / GPT-4 等兼容路径: 只要 JSON-ish
            api_kwargs["response_format"] = {"type": "json_object"}
            api_kwargs["temperature"] = self.temperature
        else:
            api_kwargs["temperature"] = self.temperature

        for attempt in range(self._MAX_QUERY_RETRIES):
            t0 = time.time()
            try:
                resp = self.client.chat.completions.create(**api_kwargs)
                elapsed = time.time() - t0
                content = resp.choices[0].message.content or ""
                usage = resp.usage
                if self.verbose and usage is not None:
                    print(f"  [LLM] {elapsed:.1f}s, "
                          f"tokens={usage.prompt_tokens}+"
                          f"{usage.completion_tokens}")
                return content
            except Exception as e:
                if self._is_authentication_error(e):
                    raise RuntimeError(f"LLM authentication failed: {e}") from e
                print(f"  [LLM] 调用失败 (第 {attempt+1} 次): {e}")
                time.sleep(1.0)
        return ""

    @staticmethod
    def _is_authentication_error(error: Exception) -> bool:
        status_code = getattr(error, "status_code", None)
        code = getattr(error, "code", None)
        body = getattr(error, "body", None)
        text = f"{error} {code} {body}".lower()
        if status_code == 401:
            return True
        return (
            "invalid_api_key" in text
            or "incorrect api key" in text
            or ("api key" in text and ("invalid" in text or "incorrect" in text))
            or ("401" in text and ("api" in text or "auth" in text))
        )

    # ------------------------------------------------------------------
    # 响应解析
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_plan_json(final_response: str) -> Dict:
        """
        从 final_response 中提取 EXECUTE 后的 JSON。支持两种形态:
          1) "EXECUTE\n{...json...}"
          2) 整段就是 JSON (force_json 模式下)
        """
        if not final_response:
            return {"reasoning": "空响应", "plan": []}

        text = final_response
        if "EXECUTE" in text:
            text = text.split("EXECUTE", 1)[1]

        # 去掉 markdown 代码围栏
        text = text.strip()
        if text.startswith("```"):
            # 去掉第一行 ```json 或 ```
            lines = text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        # 找第一个 { 和最后一个 }
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass
        return {"reasoning": "JSON 解析失败", "plan": [], "raw": final_response}

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    @staticmethod
    def _build_scene_text(
        snapshot: Dict, task_instruction: str, extra_context: str = ""
    ) -> str:
        scene_state = (
            snapshot.get("planner_state")
            or snapshot.get("world_state")
            or snapshot.get("state", {})
        )
        state_json = json.dumps(scene_state, ensure_ascii=False, indent=2)
        text = SCENE_PROMPT_TEMPLATE.format(
            state_json=state_json, task_instruction=task_instruction,
        )
        if extra_context:
            text += f"\n\n## 补充信息\n\n{extra_context}"
        return text

    def _commit_history(self, agent_responses: List[str], plan_dict: Dict):
        """成功 EXECUTE 后, 把本轮聊天压入历史, 供后续 step 参考。"""
        if agent_responses:
            joined = "\n".join(agent_responses)
            self._chat_history.append(
                f"[Chat]\n{joined}\n[Executed Plan]\n"
                f"{json.dumps(plan_dict.get('plan', []), ensure_ascii=False)}"
            )

    def _record_usage(
        self, replan_idx: int, task_instruction: str, plan_dict: Dict
    ):
        self._history.append({
            "replan_idx": replan_idx,
            "instruction": task_instruction,
            "response": plan_dict,
        })

    # ------------------------------------------------------------------
    # 与 LLMPlanner 对齐的属性 / 方法
    # ------------------------------------------------------------------

    @property
    def history(self) -> List[Dict]:
        return self._history

    def clear_history(self):
        self._history.clear()
        self._chat_history.clear()

    def print_plan(self, result: Dict):
        print("\n" + "=" * 60)
        print("  DialogPlanner 协商结果")
        print("=" * 60)
        print(f"  推理: {result.get('reasoning', '')}\n")
        for step in result.get("plan", []):
            arm = step.get("arm", "?")
            skill = step.get("skill", "?")
            params = step.get("params", {})
            desc = step.get("description", "")
            idx = step.get("step", "?")
            print(f"  [{idx}] {arm}臂: {skill}({params}) -- {desc}")
        print()
