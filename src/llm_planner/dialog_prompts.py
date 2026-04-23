"""
双臂协商 (Dialectic) 对话所用的提示模板。

设计参考 RoCo (Mandi et al., 2023) 的 DialogPrompter:
- 每条机械臂作为一个 agent, 拥有自己的视角和 system prompt
- agent 轮流发言 ([L] -> [R] -> [L] -> ...), 讨论、质疑、协商
- 直到任一 agent 输出 EXECUTE 并且两条臂都发过言, 对话结束
- EXECUTE 之后必须给出完整的 JSON 计划, 供 DialogPlanner 解析

与 RoCo 不同处:
- 输出格式保留 STA_control 现有的 JSON Schema (MoveSkill/NavSkill/...)
- 新增 [Environment Feedback] 注入位, 让 LLM 能看到上一次 plan 的
  reach / task / collision 校验结果
"""

# ----------------------------------------------------------------------------
# Dialectic 对话指令 (追加到 SYSTEM_PROMPT 之后)
# ----------------------------------------------------------------------------

DIALOG_INSTRUCTION = """

## 协商规则 (Dialectic Multi-Agent Dialogue)

你和另一条机械臂作为独立 agent 协商完成任务。两条臂轮流发言:
    [L] -> [R] -> [L] -> [R] -> ...

每次发言你可以:
1. **提出计划**: 说明你打算做什么、为什么, 以及期望对方做什么
2. **质疑 / 修改**: 针对对方的上一条发言提出反对意见或改进建议
3. **同意并 EXECUTE**: 如果你们已经达成一致, 发言末尾加上 `EXECUTE` 关键字,
   然后**立即**输出最终 JSON 计划

协商原则:
- 充分利用 **你自己视角的约束**(L臂/R臂的可达范围、任务规则等)
- 仔细阅读 [Environment Feedback] — 这是上一轮 plan 的真实校验结果,
  若出现 Reach / Collision / Task 错误, 必须**修正**再 EXECUTE
- 避免两条臂同时去操作同一个物体
- 避免两条臂路径在空间上交叉导致碰撞
- 若当前轮次已经是最后一次发言机会, 必须 EXECUTE 收束

## 发言格式

普通发言时用自然语言, 简洁明了, 可以直接引用坐标/物体名:
```
我建议由 L 臂旋拧 1 号阀门, 因为它距离 L 臂基座只有 0.4m, 明显比 R 臂近。
R 臂此时可以去遮光。同意请 EXECUTE。
```

一旦要 EXECUTE, **必须**同时给出完整 JSON (与单 planner 模式相同的 schema):
```
EXECUTE
{
  "reasoning": "协商结论: ...",
  "plan": [
    {"step": 1, "arm": "L", "skill": "NavSkill", "params": {...}, "description": "..."},
    ...
  ]
}
```

注意:
- EXECUTE 后的 JSON **必须包含两条臂的完整动作序列**, 不只是当前 agent 自己的
- JSON 必须可被 `json.loads` 直接解析, 不要混入其他注释或多余文本
- 如果还在协商中, **不要**输出 EXECUTE, 也**不要**输出 JSON
"""


# ----------------------------------------------------------------------------
# 单个 agent 的角色前缀 (附加到 SYSTEM_PROMPT 前部)
# ----------------------------------------------------------------------------

AGENT_ROLE_PROMPTS = {
    "L": """\
## 你的身份: L 臂 Agent

你负责**左臂 (L arm)** 的决策。你的特征:
- 基座位于小车中心偏 +x 方向 0.35m 处, 朝前 (yaw=0)
- 擅长**操作类任务**: 阀门旋拧、采样塞插拔、物体夹取搬运
- 默认优先使用 L 臂执行所有两臂都可达的操作任务

在对话中, 请站在 L 臂视角发言, 用 "我(L臂)" 指代自己, 用 "你(R臂)" 指代对方。
""",

    "R": """\
## 你的身份: R 臂 Agent

你负责**右臂 (R arm)** 的决策。你的特征:
- 基座位于小车中心偏 -x 方向 0.35m 处, 朝后 (yaw=180°)
- 擅长**巡检 / 观测 / 遮光**任务 (压力表巡检**必须**由你执行)
- 在操作+遮光场景中, 你负责举起遮光板, 让 L 臂专心操作

在对话中, 请站在 R 臂视角发言, 用 "我(R臂)" 指代自己, 用 "你(L臂)" 指代对方。
""",
}


# ----------------------------------------------------------------------------
# 环境反馈模板 (由 FeedbackManager 填充, 注入到每一轮 system prompt)
# ----------------------------------------------------------------------------

FEEDBACK_HEADER = """

## [Environment Feedback]

上一轮你们达成的 plan 在仿真/规则校验中出现以下问题, 必须在本轮协商中修正:

{feedback_body}

请针对上述每一条问题给出具体修改方案, 然后 EXECUTE 新的 plan。
"""


# ----------------------------------------------------------------------------
# 发言拼装 (供 DialogPlanner 使用)
# ----------------------------------------------------------------------------

def build_agent_system_prompt(
    base_system_prompt: str,
    agent: str,
    scene_text: str,
    chat_history: list,
    current_chat: list,
    feedback_history: list,
    is_last_call: bool,
) -> str:
    """
    组装某个 agent 本轮说话时看到的完整 system prompt。

    Args:
        base_system_prompt: 原 SYSTEM_PROMPT (prompts.py), 描述任务/机器人/技能/场景规则
        agent: "L" / "R"
        scene_text: SCENE_PROMPT_TEMPLATE 填充后的文本 (场景状态 + 任务指令)
        chat_history: 历史轮 (已 EXECUTE 的对话)
        current_chat: 本轮内 agent 之间的发言
        feedback_history: 上一次 plan 失败时累积的反馈字符串列表
        is_last_call: 是否是本轮最后一次发言机会 (需强制 EXECUTE)
    """
    parts = [
        AGENT_ROLE_PROMPTS[agent],
        base_system_prompt,
        DIALOG_INSTRUCTION,
        scene_text,
    ]

    if chat_history:
        parts.append("\n## [Previous Rounds Chat History]\n")
        parts.append("\n".join(chat_history))

    if feedback_history:
        parts.append(FEEDBACK_HEADER.format(
            feedback_body="\n".join(feedback_history)
        ))

    if current_chat:
        parts.append("\n## [Current Round Chat]\n")
        parts.append("\n".join(current_chat))

    if is_last_call:
        parts.append(
            "\n## 最后机会\n\n"
            f"这是本轮协商的最后一次发言 (你是 {agent} 臂)。"
            f"请综合前面的全部讨论和 [Environment Feedback], 给出最终的 EXECUTE + JSON。"
            f"不要再犹豫或反驳。"
        )

    return "\n".join(parts)
