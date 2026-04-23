"""
Feedback Manager —— 对 LLM 输出的 plan 做多级校验, 生成可回注 prompt 的反馈。

灵感来自 RoCo (Mandi et al., 2023) 的 prompting/feedback.py, 但适配到
STA_control 的 Skill JSON 体系:

    1. Task 级语义校验 (根据任务指令和 prompts.py 中的硬规则)
    2. Reach 级校验 (复用 PlanExecutor 的 IK + 距离检查, 捕获 ValueError)
    3. Collision 级校验 (把 plan 回放到 MuJoCo 模型上做前向碰撞检查)

关键设计:
- Reach 校验直接复用 `PlanExecutor.parse`, 它内部已对 target_pos 做 IK +
  0.85m reach 距离判断, 失败时抛 ValueError — 我们捕获并格式化成 feedback。
- Collision 校验新建一个 `mujoco.MjData`, 把每个阶段的 nav+L/R target_joints
  依次写入 qpos, 调 `mj_forward`, 读 `data.ncon` 统计接触对, 过滤同一条臂
  内部的自接触, 剩下的视为真实碰撞报告给 LLM。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import mujoco
import numpy as np

from ..pipeline.plan_executor import PlanExecutor


class FeedbackManager:
    """
    对 LLM plan 做 task + reach + collision 三级校验。

    Args:
        model:       原 MuJoCo MjModel (只读, 我们用它 spawn 一个临时 MjData)
        data:        原 MjData (只用来读当前 cart 位置作为起点)
        arm_offsets: {'L': np.array([0.35,0,0.35]), 'R': np.array([-0.35,0,0.35])}
        arm_yaws:    {'L': 0.0, 'R': np.pi}
        dof:         单臂自由度, UR5e = 6
    """

    # 单臂内部允许的"自接触": 相邻 link 之间
    ARM_LINK_SUFFIXES = [
        "shoulder_link", "upper_arm_link", "forearm_link",
        "wrist_1_link", "wrist_2_link", "wrist_3_link",
        "flange", "ee_link", "tool0",
        "robotiq_85_base_link",
        "left_outer_knuckle", "left_inner_knuckle",
        "left_outer_finger", "left_inner_finger",
        "right_outer_knuckle", "right_inner_knuckle",
        "right_outer_finger", "right_inner_finger",
    ]

    def __init__(
        self,
        model: "mujoco.MjModel",
        data: "mujoco.MjData",
        arm_offsets: Dict[str, np.ndarray],
        arm_yaws: Dict[str, float],
        dof: int = 6,
    ):
        self.model = model
        self.data = data  # 只读引用, 用来获取当前 cart 位置
        self.dof = dof

        # 复用现有 PlanExecutor 做 IK + reach 校验
        self._plan_executor = PlanExecutor(
            arm_offsets=arm_offsets, arm_yaws=arm_yaws
        )

        # 解析关键 qpos/ctrl 索引 (与 TaskRunner._resolve_indices 保持一致)
        jid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        self._cart_x_qpos = model.jnt_qposadr[jid("cart_x")]
        self._cart_y_qpos = model.jnt_qposadr[jid("cart_y")]
        self._cart_yaw_qpos = model.jnt_qposadr[jid("cart_yaw")]
        self._L_qpos_start = model.jnt_qposadr[jid("shoulder_pan_joint")]
        self._R_qpos_start = model.jnt_qposadr[jid("shoulder_pan_joint_R")]

        # 构建 L/R 臂各自的 body id 集合, 用来过滤自接触
        self._L_arm_body_ids = self._collect_arm_body_ids(model, r_side=False)
        self._R_arm_body_ids = self._collect_arm_body_ids(model, r_side=True)

    # ------------------------------------------------------------------
    # 身份识别
    # ------------------------------------------------------------------

    @classmethod
    def _collect_arm_body_ids(cls, model, r_side: bool) -> set:
        """
        通过 body 名称后缀 _R 区分左右臂 body。
        若 body 名以某个 ARM_LINK_SUFFIXES 匹配, 且 (带 _R 与否) 匹配目标侧, 则归入。
        """
        ids = set()
        for i in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
            has_R = name.endswith("_R") or name.endswith("_R_link")
            if r_side and not has_R:
                continue
            if (not r_side) and has_R:
                continue
            # 去掉 _R 再匹配关节后缀
            base = name[:-2] if has_R and name.endswith("_R") else name
            base = base.rsplit("_R_link", 1)[0] if "_R_link" in base else base
            for suffix in cls.ARM_LINK_SUFFIXES:
                if base.endswith(suffix) or name == suffix:
                    ids.add(i)
                    break
        return ids

    def _body_side(self, body_id: int) -> Optional[str]:
        if body_id in self._L_arm_body_ids:
            return "L"
        if body_id in self._R_arm_body_ids:
            return "R"
        return None

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def give_feedback(
        self, plan_dict: Dict, snapshot: Dict, task_instruction: str = ""
    ) -> Tuple[bool, str]:
        """
        检查 plan 是否可执行, 返回 (ok, feedback_str)。

        feedback_str 为空字符串表示全部通过; 否则为给 LLM 看的可读文本。
        """
        problems: List[str] = []

        # 1) Task 级语义校验 (廉价, 先做)
        task_problems = self._check_task_rules(plan_dict, task_instruction)
        problems.extend(task_problems)

        # 2) Reach + IK 结构校验 (复用 PlanExecutor.parse)
        phases, reach_problems = self._check_reach_and_structure(
            plan_dict, snapshot
        )
        problems.extend(reach_problems)

        # 3) Collision 校验 (需要 phases, 只有在 reach 通过时才有意义)
        if phases is not None and not reach_problems:
            collision_problems = self._check_collisions(phases, snapshot)
            problems.extend(collision_problems)

        if not problems:
            return True, ""

        feedback = self._format_feedback(plan_dict, problems)
        return False, feedback

    # ------------------------------------------------------------------
    # 1) Task 级规则
    # ------------------------------------------------------------------

    _INSPECTION_KEYWORDS = ["巡检", "仪表", "压力表", "读数", "观测"]

    def _check_task_rules(
        self, plan_dict: Dict, task_instruction: str
    ) -> List[str]:
        problems = []
        steps = plan_dict.get("plan", [])
        task = (task_instruction or "").strip()

        is_inspection = any(k in task for k in self._INSPECTION_KEYWORDS)

        l_grasping = False
        r_grasping = False

        for i, step in enumerate(steps):
            arm = step.get("arm", "?")
            skill = step.get("skill", "?")
            params = step.get("params", {}) or {}

            # 规则: 巡检类任务里 MoveSkill 应该是 R 臂
            if is_inspection and skill in ("MoveSkill",) and arm == "L":
                problems.append(
                    f"Step {i+1}: 巡检任务要求压力表观测由 R 臂完成, "
                    f"但这里分给了 L 臂。请改为 arm='R'。"
                )

            # 规则: MoveSkill 的 target_pos z 不能超过 0.9
            if skill in ("MoveSkill", "TranslateSkill", "InsertSkill",
                         "ExtractSkill"):
                tp = params.get("target_pos")
                if tp and len(tp) >= 3 and float(tp[2]) > 0.9:
                    problems.append(
                        f"Step {i+1} ({arm}臂 {skill}): target_pos 的 z="
                        f"{tp[2]:.3f} 超过安全上限 0.9m, 请降低目标高度。"
                    )

            # 规则: 同一条臂 PICK 前必须 RELEASE, 避免连续 close 两次
            if skill == "GraspSkill":
                action = params.get("action")
                is_l = (arm == "L")
                grasp_flag = l_grasping if is_l else r_grasping
                if action == "close" and grasp_flag:
                    problems.append(
                        f"Step {i+1}: {arm}臂在没有 open 夹爪的情况下连续 close, "
                        f"应该先 GraspSkill(action='open') 释放前一个物体。"
                    )
                if action == "open" and not grasp_flag:
                    # 非致命: 只警告 (允许空夹释放)
                    pass
                if action == "close":
                    if is_l: l_grasping = True
                    else:    r_grasping = True
                elif action == "open":
                    if is_l: l_grasping = False
                    else:    r_grasping = False

        return problems

    # ------------------------------------------------------------------
    # 2) Reach + IK 结构校验
    # ------------------------------------------------------------------

    def _check_reach_and_structure(
        self, plan_dict: Dict, snapshot: Dict
    ) -> Tuple[Optional[List[Dict]], List[str]]:
        """
        复用 PlanExecutor.parse —— 它内部会:
          - 对每个 MoveSkill 做 IK
          - 对超出 0.85m 范围的 target 抛 ValueError
          - 对未知 skill 名抛 ValueError
        我们捕获并转成 feedback。
        """
        cart_pos = self._current_cart_xy(snapshot)

        try:
            parsed = self._plan_executor.parse(plan_dict, cart_pos)
            phases = parsed.get("phases", [])
            return phases, []
        except ValueError as e:
            return None, [f"Reach/IK 校验失败: {e}"]
        except Exception as e:
            return None, [f"Plan 解析失败: {type(e).__name__}: {e}"]

    def _current_cart_xy(self, snapshot: Dict) -> np.ndarray:
        """从 snapshot 或 self.data 读当前小车 xy。"""
        planner_state = snapshot.get("planner_state", {}) or {}
        cart = planner_state.get("cart", {}) or {}
        pos = cart.get("position")
        if pos and len(pos) >= 2:
            return np.array(pos[:2], dtype=float)
        # 回退: 直接读 data
        return np.array([
            self.data.qpos[self._cart_x_qpos],
            self.data.qpos[self._cart_y_qpos],
        ])

    # ------------------------------------------------------------------
    # 3) 碰撞校验: 把 phases 回放到一份临时 MjData 上
    # ------------------------------------------------------------------

    INIT_Q = np.array([0, 0, np.pi / 2, 0, -np.pi / 2, 0])

    def _check_collisions(
        self, phases: List[Dict], snapshot: Dict
    ) -> List[str]:
        """
        对每个 phase 依次做:
          - 如果有 nav, 把 cart qpos 设到 target
          - 对该 phase 内 L/R 的每一个 target_joints (按 skill 顺序扫描),
            写入 qpos, mj_forward, 收集接触对
          - 过滤同臂自接触, 剩余报告给 LLM
        注意: 为了简化, 我们只把 "含 target_joints 的 skill 的最后一帧" 做检查,
              不做插值扫描 —— 这与 RoCo 的做法一致。
        """
        data = mujoco.MjData(self.model)
        # 从当前状态拷贝 qpos / qvel / ctrl, 保证非臂/非 cart 关节状态正确
        data.qpos[:] = self.data.qpos[:]
        data.qvel[:] = self.data.qvel[:]
        data.ctrl[:] = self.data.ctrl[:]

        # 先把两条臂复位到 INIT_Q
        data.qpos[self._L_qpos_start:self._L_qpos_start + self.dof] = self.INIT_Q
        data.qpos[self._R_qpos_start:self._R_qpos_start + self.dof] = self.INIT_Q

        problems: List[str] = []

        for p_idx, phase in enumerate(phases):
            nav = phase.get("nav")
            if nav and nav.get("target"):
                tgt = nav["target"]
                data.qpos[self._cart_x_qpos] = float(tgt[0])
                data.qpos[self._cart_y_qpos] = float(tgt[1])
                if nav.get("yaw") is not None:
                    data.qpos[self._cart_yaw_qpos] = float(nav["yaw"])

            # 收集该 phase 中每条臂所有出现过的 target_joints (按顺序)
            L_targets = [s.target_joints for s in phase.get("L", [])
                         if getattr(s, "target_joints", None) is not None]
            R_targets = [s.target_joints for s in phase.get("R", [])
                         if getattr(s, "target_joints", None) is not None]

            # 同步扫描: 取两条臂的最大步数, 不足的用最后一帧 hold
            n_steps = max(len(L_targets), len(R_targets))
            for step_i in range(n_steps):
                if L_targets:
                    q_L = L_targets[min(step_i, len(L_targets) - 1)]
                    data.qpos[self._L_qpos_start:self._L_qpos_start + self.dof] = q_L
                if R_targets:
                    q_R = R_targets[min(step_i, len(R_targets) - 1)]
                    data.qpos[self._R_qpos_start:self._R_qpos_start + self.dof] = q_R

                mujoco.mj_forward(self.model, data)

                hits = self._collect_contacts(data)
                if hits:
                    problems.append(
                        f"Phase {p_idx} 第 {step_i+1} 帧检测到碰撞: "
                        + "; ".join(hits)
                    )
                    # 单 phase 内只报一次碰撞即可, 避免反馈文本爆炸
                    break

        return problems

    def _collect_contacts(self, data) -> List[str]:
        """
        扫描 data.contact 列表, 返回需要报告给 LLM 的碰撞对描述 (字符串列表)。
        过滤规则:
          - 两 geom 属于同一条臂 (L-L 或 R-R) -> 自接触, 忽略
          - 两 geom 都是 cart / 地面 / 其它静态部分 -> 忽略 (未绑定任何臂)
          - 其余视为问题, 报告 body 名
        """
        out = []
        seen = set()
        for i in range(data.ncon):
            con = data.contact[i]
            g1, g2 = int(con.geom1), int(con.geom2)
            b1 = int(self.model.geom_bodyid[g1])
            b2 = int(self.model.geom_bodyid[g2])

            side1 = self._body_side(b1)
            side2 = self._body_side(b2)

            # 同一条臂内部接触 -> 跳过
            if side1 is not None and side1 == side2:
                continue

            # 两侧都不在任一条臂上 -> 与本次规划无关, 跳过
            if side1 is None and side2 is None:
                continue

            # 去重
            key = tuple(sorted((b1, b2)))
            if key in seen:
                continue
            seen.add(key)

            name1 = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_BODY, b1) or f"body#{b1}"
            name2 = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_BODY, b2) or f"body#{b2}"

            # 排序显示: 臂那一侧优先放前面, 环境放后面
            if side1 is None:
                name1, name2 = name2, name1
                side1, side2 = side2, side1

            if side2 is not None:
                out.append(
                    f"{side1}臂 {name1} 与 {side2}臂 {name2} 相互碰撞"
                )
            else:
                out.append(f"{side1}臂 {name1} 与环境物体 {name2} 碰撞")
        return out

    # ------------------------------------------------------------------
    # 反馈格式化
    # ------------------------------------------------------------------

    @staticmethod
    def _format_feedback(plan_dict: Dict, problems: List[str]) -> str:
        # 打印 plan 摘要, 让 LLM 能对应修改
        summary_lines = []
        for step in plan_dict.get("plan", []):
            idx = step.get("step", "?")
            arm = step.get("arm", "?")
            skill = step.get("skill", "?")
            params = step.get("params", {}) or {}
            summary_lines.append(
                f"  [{idx}] {arm}: {skill}({params})"
            )
        summary = "\n".join(summary_lines) if summary_lines else "  (empty)"

        body = "\n".join(f"- {p}" for p in problems)
        return (
            "Previous Plan:\n"
            f"{summary}\n\n"
            "Problems detected:\n"
            f"{body}"
        )
