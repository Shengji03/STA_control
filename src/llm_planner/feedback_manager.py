"""
Feedback Manager —— 对 LLM 输出的 plan 做多级校验, 生成可回注 prompt 的反馈。

灵感来自 RoCo (Mandi et al., 2023) 的 prompting/feedback.py, 但适配到
STA_control 的 Skill JSON 体系:

    1. Task 级语义校验 (根据任务指令和 prompts.py 中的硬规则)
    2. Reach 级校验 (复用 PlanExecutor 的 IK + 距离检查, 捕获 ValueError)
    3. Collision 级校验 (在临时 MuJoCo data 上采样导航、转向与关节轨迹)

关键设计:
- Reach 校验直接复用 `PlanExecutor.parse`, 它内部已对 target_pos 做 IK +
  0.85m reach 距离判断, 失败时抛 ValueError — 我们捕获并格式化成 feedback。
- Collision 校验使用独立 MjData，不修改真实仿真。按显式执行模式回放，
  并把夹取后的遮光板作为随 R 臂运动的物体检查。实际夹取仍由摩擦接触实现，
  运行后的效果监测独立核验其是否保持、遮光和放回。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import mujoco
import numpy as np

from ..pipeline.plan_executor import PlanExecutor
from ..config.robot import INITIAL_JOINTS


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
        collision_samples: int = 8,
    ):
        self.model = model
        self.data = data  # 只读引用, 用来获取当前 cart 位置
        self.dof = dof
        self.collision_samples = collision_samples

        # 复用现有 PlanExecutor 做 IK + reach 校验
        self._plan_executor = PlanExecutor(
            arm_offsets=arm_offsets, arm_yaws=arm_yaws, verbose=False,
        )

        # 解析关键 qpos/ctrl 索引 (与 TaskRunner._resolve_indices 保持一致)
        jid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        self._cart_x_qpos = model.jnt_qposadr[jid("cart_x")]
        self._cart_y_qpos = model.jnt_qposadr[jid("cart_y")]
        self._cart_yaw_qpos = model.jnt_qposadr[jid("cart_yaw")]
        self._L_qpos_start = model.jnt_qposadr[jid("shoulder_pan_joint")]
        self._R_qpos_start = model.jnt_qposadr[jid("shoulder_pan_joint_R")]
        self._shade_board = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'shade_board')
        board_joint = jid('shade_board_joint')
        self._shade_qpos = int(model.jnt_qposadr[board_joint]) if board_joint >= 0 else None
        self._R_flange = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'flange_R')

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
        root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                'ur5e_base_R' if r_side else 'ur5e_base')
        if root >= 0:
            ids = set()
            for body in range(model.nbody):
                ancestor = body
                while ancestor > 0:
                    if ancestor == root:
                        ids.add(body)
                        break
                    ancestor = int(model.body_parentid[ancestor])
            return ids
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

        if 'goals' in plan_dict and 'execution_phases' not in plan_dict:
            from ..task_planning.requests import semantic_request
            try:
                semantic_request(plan_dict, snapshot, task_instruction)
                return True, ''
            except (ValueError, TypeError, KeyError) as exc:
                return False, f'目标/阶段结构校验失败: {exc}'

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
        goal_operations = {g['id']: g.get('operation') for g in plan_dict.get('goals', [])}

        l_grasping = False
        r_grasping = False

        for i, step in enumerate(steps):
            arm = step.get("arm", "?")
            skill = step.get("skill", "?")
            params = step.get("params", {}) or {}

            # 规则: 巡检类任务里 MoveSkill 应该是 R 臂
            inspection_step = (goal_operations.get(step.get('goal_id')) == 'inspect'
                               if goal_operations else is_inspection)
            if inspection_step and skill in ("MoveSkill",) and arm == "L":
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
            parsed = self._plan_executor.parse(plan_dict, cart_pos, self._current_cart_yaw(snapshot))
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

    def _current_cart_yaw(self, snapshot):
        cart = snapshot.get('planner_state', {}).get('cart', {})
        return float(cart.get('yaw', self.data.qpos[self._cart_yaw_qpos]))

    # ------------------------------------------------------------------
    # 3) 碰撞校验: 把 phases 回放到一份临时 MjData 上
    # ------------------------------------------------------------------

    INIT_Q = np.array(INITIAL_JOINTS)

    def _check_collisions(
        self, phases: List[Dict], snapshot: Dict
    ) -> List[str]:
        """
        按执行模式采样导航、转向、关节轨迹、腕关节旋转和返回初始位姿。
        遮光按 setup -> hold/main -> cleanup 回放；并行阶段按技能时长采样。
        携带物的刚性相对位姿只用于几何预测，不强制实际仿真夹取。
        延续同臂自接触过滤，仅允许目标物体与指定抓取臂手指的操作接触。
        此采样检查不构成连续轨迹的碰撞证明。
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

        held_board = None
        for p_idx, phase in enumerate(phases):
            nav = phase.get("nav")
            if nav and nav.get("target"):
                tgt = nav["target"]
                start = data.qpos[[self._cart_x_qpos, self._cart_y_qpos]].copy()
                yaw_start = float(data.qpos[self._cart_yaw_qpos])
                # The runtime moves first and then turns; check the same order.
                for fraction in np.linspace(0, 1, self.collision_samples + 1):
                    data.qpos[[self._cart_x_qpos, self._cart_y_qpos]] = start + fraction * (np.asarray(tgt) - start)
                    mujoco.mj_forward(self.model, data)
                    hits = self._collect_contacts(data)
                    if hits:
                        problems.append(f'Phase {p_idx} 导航路径碰撞: ' + '; '.join(hits))
                        break
                if nav.get('yaw') is not None:
                    for fraction in np.linspace(0, 1, self.collision_samples + 1):
                        data.qpos[self._cart_yaw_qpos] = yaw_start + fraction * (float(nav['yaw']) - yaw_start)
                        mujoco.mj_forward(self.model, data)
                        hits = self._collect_contacts(data)
                        if hits:
                            problems.append(f'Phase {p_idx} 转向碰撞: ' + '; '.join(hits))
                            break

            allowed = phase.get('contact_targets', {})
            mode = phase.get('mode', 'parallel')
            if mode == 'assist_R_then_L':
                segments = [('R', phase['R'], False), ('L', phase['L'], True),
                            ('R', phase['R_cleanup'], True)]
            else:
                segments = [(arm, phase.get(arm, []), True) for arm in ('L', 'R')]
            # Parallel main tasks are sampled with their actual duration profiles.
            if mode == 'parallel':
                hits = self._check_parallel(data, phase, allowed)
                if hits:
                    problems.append(f'Phase {p_idx} 并行轨迹碰撞: ' + '; '.join(hits))
                continue
            for arm, skills, return_home in segments:
                address = self._L_qpos_start if arm == 'L' else self._R_qpos_start
                targets = self._joint_targets(skills, data.qpos[address:address + self.dof])
                if return_home and skills:
                    targets.append((self.INIT_Q, 3.0))
                replay = [(target, duration, skills[index] if index < len(skills) else None)
                          for index, (target, duration) in enumerate(targets)]
                for target, _duration, skill in replay:
                    start = data.qpos[address:address + self.dof].copy()
                    for fraction in np.linspace(0, 1, self.collision_samples + 1):
                        data.qpos[address:address + self.dof] = start + fraction * (target - start)
                        mujoco.mj_forward(self.model, data)
                        if held_board is not None:
                            self._place_held_board(data, held_board)
                        hits = self._collect_contacts(data, allowed)
                        if hits:
                            problems.append(f'Phase {p_idx} {arm}臂轨迹碰撞: ' + '; '.join(hits))
                            break
                    if problems:
                        break
                    if (arm == 'R' and getattr(skill, 'action', None) == 'close'
                            and 'shade_board' in allowed.get('R', []) and self._shade_qpos is not None):
                        rotation = data.xmat[self._R_flange].reshape(3, 3)
                        held_board = (rotation.T @ (data.xpos[self._shade_board] - data.xpos[self._R_flange]),
                                      rotation.T @ data.xmat[self._shade_board].reshape(3, 3))
                    elif arm == 'R' and getattr(skill, 'action', None) == 'open':
                        held_board = None
                if problems:
                    break
            if problems:
                break

        return problems

    def _place_held_board(self, data, relative_pose):
        """Replay a rigid grasp in scratch data; never modify the live simulation.

        The real grasp still uses friction contacts. This geometric model makes
        the carried board part of collision checking; effects verify the grasp.
        """
        rotation = data.xmat[self._R_flange].reshape(3, 3)
        position = data.xpos[self._R_flange] + rotation @ relative_pose[0]
        board_rotation = rotation @ relative_pose[1]
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, board_rotation.reshape(9))
        data.qpos[self._shade_qpos:self._shade_qpos + 3] = position
        data.qpos[self._shade_qpos + 3:self._shade_qpos + 7] = quat
        mujoco.mj_forward(self.model, data)

    def _joint_targets(self, skills, initial):
        targets, position = [], np.array(initial, dtype=float)
        for skill in skills:
            if getattr(skill, 'target_joints', None) is not None:
                position = skill.target_joints.copy()
            elif hasattr(skill, 'angle'):
                position = position.copy()
                position[skill.joint_index] += skill.angle
            targets.append((position.copy(), float(getattr(skill, 'duration', getattr(skill, 'wait_time', 0.8)))))
        return targets

    def _check_parallel(self, data, phase, allowed):
        profiles = {}
        for arm, address in (('L', self._L_qpos_start), ('R', self._R_qpos_start)):
            initial = data.qpos[address:address + self.dof].copy()
            targets = self._joint_targets(phase[arm], initial) + [(self.INIT_Q, 3.0)]
            profiles[arm] = (initial, targets, address)
        end = max(sum(duration for _, duration in profile[1]) for profile in profiles.values())
        boundaries = {0.0, end}
        for _initial, targets, _address in profiles.values():
            t = 0.0
            for _, duration in targets:
                boundaries.update(np.linspace(t, t + duration, self.collision_samples + 1).tolist())
                t += duration
        for time in sorted(boundaries):
            for initial, targets, address in profiles.values():
                previous, elapsed = initial, 0.0
                for target, duration in targets:
                    if time <= elapsed + duration:
                        fraction = np.clip((time - elapsed) / max(duration, 1e-8), 0, 1)
                        smooth = fraction * fraction * (3 - 2 * fraction)
                        position = previous + smooth * (target - previous)
                        break
                    previous, elapsed = target, elapsed + duration
                else:
                    position = previous
                data.qpos[address:address + self.dof] = position
            mujoco.mj_forward(self.model, data)
            hits = self._collect_contacts(data, allowed)
            if hits:
                return hits
        return []

    def _intended_contact(self, body1, body2, allowed):
        for arm_body, target_body in ((body1, body2), (body2, body1)):
            side = self._body_side(arm_body)
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, arm_body) or ''
            if side is None or not any(word in name for word in ('finger', 'knuckle', 'pad')):
                continue
            targets = {('valve_body_' + target.split('_')[-1]) if target.startswith('valve_') else target
                       for target in allowed.get(side, [])}
            while target_body > 0:
                if mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, target_body) in targets:
                    return True
                target_body = int(self.model.body_parentid[target_body])
        return False

    def _collect_contacts(self, data, allowed=None) -> List[str]:
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

            if allowed and self._intended_contact(b1, b2, allowed):
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
