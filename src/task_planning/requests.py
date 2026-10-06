"""Validate semantic goals, and adapt recognizable complete legacy plans."""

from collections import Counter
from copy import deepcopy
from math import isfinite, pi
import re

import numpy as np

from .models import Goal, PlanningRequest, Stage


def scene_objects(snapshot):
    state = snapshot.get("planner_state") or snapshot.get("world_state") or {}
    return state.get("objects", {})


def object_point(objects, name, key=None):
    obj = objects.get(name, {})
    point = obj.get("keypoints", {}).get(key, {}) if key else obj
    value = point.get("position", point.get("pos"))
    if value is None and key is None:
        value = obj.get("pose", {}).get("pos", obj.get('light', {}).get('pos'))
    if value is None or len(value) != 3 or not np.all(np.isfinite(value)):
        raise ValueError(f"场景缺少 {name}/{key or 'position'} 的有效位置")
    return np.asarray(value, dtype=float)


def validate_instruction(goals, objects, instruction):
    """Check explicit object lists/all-object requests; LLM still parses semantics."""
    required = set()
    digits = {"一": "1", "二": "2", "三": "3"}
    for prefix, noun in (("valve", "阀门"), ("gauge", "压力表")):
        ids = re.findall(rf"([123一二三])\s*号?\s*{noun}|{noun}\s*[_ ]?([123一二三])", instruction)
        for pair in ids:
            value = next(v for v in pair if v)
            required.add(f"{prefix}_{digits.get(value, value)}")
        for group in re.findall(rf"([123一二三、，,和及与\s]+)号?\s*{noun}", instruction):
            for value in re.findall(r"[123一二三]", group):
                required.add(f"{prefix}_{digits.get(value, value)}")
        required.update(re.findall(rf"{prefix}_[123]", instruction))
        if (re.search(rf"(?:所有|全部|每个).*?{noun}", instruction)
                or (prefix == "gauge" and "巡检压力表" in instruction and not ids)):
            required.update(name for name in objects if name.startswith(prefix + "_"))
    covered = {g.target for g in goals}
    missing = required - covered
    if missing:
        raise ValueError(f"计划遗漏指令中的必做目标: {', '.join(sorted(missing))}")
    if "遮光" in instruction and not re.search(r"不(?:需要|要)?遮光", instruction):
        for goal in goals:
            if goal.operation == "rotate" and goal.shade != "required":
                raise ValueError(f"指令要求遮光，目标 {goal.id} 必须设置 shade='required'")
    angle = re.search(r"([-+]?\d+(?:\.\d+)?)\s*(?:度|°)", instruction)
    if angle:
        expected = abs(float(angle.group(1))) * pi / 180
        if any(g.operation == "rotate" and abs(abs(g.angle) - expected) > 0.02 for g in goals):
            raise ValueError("旋转角度与任务指令不一致，angle 使用弧度")


def _nav(value):
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("stage.nav 必须为对象或 null")
    target = value.get("target")
    if not isinstance(target, (list, tuple)) or len(target) != 2 or not np.all(np.isfinite(target)):
        raise ValueError("NavSkill.target 必须为两个有限数值 [x,y]")
    yaw = value.get("yaw")
    if yaw is not None and not isfinite(yaw):
        raise ValueError("NavSkill.yaw 必须为有限弧度值")
    return {"target": list(target), "yaw": yaw}


def semantic_request(result, snapshot, instruction=""):
    objects = scene_objects(snapshot)
    if not isinstance(result.get("goals"), list) or not isinstance(result.get("stages"), list):
        raise ValueError("语义计划必须包含 goals 和 stages 数组")
    goals = []
    for item in result["goals"]:
        if not isinstance(item, dict):
            raise ValueError("每个 goal 必须为对象")
        goal_id, target, operation = item.get("id"), item.get("object"), item.get("operation")
        if not isinstance(goal_id, str) or not goal_id.strip():
            raise ValueError("goal.id 必须为非空字符串")
        expected_kind = {"rotate": "valve", "inspect": "gauge"}.get(operation)
        if expected_kind is None or objects.get(target, {}).get("kind") != expected_kind:
            raise ValueError(f"不支持的目标/操作组合: {target}/{operation}；当前支持阀门 rotate 和压力表 inspect")
        angle = item.get("angle", 0.0)
        if not isinstance(angle, (int, float)) or not isfinite(angle) or (operation == "rotate" and abs(angle) < 1e-8):
            raise ValueError(f"{goal_id}: rotate.angle 必须为非零有限弧度值")
        shade = item.get("shade", "none")
        if shade not in ("none", "optional", "required") or (operation == "inspect" and shade != "none"):
            raise ValueError(f"{goal_id}: shade 必须为 none/optional/required，inspect 暂不支持辅助")
        arm = item.get("preferred_arm", "R" if operation == "inspect" else "L")
        if arm not in ("L", "R"):
            raise ValueError(f"{goal_id}: preferred_arm 必须为 L/R")
        goals.append(Goal(goal_id, target, operation, float(angle), shade, arm))
    if len({g.id for g in goals}) != len(goals):
        raise ValueError("goal.id 不能重复")
    validate_instruction(goals, objects, instruction)
    stages, seen = [], []
    for item in result["stages"]:
        if not isinstance(item, dict) or not isinstance(item.get("goals"), list):
            raise ValueError("每个 stage 必须包含 goals ID 数组")
        ids = item["goals"]
        if any(not isinstance(i, str) for i in ids):
            raise ValueError("stage.goals 只能引用 goal.id")
        parallel = item.get("parallel", False)
        if not isinstance(parallel, bool) or (parallel and len(ids) > 2):
            raise ValueError("parallel 必须为布尔值；双臂并行阶段最多两个主任务")
        nav = _nav(item.get("nav"))
        seen.extend(ids)
        # Sequential goals become separate execution slots without changing nav order.
        groups = [ids] if parallel or not ids else [[i] for i in ids]
        for index, group in enumerate(groups):
            stages.append(Stage(len(stages), nav if index == 0 else None, tuple(group), parallel))
    if Counter(seen) != Counter(g.id for g in goals):
        raise ValueError("每个必做目标必须且只能在 stages 中出现一次")
    if not stages or any(not s.goals and s.nav is None for s in stages):
        raise ValueError("计划不能为空，空阶段必须包含导航")
    return PlanningRequest(tuple(goals), tuple(stages))


def legacy_request(result, snapshot, instruction=""):
    """Keep every step of valve bundles. Unknown legacy skills stay fixed."""
    objects = scene_objects(snapshot)
    phases = [{"nav": None, "L": [], "R": []}]
    for step in result.get("plan", []):
        if step.get("skill") == "NavSkill" or "nav" in step:
            if phases[-1]["nav"] or phases[-1]["L"] or phases[-1]["R"]:
                phases.append({"nav": None, "L": [], "R": []})
            phases[-1]["nav"] = _nav(step.get("params", {}))
        else:
            arm = step.get("arm", "L")
            if arm not in ("L", "R"):
                raise ValueError("未知机械臂，必须为 L/R")
            phases[-1][arm].append(deepcopy(step))
    goals, stages = [], []
    required_shade = "遮光" in instruction and not re.search(r"不(?:需要|要)?遮光", instruction)
    for phase in phases:
        phase_goals = []
        helper = any("遮光" in s.get("description", "") or s.get("role") == "shade"
                     for s in phase["R"])
        if helper and any(s.get('skill') in ('RotateSkill', 'InsertSkill', 'ExtractSkill') for s in phase['R']):
            raise ValueError('R 臂不能在同一遮光阶段兼任主操作，请拆分 stages')
        for arm in ("L", "R"):
            steps = phase[arm]
            if arm == "R" and helper:
                continue
            start = 0
            while start < len(steps):
                bundle = steps[start:start + 6]
                names = [s.get("skill") for s in bundle]
                if names == ["MoveSkill", "MoveSkill", "GraspSkill", "RotateSkill", "GraspSkill", "MoveSkill"]:
                    if (bundle[2].get("params", {}).get("action") != "close"
                            or bundle[4].get("params", {}).get("action") != "open"):
                        raise ValueError("阀门动作块必须先夹取，再旋转、释放、回退")
                    target_pos = np.asarray(bundle[1].get("params", {}).get("target_pos"), dtype=float)
                    distances = [(float(np.linalg.norm(target_pos - object_point(objects, name, "handle_center"))), name)
                                 for name, obj in objects.items() if obj.get("kind") == "valve"]
                    if not distances or min(distances)[0] > 0.12:
                        raise ValueError("无法把阀门动作块关联到场景对象，请输出 goals/stages")
                    target = min(distances)[1]
                    goal_id = f"g{len(goals) + 1}"
                    shade = "required" if helper or required_shade else "none"
                    goals.append(Goal(goal_id, target, "rotate", -float(bundle[3].get("params", {}).get("angle", pi)),
                                      shade, arm, tuple(bundle)))
                    phase_goals.append(goal_id)
                    start += 6
                else:
                    inspect_steps = steps[start:start + 2]
                    if len(inspect_steps) == 2 and all(s.get('skill') == 'MoveSkill' for s in inspect_steps):
                        position = np.asarray(inspect_steps[0].get('params', {}).get('target_pos'), dtype=float)
                        distances = [(float(np.linalg.norm(position - object_point(objects, name, 'inspect_point'))), name)
                                     for name, obj in objects.items() if obj.get('kind') == 'gauge']
                        if distances and min(distances)[0] <= 0.12:
                            goal_id = f'g{len(goals) + 1}'
                            goals.append(Goal(goal_id, min(distances)[1], 'inspect', preferred_arm=arm,
                                              steps=tuple(inspect_steps)))
                            phase_goals.append(goal_id)
                            start += 2
                            continue
                    # Explicit semantic goals are needed to optimize unfamiliar action sequences.
                    return None
        if helper and not phase_goals:
            raise ValueError("遮光辅助阶段缺少主操作目标")
        groups = [[i] for i in phase_goals] or [[]]
        for index, group in enumerate(groups):
            stages.append(Stage(len(stages), phase["nav"] if index == 0 else None, tuple(group)))
    validate_instruction(goals, objects, instruction)
    if not stages or all(not s.nav and not s.goals for s in stages):
        raise ValueError("LLM 未返回可执行目标或计划")
    return PlanningRequest(tuple(goals), tuple(stages), "legacy_bundles")


def make_request(result, snapshot, instruction=""):
    if "goals" in result or "stages" in result:
        return semantic_request(result, snapshot, instruction)
    return legacy_request(result, snapshot, instruction)
