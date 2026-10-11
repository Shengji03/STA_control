"""Real-prefix skill benchmarks and physical effects for the complete CARLA suite."""

import io
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict, dataclass
from math import isfinite, pi

import mujoco
import numpy as np

from experiments.long_sequence_task import LongSequenceTask
from src.config.paths import scene_path
from src.controller.arm_torque import compute_arm_torque
from src.controller.control_metrics import ControlMetrics
from src.controller.control_parameters import SkillParameterBank
from src.pipeline.task_runner import TaskRunner
from src.skills import GraspSkill, RotateSkill, SkillType


@dataclass(frozen=True)
class Scenario:
    name: str
    family: str = "valve_L"
    scene: str = "scene5_glare.xml"
    speed: float = 1.0
    payload: float = 1.0
    resistance: float = 1.0
    grasp: float = 1.0
    angle: float = pi
    target: str = "valve_1"
    cart_offset: float = 0.0
    order: tuple = (0, 1, 2)
    arm_order: tuple = ("L", "R")


@dataclass(frozen=True)
class Profile:
    id: str
    arm: str
    skill: str
    domain: str
    action: str | None = None

    def matches(self, label, skill, scene):
        return (
            label == self.arm
            and skill is not None
            and skill.skill_type.name == self.skill
            and (scene == "scene3.xml") == (self.domain == "assembly")
            and (self.action is None or getattr(skill, "action", None) == self.action)
        )

    def scope(self):
        scope = {
            "scenes": ["scene3.xml"]
            if self.domain == "assembly"
            else ["scene4_pipeline.xml", "scene5_glare.xml"]
        }
        if self.action:
            scope["actions"] = [self.action]
        if self.domain == "assembly":
            durations = {
                "MOVE": (1.9, 3.15),
                "INSERT": (3.8, 4.2),
                "EXTRACT": (2.85, 3.15),
                "TRANSLATE": (3.8, 4.2),
                "ROTATE": (4.75, 5.25),
                "GRASP": (0.776, 0.824) if self.action == "close" else (0.485, 0.515),
            }
            low, high = durations[self.skill]
            scope.update(
                duration_min=low - 1e-6,
                duration_max=high + 1e-6,
                objects=["pin_a", "block", "pin_b"]
                if self.skill in ("MOVE", "GRASP")
                else ["pin_a"]
                if self.skill in ("INSERT", "ROTATE")
                else ["pin_b"]
                if self.skill == "EXTRACT"
                else ["block"],
            )
            if self.skill == "ROTATE":
                scope.update(
                    angle_min=2 * pi - 0.01, angle_max=2 * pi + 0.01, angle_sign=1
                )
        if self.domain == "mobile":
            scope.update(
                workflows=[
                    "valve_L",
                    "valve_R",
                    "shade",
                    "two_valves",
                    "repeat",
                    "inspect",
                    "rotate_inspect",
                ],
                cart_poses=[
                    [0.45, 0, 0],
                    [3.5, 0, 0],
                    [1.0, 0, 0],
                    [4.05, 0, 0],
                    [-0.1, 0, 0],
                    [2.8, 0, 0],
                ],
                cart_tolerance=[0.05, 0.05, 0.05],
            )
            if self.skill == "MOVE":
                scope.update(duration_min=2.25 - 1e-6, duration_max=4.4 + 1e-6)
            if self.skill == "GRASP":
                scope.update(
                    duration_min=0.68 - 1e-6,
                    duration_max=(1.38 if self.action == "close" else 0.92) + 1e-6,
                )
        if self.domain == "mobile" and self.skill == "ROTATE":
            scope.update(
                objects=["valve_1", "valve_2"],
                angle_min=pi / 2 - 0.02,
                angle_max=pi + 0.02,
                angle_sign=-1,
                duration_min=4.0,
                duration_max=6.0,
            )
        return scope


PROFILES = tuple(
    [
        Profile(f"assembly_L_{kind}", "L", kind, "assembly")
        for kind in ("MOVE", "INSERT", "EXTRACT", "TRANSLATE", "ROTATE")
    ]
    + [
        Profile(f"assembly_L_GRASP_{action}", "L", "GRASP", "assembly", action)
        for action in ("close", "open")
    ]
    + [
        Profile(f"mobile_{arm}_{kind}", arm, kind, "mobile")
        for arm in ("L", "R")
        for kind in ("MOVE", "ROTATE")
    ]
    + [
        Profile(f"mobile_{arm}_GRASP_{action}", arm, "GRASP", "mobile", action)
        for arm in ("L", "R")
        for action in ("close", "open")
    ]
)


def make_bank(parameters):
    from src.skills.base_skill import SKILL_STA_PARAMS

    entries = {}
    for profile in PROFILES:
        if profile.id not in parameters:
            continue
        key = f"{profile.arm}:{profile.skill}"
        entries.setdefault(key, {"profiles": []})["profiles"].append(
            {
                "id": profile.id,
                "baseline": SKILL_STA_PARAMS[SkillType[profile.skill]],
                "ratios": list(parameters[profile.id]),
                "scope": profile.scope(),
            }
        )
    if "mobile_R_MOVE_cleanup" in parameters:
        profile = Profile("mobile_R_MOVE_cleanup", "R", "MOVE", "mobile")
        entries.setdefault("R:MOVE", {"profiles": []})["profiles"].insert(
            0,
            {
                "id": profile.id,
                "baseline": SKILL_STA_PARAMS[SkillType.MOVE],
                "ratios": list(parameters[profile.id]),
                "scope": {
                    **profile.scope(),
                    "workflows": ["shade"],
                    "stages": ["assist_cleanup"],
                },
            },
        )
    return SkillParameterBank(entries)


def plan_for(case):
    arm = "R" if case.family == "valve_R" else "L"

    def goal(gid, obj, preferred=arm, shade="none"):
        return {
            "id": gid,
            "object": obj,
            "operation": "rotate",
            "angle": case.angle,
            "preferred_arm": preferred,
            "shade": shade,
        }

    cart = (
        (0.45 if case.target == "valve_1" else 3.5)
        + (0.55 if arm == "R" else 0)
        + case.cart_offset
    )
    if case.family == "two_valves":
        first, second = case.arm_order
        goals = [goal("v1", "valve_1", first), goal("v2", "valve_2", second)]
        stages = [
            {
                "nav": {
                    "target": [(0.45 if first == "L" else 1.0) + case.cart_offset, 0]
                },
                "goals": ["v1"],
            },
            {
                "nav": {
                    "target": [(3.5 if second == "L" else 4.05) + case.cart_offset, 0]
                },
                "goals": ["v2"],
            },
        ]
    elif case.family == "repeat":
        goals = [
            goal(gid, "valve_1", chosen)
            for gid, chosen in zip(("v1a", "v1b"), case.arm_order)
        ]
        stages = [
            {
                "nav": {
                    "target": [(0.45 if chosen == "L" else 1.0) + case.cart_offset, 0]
                },
                "goals": [g["id"]],
            }
            for g, chosen in zip(goals, case.arm_order)
        ]
    elif case.family in ("inspect", "rotate_inspect"):
        inspect = {
            "id": "i1",
            "object": "gauge_1",
            "operation": "inspect",
            "preferred_arm": "R",
        }
        goals, stages = [], []
        if case.family == "rotate_inspect":
            goals.append(goal("v1", "valve_1", "L"))
            stages.append(
                {"nav": {"target": [0.45 + case.cart_offset, 0]}, "goals": ["v1"]}
            )
        goals.append(inspect)
        stages.append(
            {"nav": {"target": [-0.1 + case.cart_offset, 0]}, "goals": ["i1"]}
        )
    else:
        goals = [
            goal(
                "v1", case.target, arm, "required" if case.family == "shade" else "none"
            )
        ]
        stages = [{"nav": {"target": [cart, 0]}, "goals": ["v1"]}]
    return {"goals": goals, "stages": stages}


class AssemblyRunner:
    """Headless execution of the actual assembly skills, with measured object goals."""

    def __init__(self, case, bank):
        path = scene_path("scene3")
        spec = mujoco.MjSpec.from_file(str(path))
        for name in ("pin_geom", "pin_b_geom", "block_geom"):
            spec.geom(name).mass *= case.payload
        self.task = LongSequenceTask(
            str(path), enable_ddpg=False, sta_parameters=bank, model=spec.compile()
        )
        self.model, self.data = self.task.model, self.task.data
        m, d, task = self.model, self.data, self.task
        chunks = [task.skills[:9], task.skills[9:15], task.skills[15:]]
        task.skills = [skill for i in case.order for skill in chunks[i]]
        from src.skills import SkillExecutor

        task.executor = SkillExecutor(task.skills, task.sta_controllers, task.dof)
        for skill in task.skills:
            if isinstance(skill, GraspSkill):
                skill.wait_time *= case.grasp
            else:
                skill.duration *= case.speed
        d.qpos[task.ctx.joint_qpos_indices] = task.initial_q
        mujoco.mj_forward(m, d)
        from types import SimpleNamespace

        self.arms = {
            "L": SimpleNamespace(
                executor=task.executor,
                ctx=task.ctx,
                robot=task.robot,
                sta_controllers=task.sta_controllers,
            )
        }
        self.state = "PHASE_EXEC"
        self.control_references = {}
        self.execution_report = {}
        self._events = []
        self._last_index = -1
        self._wrist_start = None
        self._wrist_rotation = 0.0
        self._pin_yaw = 0.0
        self._previous_pin_yaw = None
        self.case = case

    def step_once(self):
        if self.state == "DONE":
            return False
        t, m, d = self.task, self.model, self.data
        complete = t.update(d.time)
        if t.executor.current_index != self._last_index:
            self._events.append(
                {
                    "index": t.executor.current_index,
                    "time": float(d.time),
                    "objects": {
                        n: d.body(n).xpos.tolist() for n in ("pin", "pin_b", "block")
                    },
                }
            )
            self._last_index = t.executor.current_index
        if complete and d.time > t.executor.completion_time + 1.0:
            self.state = "DONE"
            self.execution_report = self.effects()
            return False
        if d.time > 100.0:
            self.state = "TIMEOUT"
            return False
        ref = t.get_desired_joint_position(d.time)
        vel = t.executor.get_desired_velocity(t.ctx)
        d.ctrl[:6] = compute_arm_torque(m, d, t.sta_controllers, range(6), ref, vel)
        d.ctrl[6] = t.ctx.gripper_target
        self.control_references["L"] = {"position": ref, "velocity": vel}
        mujoco.mj_step(m, d)
        if isinstance(current_skill(self, "L"), RotateSkill):
            if self._wrist_start is None:
                self._wrist_start = float(d.qpos[t.ctx.joint_qpos_indices[-1]])
            self._wrist_rotation = float(
                d.qpos[t.ctx.joint_qpos_indices[-1]] - self._wrist_start
            )
            rot = d.body("pin").xmat.reshape(3, 3)
            angle = float(np.arctan2(rot[1, 0], rot[0, 0]))
            if self._previous_pin_yaw is not None:
                self._pin_yaw += float(
                    np.arctan2(
                        np.sin(angle - self._previous_pin_yaw),
                        np.cos(angle - self._previous_pin_yaw),
                    )
                )
            self._previous_pin_yaw = angle
        return True

    def effects(self):
        goals = []
        positions = {
            name: self.data.body(name).xpos.copy() for name in ("pin", "pin_b", "block")
        }
        targets = {
            "pin": np.array([1.45, 0.8, 0.824]),
            "pin_b": np.array([1.45, 0.4, 0.780]),
            "block": np.array([1.30, 0.8, 0.780]),
        }
        for name, position in positions.items():
            delta = position - targets[name]
            valid = bool(
                np.linalg.norm(delta[:2]) <= 0.02
                and abs(delta[2]) <= (0.014 if name == "pin" else 0.015)
            )
            goals.append(
                {
                    "object": name,
                    "position": position.tolist(),
                    "target": targets[name].tolist(),
                    "position_error_m": float(np.linalg.norm(delta)),
                    "verified": valid,
                }
            )
        wrist_error = abs(self._wrist_rotation - 2 * pi)
        return {
            "verified": all(g["verified"] for g in goals) and wrist_error < 0.05,
            "goals": goals,
            "wrist_rotation_error_rad": wrist_error,
            "measured_pin_yaw_delta_rad": self._pin_yaw,
            "rotation_contract": "wrist trajectory only; this free cylindrical pin/socket has no threaded mechanism",
        }


def prepare(case, bank=None):
    if case.family == "assembly":
        return AssemblyRunner(case, bank)
    spec = mujoco.MjSpec.from_file(str(scene_path(case.scene)))
    board_spec = spec.body("shade_board")
    if board_spec is not None:
        if board_spec.mass > 0:
            board_spec.mass *= case.payload
            board_spec.inertia *= case.payload
        else:
            for geom in board_spec.geoms:
                if geom.mass > 0:
                    geom.mass *= case.payload
    runner = TaskRunner(
        str(scene_path(case.scene)),
        render_perception=False,
        sta_parameters=bank,
        model=spec.compile(),
    )
    for spec in runner.perception.registry.objects.values():
        if spec.joint:
            joint = mujoco.mj_name2id(
                runner.model, mujoco.mjtObj.mjOBJ_JOINT, spec.joint
            )
            if (
                joint >= 0
                and runner.model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_HINGE
            ):
                dof = int(runner.model.jnt_dofadr[joint])
                runner.model.dof_frictionloss[dof] *= case.resistance
                runner.model.dof_damping[dof] *= case.resistance
    runner.start(total_time=160, show_trajectory=False, plan_dict=plan_for(case))
    runner._do_planning()
    for phase in runner._phases:
        for arm in ("L", "R", "R_cleanup"):
            for skill in phase.get(arm, []):
                if isinstance(skill, GraspSkill):
                    skill.wait_time *= case.grasp
                else:
                    skill.duration *= case.speed
    runner._start_phase(0)
    return runner


def current_skill(runner, label):
    arm = runner.arms.get(label)
    executor = arm.executor if arm else None
    return executor.current_skill if executor and not executor.is_all_complete else None


def context_for(runner, label, skill):
    ctx = runner.arms[label].ctx
    return {
        "scene": ctx.scene_name,
        "object": getattr(skill, "target_object", None),
        "action": getattr(skill, "action", None),
        "angle": getattr(skill, "angle", None),
        "workflow": getattr(ctx, "workflow_tag", None),
        "stage": ctx.execution_stage,
        "goal_count": getattr(ctx, "workflow_goal_count", None),
        "goal_objects": getattr(ctx, "workflow_goal_objects", None),
        "goal_angles": getattr(ctx, "workflow_goal_angles", None),
        "duration": getattr(skill, "duration", getattr(skill, "wait_time", None)),
        "cart_pose": ctx.data.qpos[ctx.cart_qpos_indices].tolist()
        if ctx.cart_qpos_indices is not None
        else None,
    }


class PrefixState:
    def __init__(self, case, profile, occurrence=0):
        self.case, self.profile = case, profile
        self.runner = prepare(case)
        seen, previous = 0, None
        while self.runner.step_once():
            skill = current_skill(self.runner, profile.arm)
            arm = self.runner.arms.get(profile.arm)
            index = (
                (id(arm.executor), arm.executor.current_index)
                if arm and arm.executor
                else None
            )
            if profile.matches(profile.arm, skill, case.scene) and index != previous:
                if seen == occurrence:
                    break
                seen += 1
            previous = index
        else:
            raise ValueError(
                f"No actual prefix for {profile.id} / {case.name} occurrence {occurrence}"
            )
        self.specification = mujoco.mjtState.mjSTATE_INTEGRATION
        self.physical = np.empty(
            mujoco.mj_stateSize(self.runner.model, self.specification)
        )
        mujoco.mj_getState(
            self.runner.model, self.runner.data, self.physical, self.specification
        )
        self.arms = {}
        for label, arm in self.runner.arms.items():
            self.arms[label] = {
                "executor": deepcopy(arm.executor),
                "controllers": [c.get_state() for c in arm.sta_controllers],
                "robot_q": np.asarray(arm.robot.get_joint()).copy(),
                "config": deepcopy(arm.robot.robot_config.__dict__),
                "sensor_offset": arm.ctx.sensor_offset
                if arm.ctx
                else (0 if label == "L" else 6),
                "ctx": {
                    name: deepcopy(getattr(arm.ctx, name))
                    for name in (
                        "gripper_target",
                        "current_time",
                        "skill_start_time",
                        "last_reference",
                        "execution_stage",
                    )
                }
                if arm.ctx
                else None,
            }
        if isinstance(self.runner, AssemblyRunner):
            self.runtime = {
                name: deepcopy(getattr(self.runner, name))
                for name in (
                    "_events",
                    "_last_index",
                    "_wrist_start",
                    "_wrist_rotation",
                    "_pin_yaw",
                    "_previous_pin_yaw",
                )
            }
        else:
            self.runtime = {
                name: deepcopy(getattr(self.runner, name))
                for name in (
                    "_phase_index",
                    "_R_cleanup_pending",
                    "_L_waiting_for_R",
                    "_R_cleanup_skills",
                    "_pending_L_skills",
                    "_R_hold_position",
                )
            }
            self.effects = deepcopy(self.runner._effect_monitor.records)
            self.effect_tick = self.runner._effect_monitor._tick

    def restore(self, parameters):
        runner, profile = self.runner, self.profile
        mujoco.mj_setState(runner.model, runner.data, self.physical, self.specification)
        mujoco.mj_forward(runner.model, runner.data)
        runner.state, runner.execution_report, runner.control_references = (
            "PHASE_EXEC",
            {},
            {},
        )
        bank = make_bank(parameters)
        for name, value in self.runtime.items():
            setattr(runner, name, deepcopy(value))
        if not isinstance(runner, AssemblyRunner):
            runner._is_started = True
            runner.sta_parameters = bank
            runner._effect_monitor.records = deepcopy(self.effects)
            runner._effect_monitor._tick = self.effect_tick
        for label, saved in self.arms.items():
            arm = runner.arms[label]
            arm.executor = deepcopy(saved["executor"])
            arm.robot.set_joint(saved["robot_q"])
            arm.robot.robot_config.__dict__.update(deepcopy(saved["config"]))
            for ctrl, state in zip(arm.sta_controllers, saved["controllers"]):
                ctrl.set_state(state)
            if arm.executor:
                arm.executor.sta_controllers = arm.sta_controllers
            if saved["ctx"]:
                from src.skills import SkillContext

                arm.ctx = SkillContext(
                    runner.model, runner.data, arm.robot, 6, saved["sensor_offset"]
                )
                arm.ctx.scene_name = self.case.scene
                arm.ctx.workflow_tag = (
                    "assembly"
                    if isinstance(runner, AssemblyRunner)
                    else runner.workflow_tag
                )
                if not isinstance(runner, AssemblyRunner):
                    goals = runner.prepared_plan.get("goals", [])
                    arm.ctx.workflow_goal_count = len(goals)
                    arm.ctx.workflow_goal_objects = [g["object"] for g in goals]
                    arm.ctx.workflow_goal_angles = [
                        g["angle"] for g in goals if g.get("operation") == "rotate"
                    ]
                for name, value in saved["ctx"].items():
                    setattr(arm.ctx, name, deepcopy(value))
                arm.ctx.parameter_bank = bank
            else:
                arm.ctx = None
        if isinstance(runner, AssemblyRunner):
            runner.task.executor = runner.arms["L"].executor
            runner.task.ctx = runner.arms["L"].ctx
        arm = runner.arms[profile.arm]
        skill = current_skill(runner, profile.arm)
        params = bank.resolve(
            profile.arm,
            skill.skill_type,
            skill.sta_params,
            context_for(runner, profile.arm, skill),
        )
        for i, ctrl in enumerate(arm.sta_controllers):
            ctrl.set_parameter(
                params["alpha"][i], params["beta"][i], params["lambda_s"][i]
            )
        return runner

    def evaluate(self, ratios):
        return evaluate(
            self.restore({self.profile.id: list(ratios)}), self.case, self.profile
        )


def evaluate(runner, case, profile=None, record=False):
    whole, target, suffix = (
        {},
        ControlMetrics(force_limit=120.0, error_limit=0.15),
        ControlMetrics(force_limit=120.0, error_limit=0.15),
    )
    by_skill, traces, violations = {}, [], []
    previous, transition_jump, target_finished = {}, 0.0, None
    start = float(runner.data.time)
    initial_skill = current_skill(runner, profile.arm) if profile else None
    mappings = {}
    for label in runner.arms:
        ids = np.array(
            [
                mujoco.mj_name2id(
                    runner.model,
                    mujoco.mjtObj.mjOBJ_ACTUATOR,
                    name + ("_R" if label == "R" else ""),
                )
                for name in (
                    "shoulder_pan",
                    "shoulder_lift",
                    "elbow",
                    "wrist_1",
                    "wrist_2",
                    "wrist_3",
                )
            ]
        )
        mappings[label] = (
            ids,
            np.max(np.abs(runner.model.actuator_forcerange[ids]), axis=1),
        )
    while runner.step_once():
        for label, arm in runner.arms.items():
            if not arm.ctx or label not in runner.control_references:
                continue
            skill = current_skill(runner, label)
            ref = runner.control_references[label]["position"]
            ids, limits = mappings[label]
            torque = runner.data.actuator_force[ids].copy()
            actual = runner.data.qpos[arm.ctx.joint_qpos_indices].copy()
            wrench = arm.ctx.get_external_force()
            error = np.asarray(ref) - actual
            normalized = torque / limits
            jitter = (
                ((torque - previous[label][1]) / limits)
                if label in previous
                else np.zeros(6)
            )
            error_mse, effort_mse, jitter_mse = (
                float(error @ error / 6),
                float(normalized @ normalized / 6),
                float(jitter @ jitter / 6),
            )
            force = float(np.linalg.norm(wrench[:3]))
            observation = (
                error_mse,
                float(np.max(np.abs(error))),
                effort_mse,
                jitter_mse,
                force,
                float(np.max(np.abs(torque))),
                isfinite(error_mse + effort_mse + jitter_mse + force)
                and bool(np.isfinite(wrench).all()),
            )
            if label not in whole:
                whole[label] = ControlMetrics(force_limit=120.0, error_limit=0.15)
            metric = whole[label]
            metric.force_limit = (
                60.0
                if case.family != "assembly" and isinstance(skill, RotateSkill)
                else 120.0
            )
            metric.sample_observation(*observation)
            name = (
                f"{label}:{skill.skill_type.name}:{getattr(skill, 'action', '')}"
                if skill
                else f"{label}:HOLD"
            )
            if name not in by_skill:
                by_skill[name] = ControlMetrics(force_limit=120.0, error_limit=0.15)
            by_skill[name].sample_observation(*observation)
            if profile and label == profile.arm:
                if skill is initial_skill:
                    target.sample_observation(*observation)
                elif target.count and target_finished is None:
                    target_finished = float(runner.data.time)
                if target.count:
                    suffix.sample_observation(*observation)
            if label in previous and previous[label][0] != id(skill):
                transition_jump = max(
                    transition_jump, float(np.max(np.abs(torque - previous[label][1])))
                )
            previous[label] = (id(skill), torque)
            if metric.violation:
                violations.append(f"{label}:{metric.violation}")
            if record and round(runner.data.time / runner.model.opt.timestep) % 5 == 0:
                traces.append(
                    [
                        float(runner.data.time),
                        label,
                        name,
                        ref.tolist(),
                        actual.tolist(),
                        torque.tolist(),
                        wrench.tolist(),
                    ]
                )
        if violations:
            runner.state = "SAFETY_STOP"
            break
        if (
            profile
            and target_finished is not None
            and runner.data.time >= target_finished + 2.0
        ):
            break
    reports = {name: metric.report() for name, metric in by_skill.items()}
    if profile:
        local = target.report()
        valid = bool(
            target.count > 100 and not violations and target_finished is not None
        )
        cost = (
            local["cost"]
            + 0.25 * suffix.report()["cost"]
            + 0.01 * (transition_jump / 5.0) ** 2
        )
        effects = {
            "verified": None,
            "contract": "training window only; complete task required during selection and holdout",
        }
    else:
        effects = runner.execution_report
        valid = bool(
            runner.state == "DONE" and effects.get("verified") and not violations
        )
        if case.family == "assembly":
            penalty = (
                np.mean(
                    [
                        (g["position_error_m"] / 0.02) ** 2
                        for g in effects.get("goals", [])
                    ]
                )
                if effects
                else 1000.0
            )
        else:
            goals = effects.get("goals", [])
            errors = [
                g.get("angle_error_rad", pi)
                for g in goals
                if g.get("operation") == "rotate"
            ]
            valid &= bool(goals and (not errors or max(errors) <= np.deg2rad(5)))
            penalties = [(error / np.deg2rad(5)) ** 2 for error in errors]
            penalties.extend(
                (g.get("best_inspect_distance_m", 1.0) / 0.05) ** 2
                for g in goals
                if g.get("operation") == "inspect"
            )
            penalty = np.mean(penalties) if penalties else 1000.0
        costs = [m.report()["cost"] for m in whole.values()]
        cost = (
            float(
                np.mean(costs)
                + 0.2 * max(costs)
                + penalty
                + 0.01 * (transition_jump / 5.0) ** 2
            )
            if costs
            else 1e6
        )
        local = {"samples": sum(m.count for m in whole.values()), "cost": cost}
    result = {
        "case": asdict(case),
        "profile": profile.id if profile else None,
        "valid": valid,
        "state": runner.state,
        "cost": float(cost if valid else 1000.0 + cost),
        "target": local,
        "per_skill": reports,
        "transition_torque_jump_Nm": transition_jump,
        "violations": sorted(set(violations)),
        "execution_effects": effects,
        "start_time": start,
        "end_time": float(runner.data.time),
    }
    if record:
        result["traces"] = traces
    return result


def complete_evaluation(case, parameters=None, record=False):
    with redirect_stdout(io.StringIO()):
        runner = prepare(case, make_bank(parameters or {}))
        return evaluate(runner, case, record=record)
