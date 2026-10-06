"""Physical conventions, finite-difference kinematics, and reference tracking."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from src.config.paths import scene_path
from src.config.robot import INITIAL_JOINTS
from src.controller.admittance_controller import AdmittanceController, AdmittanceMode
from src.motion_planning import JointTrajectory
from src.pipeline.arm_state import ArmState
from src.pipeline.task_runner import TaskRunner
from src.skills.skill_executor import SkillContext, SkillExecutor
from src.skills.skills import GraspSkill, MoveSkill, RotateSkill


def environment(scene='scene5_glare', arm='L'):
    model = mujoco.MjModel.from_xml_path(str(scene_path(scene)))
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    data = mujoco.MjData(model)
    for suffix in ('', '_R'):
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'shoulder_pan_joint' + suffix)
        if joint >= 0:
            address = int(model.jnt_qposadr[joint])
            data.qpos[address:address + 6] = INITIAL_JOINTS
    for name, value in [('cart_x', .3), ('cart_y', -.1), ('cart_yaw', .65)]:
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint >= 0:
            data.qpos[model.jnt_qposadr[joint]] = value
    mujoco.mj_forward(model, data)
    ctx = SkillContext(model, data, None, 6, 0 if arm == 'L' else 6)
    return model, data, ctx


@pytest.mark.parametrize('scene,arm', [('scene3', 'L'), ('scene4_pipeline', 'L'),
                                       ('scene4_pipeline', 'R'), ('scene5_glare', 'L'), ('scene5_glare', 'R')])
def test_gripper_jacobian_matches_cartesian_finite_differences(scene, arm):
    model, data, ctx = environment(scene, arm)
    data.qpos[ctx.joint_qpos_indices] += [.15, -.05, .04, .1, .2, -.2]
    mujoco.mj_forward(model, data)
    actual = ctx.get_jacobian()
    original = data.qpos.copy()
    position = ctx.control_point
    rotation = data.site_xmat[ctx.control_site_id].reshape(3, 3).copy()
    numerical = np.zeros((6, 6))
    epsilon = 1e-6
    for col, address in enumerate(ctx.joint_qpos_indices):
        data.qpos[:] = original
        data.qpos[address] += epsilon
        mujoco.mj_forward(model, data)
        numerical[:3, col] = (ctx.control_point - position) / epsilon
        delta = data.site_xmat[ctx.control_site_id].reshape(3, 3) @ rotation.T
        numerical[3:, col] = np.array([delta[2, 1] - delta[1, 2],
                                       delta[0, 2] - delta[2, 0],
                                       delta[1, 0] - delta[0, 1]]) / (2 * epsilon)
    np.testing.assert_allclose(actual, numerical, atol=2e-6)
    assert np.linalg.norm(actual) > 1
    if scene != 'scene3':
        np.testing.assert_array_equal(ctx.joint_dof_indices, range(3, 9) if arm == 'L' else range(17, 23))


@pytest.mark.parametrize('arm', ['L', 'R'])
def test_wrench_includes_finger_forces_with_world_axes_and_moment_at_pinch(arm):
    model, data, ctx = environment(arm=arm)
    suffix = '' if arm == 'L' else '_R'
    force = np.array([1.5, -2., 3.])
    torque = np.array([.7, -.4, .2])
    finger = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'right_pad' + suffix)
    data.xfrc_applied[finger] = np.r_[force, torque]
    # A force on the opposite arm must not leak into this tool's wrench.
    opposite = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'flange_R' if arm == 'L' else 'flange')
    data.xfrc_applied[opposite] = [30, 40, 50, 1, 2, 3]
    expected = np.r_[force, torque + np.cross(data.xipos[finger] - ctx.control_point, force)]
    np.testing.assert_allclose(ctx.get_external_force(), expected, atol=1e-10)
    data.xfrc_applied[finger] *= 2
    # No time change or pre-existing force sensor: the getter still refreshes RNE.
    np.testing.assert_allclose(ctx.get_external_force(), 2 * expected, atol=1e-10)


def test_wrench_sums_multiple_gripper_bodies_and_cancels_internal_force_pairs():
    model, data, ctx = environment()
    force = np.array([3., 1., -2.])
    first = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'right_pad')
    second = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'left_pad')
    point = (data.xipos[first] + data.xipos[second]) / 2
    data.xfrc_applied[first] = np.r_[force, np.cross(point - data.xipos[first], force)]
    data.xfrc_applied[second] = np.r_[-force, np.cross(point - data.xipos[second], -force)]
    np.testing.assert_allclose(ctx.get_external_force(), np.zeros(6), atol=1e-10)


def test_actual_contact_on_a_finger_is_reported_without_force_sensors():
    xml = '<mujoco><option gravity="0 0 0"/><worldbody><geom type="plane" size="1 1 .1"/>'
    for i in range(6):
        xml += f'<body pos="0 0 {0.04 if i == 0 else 0}"><inertial pos="0 0 0" mass=".1" diaginertia=".01 .01 .01"/><joint name="j{i}" type="{ "slide" if i == 0 else "hinge"}" axis="0 0 1"/>'
    xml += '<body name="flange"><site name="pinch" pos=".02 0 0"/><body name="finger"><geom type="sphere" size=".05" mass="1"/></body></body>'
    xml += '</body>' * 6 + '</worldbody><sensor>'
    xml += ''.join(f'<jointpos joint="j{i}"/>' for i in range(6)) + '</sensor></mujoco>'
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ctx = SkillContext(model, data, None, 6)
    assert data.ncon > 0
    np.testing.assert_array_equal(data.cfrc_ext, 0)
    wrench = ctx.get_external_force()
    assert wrench[2] > 1
    np.testing.assert_allclose(wrench[[0, 1, 3, 5]], 0, atol=1e-10)
    assert wrench[4] == pytest.approx(.02 * wrench[2])


def test_cubic_reference_velocity_is_the_derivative_of_position():
    trajectory = JointTrajectory([.1, -.2], [.8, .6], 3)
    for t in (.25, 1., 2.75):
        numerical = (trajectory.interpolate(t + 1e-6) - trajectory.interpolate(t - 1e-6)) / 2e-6
        np.testing.assert_allclose(trajectory.velocity(t), numerical, atol=1e-9)
    for t in (-1, 0, 3, 4):
        np.testing.assert_array_equal(trajectory.velocity(t), [0, 0])


def test_stationary_grasp_reference_has_zero_velocity():
    skill = GraspSkill()
    np.testing.assert_array_equal(skill.get_desired_velocity(SimpleNamespace(dof=6)), np.zeros(6))


def test_task_runner_uses_desired_velocity_and_holding_does_not_reuse_motion_velocity():
    seen = []
    class Controller:
        def control(self, position_error, velocity_error):
            seen.append((position_error, velocity_error))
            return position_error + velocity_error
    arm = ArmState('L', np.zeros(3), 0, 0)
    arm.sta_controllers = [Controller() for _ in range(6)]
    arm.prev_sensor = np.zeros(6)
    runner = object.__new__(TaskRunner)
    runner.model = SimpleNamespace(opt=SimpleNamespace(timestep=.01))
    runner._get_arm_sensor = lambda _: np.ones(6) * .02
    # Exact tracking of a 2 rad/s moving reference produces zero sliding error.
    np.testing.assert_allclose(runner._compute_arm_control(arm, np.ones(6) * .02, np.ones(6) * 2), 0)
    assert all(p == 0 and v == 0 for p, v in seen)
    seen.clear()
    np.testing.assert_allclose(runner._compute_arm_control(arm, np.ones(6) * .02), 0)
    assert all(v == 0 for _, v in seen)


def test_compensated_velocity_uses_command_derivative_and_same_time_queries_do_not_integrate_twice():
    class Robot:
        def move_joint(self, q):
            self.q = np.array(q)
        def get_joint(self):
            return self.q.copy()
    ctx = SimpleNamespace(dof=6, model=SimpleNamespace(opt=SimpleNamespace(timestep=.002)),
                          current_time=0., elapsed=0., robot=Robot(),
                          get_sensor_data=lambda: np.zeros(6),
                          get_external_force=lambda: np.array([0, 0, 1, 0, 0, 0]),
                          get_jacobian=lambda: np.eye(6))
    skill = RotateSkill(angle=.1, duration=2)
    skill.setup(ctx)
    first = skill.get_desired_position(ctx)
    ctx.current_time = ctx.elapsed = .002
    second = skill.get_desired_position(ctx)
    velocity = skill.get_desired_velocity(ctx)
    compensation_derivative = ((second - skill._planner.interpolate(.002))
                               - (first - skill._planner.interpolate(0))) / .002
    np.testing.assert_allclose(velocity - skill._planner.velocity(.002), compensation_derivative)
    np.testing.assert_array_equal(skill.get_desired_position(ctx), second)
    np.testing.assert_array_equal(skill.get_desired_velocity(ctx), velocity)
    # A skill restart resets both the physical admittance and reference derivative.
    ctx.elapsed = 0.
    skill.setup(ctx)
    np.testing.assert_array_equal(skill.get_desired_velocity(ctx), np.zeros(6))
    np.testing.assert_array_equal(skill._admittance.displacement, np.zeros(6))


def test_completed_executor_returns_zero_velocity_instead_of_last_trajectory_velocity():
    ctx = SimpleNamespace(dof=6)
    executor = SkillExecutor([MoveSkill(np.ones(6))], [], 6)
    executor.current_index = 0
    executor._complete = True
    np.testing.assert_array_equal(executor.get_desired_velocity(ctx), np.zeros(6))


@pytest.mark.parametrize('mode,reference', [(AdmittanceMode.INSERT, 5), (AdmittanceMode.EXTRACT, -3)])
def test_contact_reference_is_environment_on_tool_and_drives_correct_approach_direction(mode, reference):
    controller = AdmittanceController(mode)
    expected = np.zeros(6); expected[2] = reference
    np.testing.assert_array_equal(controller.compute(expected, np.eye(6)), np.zeros(6))
    dq = controller.compute(np.zeros(6), np.eye(6))
    assert dq[2] * reference < 0  # Insert moves down, extract moves up.
    np.testing.assert_array_equal(dq[[0, 1, 3, 4, 5]], np.zeros(5))


def test_admittance_force_and_torque_channels_are_separate_and_resettable():
    force = AdmittanceController(AdmittanceMode.ROTATE)
    torque = AdmittanceController(AdmittanceMode.ROTATE)
    dq_force = force.compute([0, 0, 1, 0, 0, 0], np.eye(6))
    dq_torque = torque.compute([0, 0, 0, 0, 0, 1], np.eye(6))
    assert dq_force[2] > 0 and dq_force[5] == 0
    assert dq_torque[5] > 0 and dq_torque[2] == 0
    force.reset()
    np.testing.assert_array_equal(force.compute(np.zeros(6), np.eye(6)), 0)


def test_singular_jacobian_and_large_contact_force_keep_compensation_bounded():
    controller = AdmittanceController(AdmittanceMode.ROTATE)
    singular = np.eye(6); singular[5, 5] = 0
    previous = np.zeros(6)
    for _ in range(200):
        dq = controller.compute([0, 0, 1e5, 0, 0, 1e5], singular)
        assert np.all(np.isfinite(dq))
        assert np.max(np.abs(dq)) <= .1
        assert np.max(np.abs(dq - previous)) <= .5 * .002 + 1e-12
        previous = dq
    assert abs(controller.displacement[2]) <= .003
    assert abs(controller.displacement[5]) <= .02
    # Saturated displacement stores no outward velocity that keeps driving it.
    before = controller.displacement[2]
    controller.compute(np.zeros(6), singular)
    assert controller.displacement[2] < before


@pytest.mark.parametrize('force,jac', [(np.zeros(3), np.eye(6)), (np.ones(6) * np.nan, np.eye(6)),
                                      (np.zeros(6), np.zeros((6, 8)))])
def test_bad_control_measurements_fail_explicitly(force, jac):
    with pytest.raises(ValueError):
        AdmittanceController(AdmittanceMode.ROTATE).compute(force, jac)
