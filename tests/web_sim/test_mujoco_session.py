from pathlib import Path

from src.web_sim.mujoco_session import MujocoStreamSession, SimulationConfig


SAMPLE_NAV_PLAN = {
    "plan": [
        {
            "step": 1,
            "arm": "L",
            "skill": "NavSkill",
            "params": {"target": [-2.8, -0.5]},
            "description": "小车沿原项目 NavSkill 移动",
        }
    ]
}


def test_mujoco_stream_session_renders_jpeg_from_project_scene():
    scene = Path("src/assets/scenes/scene5_glare.xml").resolve()
    config = SimulationConfig(
        scene_path=scene,
        width=160,
        height=90,
        fps=5,
        jpeg_quality=70,
    )
    session = MujocoStreamSession(config)

    try:
        frame = session.step_and_render()
        status = session.get_status()
    finally:
        session.close()

    assert frame[:2] == b"\xff\xd8"
    assert status["state"] == "running"
    assert status["sim_time"] == 0
    assert status["resolution"] == [160, 90]
    assert "cam_global" in status["available_cameras"]


def test_idle_stream_renders_without_advancing_uncontrolled_physics():
    scene = Path("src/assets/scenes/scene5_glare.xml").resolve()
    config = SimulationConfig(scene_path=scene, width=80, height=60, fps=5)
    session = MujocoStreamSession(config)

    try:
        session.step_and_render()
        status = session.get_status()
    finally:
        session.close()

    assert status["sim_time"] == 0


def test_mujoco_stream_session_accepts_camera_commands():
    scene = Path("src/assets/scenes/scene5_glare.xml").resolve()
    config = SimulationConfig(scene_path=scene, width=80, height=60, fps=5)
    session = MujocoStreamSession(config)

    try:
        session.apply_camera_command({"action": "set_fixed", "camera": "cam_global"})
        status = session.get_status()
    finally:
        session.close()

    assert status["camera"]["mode"] == "fixed"
    assert status["camera"]["fixed_camera"] == "cam_global"


def test_mujoco_stream_session_dispatches_pipeline_task():
    scene = Path("src/assets/scenes/scene5_glare.xml").resolve()
    config = SimulationConfig(scene_path=scene, width=80, height=60, fps=5)
    session = MujocoStreamSession(config)

    try:
        session.dispatch_pipeline_task(
            task_id="LAB-TEST",
            instruction="执行原项目 NavSkill 计划",
            total_time=10.0,
            plan_dict=SAMPLE_NAV_PLAN,
        )
        before = session.get_status()["active_task"]
        session.step_and_render()
        after = session.get_status()["active_task"]
    finally:
        session.close()

    assert before["id"] == "LAB-TEST"
    assert after["id"] == "LAB-TEST"
    assert after["runner_state"] in {"PHASE_NAV", "PHASE_EXEC", "DONE"}
    assert after["progress"] > before["progress"]


def test_runner_run_preserves_preloaded_plan_without_llm():
    scene = Path("src/assets/scenes/scene5_glare.xml").resolve()
    session = MujocoStreamSession(SimulationConfig(scene_path=scene, width=80, height=60))
    plan = {"plan": [{
        "arm": "L", "skill": "NavSkill", "params": {"target": [-2.99, -0.5]},
    }]}
    try:
        session.runner.plan_dict = plan
        session.runner.run(total_time=3.0, use_viewer=False, show_trajectory=False)
        assert session.runner.state == "DONE"
        assert session.runner.plan_dict is plan
        assert session.runner.progress() == 1.0
    finally:
        session.close()
