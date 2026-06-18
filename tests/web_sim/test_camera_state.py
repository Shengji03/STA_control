from src.web_sim.camera import CameraCommand, CameraState


def test_camera_state_exposes_default_free_camera_payload():
    state = CameraState()

    payload = state.to_payload()

    assert payload["mode"] == "free"
    assert payload["azimuth"] == 150.0
    assert payload["elevation"] == -25.0
    assert payload["distance"] == 4.0
    assert payload["lookat"] == [0.0, 0.3, 0.6]


def test_camera_state_applies_orbit_pan_and_zoom_commands():
    state = CameraState()

    state.apply(CameraCommand(action="orbit", dx=15, dy=-10))
    state.apply(CameraCommand(action="pan", dx=0.2, dy=-0.1, dz=0.3))
    state.apply(CameraCommand(action="zoom", amount=-1.5))

    assert state.azimuth == 165.0
    assert state.elevation == -35.0
    assert state.lookat == [0.2, 0.19999999999999998, 0.8999999999999999]
    assert state.distance == 2.5


def test_camera_state_clamps_distance_and_elevation():
    state = CameraState()

    state.apply(CameraCommand(action="orbit", dx=0, dy=200))
    state.apply(CameraCommand(action="zoom", amount=-100))

    assert state.elevation == 89.0
    assert state.distance == 0.2


def test_camera_state_switches_to_named_fixed_camera():
    state = CameraState()

    state.apply(CameraCommand(action="set_fixed", camera="cam_global"))

    assert state.mode == "fixed"
    assert state.fixed_camera == "cam_global"
    assert state.to_payload()["fixed_camera"] == "cam_global"
