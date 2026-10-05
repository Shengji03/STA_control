from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageFilter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "src" / "assets" / "universal_robots_ur5e" / "ur5e_mobile.xml"
BUILD_DIR = PROJECT_ROOT / "outputs" / "robot_only_render"

LEFT_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
RIGHT_JOINTS = [f"{name}_R" for name in LEFT_JOINTS]

POSES = {
    "display": {
        "L": [-0.55, -0.75, 1.55, -0.95, -1.45, 0.35],
        "R": [-0.55, -0.75, 1.55, -0.95, -1.45, -0.35],
    },
    "open": {
        "L": [0.85, -0.95, 1.75, -1.05, -1.35, 0.0],
        "R": [0.85, -0.95, 1.75, -1.05, -1.35, 0.0],
    },
    "home": {
        "L": [0.0, 0.0, np.pi / 2.0, 0.0, -np.pi / 2.0, 0.0],
        "R": [0.0, 0.0, np.pi / 2.0, 0.0, -np.pi / 2.0, 0.0],
    },
}

VIEWS = {
    "front": {"azimuth": 140.0, "elevation": -22.0, "distance": 2.55, "lookat": [0.0, 0.0, 0.78]},
    "left": {"azimuth": 118.0, "elevation": -20.0, "distance": 2.50, "lookat": [0.0, 0.0, 0.78]},
    "right": {"azimuth": 160.0, "elevation": -20.0, "distance": 2.50, "lookat": [0.0, 0.0, 0.78]},
    "high": {"azimuth": 140.0, "elevation": -30.0, "distance": 2.80, "lookat": [0.0, 0.0, 0.70]},
}


def write_robot_only_scene(width: int, height: int, include_floor: bool) -> Path:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    scene_path = BUILD_DIR / "robot_only_scene.xml"
    include_path = MODEL_PATH.as_posix()
    floor_xml = '    <geom name="floor" type="plane" size="3 3 0.05" material="plain_floor"/>' if include_floor else ""
    scene_path.write_text(
        f"""<mujoco model="mobile_dual_arm_robot_only">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 -9.82"/>
  <include file="{include_path}"/>
  <statistic center="0 0 0.7" extent="2.0" meansize="0.06"/>
  <visual>
    <headlight diffuse="0.45 0.45 0.45" ambient="0.26 0.26 0.26" specular="0.04 0.04 0.04"/>
    <global azimuth="145" elevation="-25" offwidth="{width}" offheight="{height}"/>
    <quality shadowsize="4096"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.82 0.87 0.92" rgb2="0.96 0.97 0.98" width="512" height="3072"/>
    <material name="plain_floor" rgba="0.78 0.78 0.76 1" reflectance="0.0"/>
  </asset>
  <worldbody>
    <light name="key_light" pos="-1.8 -2.2 4.0" dir="0.4 0.6 -1" directional="true" diffuse="0.70 0.70 0.70" specular="0.18 0.18 0.18"/>
    <light name="fill_light" pos="2.0 1.5 2.5" dir="-0.4 -0.3 -1" directional="true" diffuse="0.22 0.22 0.22"/>
{floor_xml}
  </worldbody>
</mujoco>
""",
        encoding="utf-8",
    )
    return scene_path


def set_joint(model: mujoco.MjModel, data: mujoco.MjData, name: str, value: float) -> None:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
        raise ValueError(f"Joint not found: {name}")
    data.qpos[model.jnt_qposadr[joint_id]] = value


def apply_pose(model: mujoco.MjModel, data: mujoco.MjData, pose_name: str) -> None:
    pose = POSES[pose_name]
    for name, value in zip(LEFT_JOINTS, pose["L"]):
        set_joint(model, data, name, float(value))
    for name, value in zip(RIGHT_JOINTS, pose["R"]):
        set_joint(model, data, name, float(value))
    set_joint(model, data, "right_driver_joint", 0.18)
    set_joint(model, data, "right_driver_joint_R", 0.18)
    mujoco.mj_forward(model, data)


def make_camera(view_name: str) -> mujoco.MjvCamera:
    view = VIEWS[view_name]
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth = view["azimuth"]
    camera.elevation = view["elevation"]
    camera.distance = view["distance"]
    camera.lookat[:] = np.array(view["lookat"], dtype=float)
    return camera


def render_robot(
    output: Path,
    pose_name: str,
    view_name: str,
    width: int,
    height: int,
    shadows: bool,
    transparent: bool,
) -> None:
    scene_path = write_robot_only_scene(width, height, include_floor=not transparent)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    apply_pose(model, data, pose_name)

    renderer = mujoco.Renderer(model, width=width, height=height)
    try:
        renderer.update_scene(data, camera=make_camera(view_name))
        if not shadows:
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        image = renderer.render()
        if transparent:
            renderer.update_scene(data, camera=make_camera(view_name))
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = 1
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = 1
            segment = renderer.render()
    finally:
        renderer.close()

    output.parent.mkdir(parents=True, exist_ok=True)
    if transparent:
        alpha = ((segment[:, :, 0] != 0) | (segment[:, :, 1] != 0) | (segment[:, :, 2] != 0)).astype(np.uint8) * 255
        alpha_image = Image.fromarray(alpha, mode="L").filter(ImageFilter.GaussianBlur(radius=0.6))
        rgba = Image.fromarray(image).convert("RGBA")
        rgba.putalpha(alpha_image)
        rgba.save(output)
    else:
        Image.fromarray(image).save(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the mobile dual-arm MuJoCo robot without task-scene objects.")
    parser.add_argument("--output", type=Path, required=True, help="Output PNG path.")
    parser.add_argument("--pose", choices=sorted(POSES), default="display", help="Robot joint pose.")
    parser.add_argument("--view", choices=sorted(VIEWS), default="front", help="Camera view.")
    parser.add_argument("--width", type=int, default=1600, help="Output image width.")
    parser.add_argument("--height", type=int, default=900, help="Output image height.")
    parser.add_argument("--shadows", action="store_true", help="Keep MuJoCo shadows in the output.")
    parser.add_argument("--transparent", action="store_true", help="Export robot body with a transparent background.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render_robot(args.output, args.pose, args.view, args.width, args.height, args.shadows, args.transparent)
    print(args.output)


if __name__ == "__main__":
    main()
