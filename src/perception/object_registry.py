from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping, Optional


@dataclass(frozen=True)
class RefSpec:
    ref_type: str
    name: str


@dataclass(frozen=True)
class ObjectSpec:
    name: str
    kind: str
    body: Optional[str] = None
    site: Optional[str] = None
    joint: Optional[str] = None
    light: Optional[str] = None
    camera: Optional[str] = None
    keypoints: Mapping[str, RefSpec] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)


class SceneObjectRegistry:
    def __init__(
        self,
        scene_name: str,
        objects: Mapping[str, ObjectSpec],
        default_cameras: tuple[str, ...] = (),
    ) -> None:
        self.scene_name = scene_name
        self.objects = dict(objects)
        self.default_cameras = tuple(default_cameras)

    @classmethod
    def from_scene_path(cls, scene_path: Optional[str]) -> "SceneObjectRegistry":
        if not scene_path:
            return cls.empty()
        scene_name = Path(scene_path).stem
        return cls.for_scene(scene_name)

    @classmethod
    def for_scene(cls, scene_name: str) -> "SceneObjectRegistry":
        registry = _SCENE_REGISTRIES.get(scene_name)
        if registry is None:
            return cls(scene_name, {})
        return registry

    @classmethod
    def empty(cls) -> "SceneObjectRegistry":
        return cls("unknown", {})


_SCENE_REGISTRIES: Dict[str, SceneObjectRegistry] = {
    "scene3": SceneObjectRegistry(
        "scene3",
        {
            "ur5e_base": ObjectSpec("ur5e_base", kind="robot_base", body="ur5e_base"),
            "pin_a": ObjectSpec(
                "pin_a",
                kind="pin",
                body="pin",
                keypoints={"top": RefSpec("site", "pin_top")},
            ),
            "socket_a": ObjectSpec(
                "socket_a",
                kind="socket",
                body="socket",
                keypoints={
                    "entry": RefSpec("site", "socket_entry"),
                    "center": RefSpec("site", "socket_center"),
                },
            ),
            "pin_b": ObjectSpec(
                "pin_b",
                kind="pin",
                body="pin_b",
                keypoints={"top": RefSpec("site", "pin_b_top")},
            ),
            "socket_b": ObjectSpec(
                "socket_b",
                kind="socket",
                body="socket_b",
                keypoints={
                    "entry": RefSpec("site", "socket_b_entry"),
                    "center": RefSpec("site", "socket_b_center"),
                },
            ),
            "block": ObjectSpec(
                "block",
                kind="block",
                body="block",
                keypoints={"top": RefSpec("site", "block_top")},
            ),
            "zone_block_target": ObjectSpec(
                "zone_block_target", kind="target_zone", body="zone_block_target"
            ),
            "zone_place_b": ObjectSpec("zone_place_b", kind="target_zone", body="zone_place_b"),
            "tcp_L": ObjectSpec("tcp_L", kind="tcp", site="tcp"),
        },
        default_cameras=("cam_overview", "cam_side", "cam_pickup", "cam_drop"),
    ),
    "scene4_pipeline": SceneObjectRegistry(
        "scene4_pipeline",
        {
            "mobile_cart": ObjectSpec("mobile_cart", kind="mobile_base", body="mobile_cart"),
            "valve_1": ObjectSpec(
                "valve_1",
                kind="valve",
                body="valve_body_1",
                joint="valve_joint",
                keypoints={"handle_center": RefSpec("site", "handwheel_center")},
            ),
            "valve_2": ObjectSpec(
                "valve_2",
                kind="valve",
                body="valve_body_2",
                joint="valve_joint_2",
                keypoints={"handle_center": RefSpec("site", "handwheel_center_2")},
            ),
            "cart_nav_target_1": ObjectSpec(
                "cart_nav_target_1", kind="nav_target", site="cart_nav_target_1"
            ),
            "cart_nav_target_2": ObjectSpec(
                "cart_nav_target_2", kind="nav_target", site="cart_nav_target_2"
            ),
            "tcp_L": ObjectSpec("tcp_L", kind="tcp", site="tcp"),
            "tcp_R": ObjectSpec("tcp_R", kind="tcp", site="tcp_R"),
        },
        default_cameras=("cam_overview", "cam_side", "cam_valve", "cam_bird"),
    ),
    "scene5_glare": SceneObjectRegistry(
        "scene5_glare",
        {
            "mobile_cart": ObjectSpec("mobile_cart", kind="mobile_base", body="mobile_cart"),
            "valve_1": ObjectSpec(
                "valve_1",
                kind="valve",
                body="valve_body_1",
                joint="valve_joint",
                keypoints={"handle_center": RefSpec("site", "handwheel_center")},
                metadata={"label": "1号阀门", "location_x": 0.45},
            ),
            "valve_2": ObjectSpec(
                "valve_2",
                kind="valve",
                body="valve_body_2",
                joint="valve_joint_2",
                keypoints={"handle_center": RefSpec("site", "handwheel_center_2")},
                metadata={"label": "2号阀门", "location_x": 3.5},
            ),
            "shade_board": ObjectSpec(
                "shade_board",
                kind="tool",
                body="shade_board",
                keypoints={"grasp_center": RefSpec("site", "shade_handle_center")},
            ),
            "shade_block_target": ObjectSpec(
                "shade_block_target", kind="target_zone", site="shade_block_target"
            ),
            "cart_nav_target": ObjectSpec(
                "cart_nav_target", kind="nav_target", site="cart_nav_target"
            ),
            "cart_nav_target_2": ObjectSpec(
                "cart_nav_target_2", kind="nav_target", site="cart_nav_target_2"
            ),
            "cart_nav_valve3": ObjectSpec(
                "cart_nav_valve3", kind="nav_target", site="cart_nav_valve3"
            ),
            "cart_nav_gauge1": ObjectSpec(
                "cart_nav_gauge1", kind="nav_target", site="cart_nav_gauge1"
            ),
            "cart_nav_gauge2": ObjectSpec(
                "cart_nav_gauge2", kind="nav_target", site="cart_nav_gauge2"
            ),
            "cart_nav_loop_right_1": ObjectSpec(
                "cart_nav_loop_right_1", kind="nav_target", site="cart_nav_loop_right_1"
            ),
            "cart_nav_loop_right_2": ObjectSpec(
                "cart_nav_loop_right_2", kind="nav_target", site="cart_nav_loop_right_2"
            ),
            "cart_nav_pipeB_mid": ObjectSpec(
                "cart_nav_pipeB_mid", kind="nav_target", site="cart_nav_pipeB_mid"
            ),
            "cart_nav_loop_left_1": ObjectSpec(
                "cart_nav_loop_left_1", kind="nav_target", site="cart_nav_loop_left_1"
            ),
            "cart_nav_loop_left_2": ObjectSpec(
                "cart_nav_loop_left_2", kind="nav_target", site="cart_nav_loop_left_2"
            ),
            "cart_nav_loop_left_3": ObjectSpec(
                "cart_nav_loop_left_3", kind="nav_target", site="cart_nav_loop_left_3"
            ),
            "valve_3": ObjectSpec(
                "valve_3",
                kind="valve",
                body="valve_body_3",
                joint="valve_joint_3",
                keypoints={"handle_center": RefSpec("site", "handwheel_center_3")},
                metadata={"label": "3号阀门(管道B截止阀)", "location_x": 1.5, "location_y": 2.0},
            ),
            "gauge_1": ObjectSpec(
                "gauge_1",
                kind="gauge",
                body="pressure_gauge_1",
                keypoints={"inspect_point": RefSpec("site", "gauge_inspect_1")},
                metadata={"label": "1号压力表", "location_x": -0.1},
            ),
            "gauge_2": ObjectSpec(
                "gauge_2",
                kind="gauge",
                body="pressure_gauge_2",
                keypoints={"inspect_point": RefSpec("site", "gauge_inspect_2")},
                metadata={"label": "2号压力表", "location_x": 2.8},
            ),
            "gauge_3": ObjectSpec(
                "gauge_3",
                kind="gauge",
                body="pressure_gauge_3",
                keypoints={"inspect_point": RefSpec("site", "gauge_inspect_3")},
                metadata={"label": "3号压力表", "location_x": 1.8, "location_y": 2.0},
            ),
            "glare_light": ObjectSpec("glare_light", kind="light", light="glare_light"),
            "tcp_L": ObjectSpec("tcp_L", kind="tcp", site="tcp"),
            "tcp_R": ObjectSpec("tcp_R", kind="tcp", site="tcp_R"),
        },
        default_cameras=(
            "cam_overview",
            "cam_side",
            "cam_valve",
            "cam_bird",
            "cam_shade",
            "cam_global",
        ),
    ),
}
