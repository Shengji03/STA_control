"""
场景感知模块

从 MuJoCo 仿真中提取全局视觉 (RGB/深度) 和结构化状态信息,
打包为大模型可消费的格式 (base64 图像 + JSON 状态)。

兼容两种渲染方式:
  - 优先使用高层 mujoco.Renderer API
  - 失败时回退到低层 MjrContext + mjr_render API
"""

import base64
import io
from typing import Dict, List, Optional, Tuple

import mujoco
import numpy as np

from ..config.robot import MOBILE_ROBOT

from .object_registry import SceneObjectRegistry
from .world_state import WorldStateBuilder


def _create_offscreen_context(model, width, height):
    """创建低层离屏渲染上下文"""
    gl_ctx = mujoco.GLContext(width, height)
    gl_ctx.make_current()
    scn = mujoco.MjvScene(model, maxgeom=10000)
    ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    viewport = mujoco.MjrRect(0, 0, width, height)
    return gl_ctx, scn, ctx, cam, opt, viewport


class PerceptionModule:
    """MuJoCo 场景感知: 视觉渲染 + 结构化状态提取"""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        camera_names: Optional[List[str]] = None,
        tracked_bodies: Optional[List[str]] = None,
        tracked_sites: Optional[List[str]] = None,
        sensor_config: Optional[Dict] = None,
        resolution: Tuple[int, int] = (1280, 720),
        scene_path: Optional[str] = None,
        registry: Optional[SceneObjectRegistry] = None,
    ):
        """
        Args:
            model:           MuJoCo 模型
            data:            MuJoCo 数据
            camera_names:    要渲染的相机名称列表
            tracked_bodies:  要追踪位姿的刚体名称列表
            tracked_sites:   要追踪位置的站点名称列表 (如 tcp, pinch)
            sensor_config:   传感器配置, 定义各传感器在 sensordata 中的位置, 例如:
                             {'arm_L_joints': (0, 6), 'arm_R_joints': (6, 12),
                              'cart': (12, 15), 'valve_1': (15, 16)}
            resolution:      渲染分辨率 (width, height)
        """
        self.model = model
        self.data = data
        self.scene_path = scene_path
        self.registry = registry or SceneObjectRegistry.from_scene_path(scene_path)
        self.camera_names = camera_names or list(self.registry.default_cameras) or ["cam_global"]
        self.tracked_bodies = tracked_bodies or []
        self.tracked_sites = tracked_sites or []
        self.sensor_config = sensor_config or {}
        self.resolution = resolution

        self._use_high_level = True
        self._renderer = None
        self._low_level = None

        try:
            self._renderer = mujoco.Renderer(
                model, height=resolution[1], width=resolution[0])
            self._renderer.update_scene(data)
            self._renderer.render()
            print("[Perception] 使用高层 Renderer API")
        except Exception as e:
            print(f"[Perception] 高层 Renderer 不可用 ({e}), 回退到低层 API")
            self._use_high_level = False
            self._renderer = None
            self._low_level = _create_offscreen_context(
                model, resolution[0], resolution[1])

        self._body_ids = {}
        for name in self.tracked_bodies:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                self._body_ids[name] = bid

        self._site_ids = {}
        for name in self.tracked_sites:
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
            if sid >= 0:
                self._site_ids[name] = sid

        self._world_state_builder = WorldStateBuilder(
            model,
            data,
            self.registry,
            resolution=resolution,
        )

    # ------------------------------------------------------------------
    # 视觉渲染
    # ------------------------------------------------------------------

    def render_camera(self, camera_name: str) -> np.ndarray:
        """渲染指定相机的 RGB 图像, 返回 (H, W, 3) uint8 数组"""
        if self._use_high_level:
            self._renderer.update_scene(self.data, camera=camera_name)
            return self._renderer.render().copy()
        else:
            return self._render_lowlevel(camera_name)

    def _render_lowlevel(self, camera_name: str) -> np.ndarray:
        """低层 API 渲染"""
        gl_ctx, scn, ctx, cam, opt, viewport = self._low_level
        gl_ctx.make_current()

        cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        cam.fixedcamid = cam_id

        mujoco.mjv_updateScene(
            self.model, self.data, opt, None, cam,
            mujoco.mjtCatBit.mjCAT_ALL, scn)
        mujoco.mjr_render(viewport, scn, ctx)

        w, h = self.resolution
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(rgb, None, viewport, ctx)
        return np.flipud(rgb)

    def render_depth(self, camera_name: str) -> np.ndarray:
        """渲染指定相机的深度图, 返回 (H, W) float32 数组"""
        if self._use_high_level:
            self._renderer.enable_depth_rendering()
            self._renderer.update_scene(self.data, camera=camera_name)
            depth = self._renderer.render().copy()
            self._renderer.disable_depth_rendering()
            return depth
        else:
            return self._render_depth_lowlevel(camera_name)

    def _render_depth_lowlevel(self, camera_name: str) -> np.ndarray:
        """低层 API 深度渲染"""
        gl_ctx, scn, ctx, cam, opt, viewport = self._low_level
        gl_ctx.make_current()

        cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        cam.fixedcamid = cam_id

        mujoco.mjv_updateScene(
            self.model, self.data, opt, None, cam,
            mujoco.mjtCatBit.mjCAT_ALL, scn)
        mujoco.mjr_render(viewport, scn, ctx)

        w, h = self.resolution
        depth = np.zeros((h, w), dtype=np.float32)
        mujoco.mjr_readPixels(None, depth, viewport, ctx)
        return np.flipud(depth)

    # ------------------------------------------------------------------
    # 图像编码
    # ------------------------------------------------------------------

    @staticmethod
    def image_to_base64(image: np.ndarray, fmt: str = "png") -> str:
        """将 numpy RGB 图像编码为 base64 字符串 (用于 API 传输)"""
        from PIL import Image

        img = Image.fromarray(image)
        buf = io.BytesIO()
        img.save(buf, format=fmt.upper())
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    # ------------------------------------------------------------------
    # 结构化状态提取
    # ------------------------------------------------------------------

    def get_world_state(self, camera_names: Optional[List[str]] = None) -> Dict:
        """鏋勫缓闈㈠悜瑙勫垝鍣ㄧ殑瀵硅薄绾т笘鐣岀姸鎬?"""
        selected_cameras = camera_names or self.camera_names
        return self._world_state_builder.build(selected_cameras)

    def get_structured_state(self, world_state: Optional[Dict] = None) -> Dict:
        """提取完整场景状态, 返回 JSON 可序列化的字典"""
        state = {
            "sim_time": round(float(self.data.time), 4),
        }

        for label, (start, end) in self.sensor_config.items():
            raw = self.data.sensordata[start:end].copy()
            if "joint" in label:
                state[label] = {
                    "degrees": np.degrees(raw).round(2).tolist(),
                    "radians": raw.round(4).tolist(),
                }
            else:
                state[label] = raw.round(4).tolist()

        bodies = {}
        for name, bid in self._body_ids.items():
            bodies[name] = {
                "pos": self.data.xpos[bid].round(4).tolist(),
                "quat": self.data.xquat[bid].round(4).tolist(),
            }
        if bodies:
            state["bodies"] = bodies

        sites = {}
        for name, sid in self._site_ids.items():
            sites[name] = {
                "pos": self.data.site_xpos[sid].round(4).tolist(),
            }
        if sites:
            state["sites"] = sites

        world_state = world_state or self.get_world_state()
        if world_state["objects"]:
            state["objects"] = world_state["objects"]
        if world_state["cameras"]:
            state["cameras"] = world_state["cameras"]

        return state

    @staticmethod
    def _preferred_position(obj_state: Dict) -> Optional[np.ndarray]:
        pose = obj_state.get("pose", {})
        pos = pose.get("pos")
        if pos is not None:
            return np.asarray(pos, dtype=np.float64)

        for key_state in obj_state.get("keypoints", {}).values():
            key_pos = key_state.get("pos")
            if key_pos is not None:
                return np.asarray(key_pos, dtype=np.float64)

        return None

    @staticmethod
    def _visible_cameras(key_state: Dict) -> List[str]:
        visible = []
        for cam_name, view in key_state.get("views", {}).items():
            if view.get("visible") and view.get("in_frame"):
                visible.append(cam_name)
        return visible

    _ARM_L_OFFSET = np.array(MOBILE_ROBOT.left_offset)
    _ARM_R_OFFSET = np.array(MOBILE_ROBOT.right_offset)
    _ARM_REACH = MOBILE_ROBOT.arm_reach

    def _build_planner_state(self, world_state: Dict) -> Dict:
        planner_state = {
            "scene_name": self.registry.scene_name,
            "sim_time": world_state.get("sim_time", round(float(self.data.time), 4)),
            "robot": {},
            "objects": {},
            "cameras": {},
            "relations": {},
        }

        cart_cfg = self.sensor_config.get('cart_pos')
        if cart_cfg:
            start, end = cart_cfg
            cart_raw = self.data.sensordata[start:end].copy()
            cx, cy = float(cart_raw[0]), float(cart_raw[1])
            cz = float(cart_raw[2]) if len(cart_raw) > 2 else 0.0
            arm_L_base = [round(cx + self._ARM_L_OFFSET[0], 4),
                          round(cy + self._ARM_L_OFFSET[1], 4),
                          round(self._ARM_L_OFFSET[2], 4)]
            arm_R_base = [round(cx + self._ARM_R_OFFSET[0], 4),
                          round(cy + self._ARM_R_OFFSET[1], 4),
                          round(self._ARM_R_OFFSET[2], 4)]
            planner_state["cart"] = {
                "position": [round(cx, 4), round(cy, 4), round(cz, 4)],
                "arm_L_base": arm_L_base,
                "arm_R_base": arm_R_base,
                "arm_reach_m": self._ARM_REACH,
                "note": "目标必须在对应臂基座的 arm_reach_m 范围内才可达, 否则需先用 NavSkill 移动小车",
            }

        tip_positions = {}
        for obj_name, obj_state in world_state.get("objects", {}).items():
            if obj_state.get("kind") != "tcp":
                continue
            pose = obj_state.get("pose", {})
            planner_state["robot"][obj_name] = {
                "position": pose.get("pos"),
                "orientation": pose.get("mat"),
            }
            tip_pos = pose.get("pos")
            if tip_pos is not None:
                tip_positions[obj_name] = np.asarray(tip_pos, dtype=np.float64)

        for cam_name, cam_state in world_state.get("cameras", {}).items():
            planner_state["cameras"][cam_name] = {
                "position": cam_state.get("pos"),
                "fovy_deg": cam_state.get("fovy_deg"),
                "resolution": cam_state.get("resolution"),
            }

        nearest = {tip_name: {"object": None, "distance_m": None} for tip_name in tip_positions}

        for obj_name, obj_state in world_state.get("objects", {}).items():
            if obj_state.get("kind") == "tcp":
                continue

            summary = {
                "kind": obj_state.get("kind"),
            }
            if "metadata" in obj_state:
                summary["metadata"] = obj_state["metadata"]

            pose = obj_state.get("pose", {})
            if pose.get("pos") is not None:
                summary["position"] = pose["pos"]
            if pose.get("quat") is not None:
                summary["orientation"] = pose["quat"]

            joint_state = obj_state.get("joint")
            if joint_state and joint_state.get("position") is not None:
                summary["joint_position"] = joint_state["position"]
            elif joint_state and joint_state.get("qpos") is not None:
                summary["joint_qpos"] = joint_state["qpos"]

            keypoints = {}
            visible_cameras = set()
            for key_name, key_state in obj_state.get("keypoints", {}).items():
                visible = self._visible_cameras(key_state)
                key_summary = {
                    "position": key_state.get("pos"),
                    "visible_cameras": visible,
                }
                for cam_name, view in key_state.get("views", {}).items():
                    if view.get("visible") and view.get("in_frame"):
                        key_summary["best_view"] = {
                            "camera": cam_name,
                            "pixel": view.get("pixel"),
                            "depth_m": view.get("depth_m"),
                        }
                        break
                keypoints[key_name] = key_summary
                visible_cameras.update(visible)

            if keypoints:
                summary["keypoints"] = keypoints
            if visible_cameras:
                summary["visible_cameras"] = sorted(visible_cameras)

            target_pos = self._preferred_position(obj_state)
            if target_pos is not None and target_pos.size == 3:
                distances = {}
                for tip_name, tip_pos in tip_positions.items():
                    distance = round(float(np.linalg.norm(target_pos - tip_pos)), 6)
                    distances[tip_name] = distance
                    nearest_entry = nearest[tip_name]
                    if nearest_entry["distance_m"] is None or distance < nearest_entry["distance_m"]:
                        nearest_entry["object"] = obj_name
                        nearest_entry["distance_m"] = distance
                if distances:
                    summary["distance_to_tips_m"] = distances

                if cart_cfg:
                    arm_L_world = np.array(arm_L_base)
                    arm_R_world = np.array(arm_R_base)
                    dist_L = round(float(np.linalg.norm(target_pos - arm_L_world)), 4)
                    dist_R = round(float(np.linalg.norm(target_pos - arm_R_world)), 4)
                    summary["reachable_by_L"] = dist_L <= self._ARM_REACH
                    summary["reachable_by_R"] = dist_R <= self._ARM_REACH
                    summary["distance_to_arm_L_base_m"] = dist_L
                    summary["distance_to_arm_R_base_m"] = dist_R
                    if not summary["reachable_by_L"] and not summary["reachable_by_R"]:
                        nav_x = round(float(target_pos[0]), 4)
                        nav_y = round(float(target_pos[1]) - 0.4, 4)
                        summary["suggested_nav_target"] = [nav_x, nav_y]
                        summary["nav_required"] = True

            planner_state["objects"][obj_name] = summary

        planner_state["relations"] = {
            tip_name: relation
            for tip_name, relation in nearest.items()
            if relation["object"] is not None
        }
        return planner_state

    # ------------------------------------------------------------------
    # 打包输出
    # ------------------------------------------------------------------

    def get_scene_snapshot(self, include_depth: bool = False) -> Dict:
        """获取完整场景快照: RGB 图像 (base64) + 结构化状态

        Args:
            include_depth: 是否包含深度图

        Returns:
            {
                'timestamp': float,
                'images': {'cam_global': 'data:image/png;base64,...', ...},
                'depth':  {'cam_global': ndarray, ...},  (可选)
                'state':  { ... 结构化状态 ... },
            }
        """
        world_state = self.get_world_state()
        snapshot = {
            "timestamp": round(float(self.data.time), 4),
            "scene_name": self.registry.scene_name,
            "images": {},
            "state": self.get_structured_state(world_state=world_state),
            "world_state": world_state,
            "planner_state": self._build_planner_state(world_state),
        }

        for cam in self.camera_names:
            rgb = self.render_camera(cam)
            b64 = self.image_to_base64(rgb)
            snapshot["images"][cam] = f"data:image/png;base64,{b64}"

        if include_depth:
            snapshot["depth"] = {}
            for cam in self.camera_names:
                snapshot["depth"][cam] = self.render_depth(cam)

        return snapshot

    def get_image_for_api(self, camera_name: str = None) -> str:
        """获取单张相机图像的 base64 data URL, 可直接用于 OpenAI API"""
        cam = camera_name or self.camera_names[0]
        rgb = self.render_camera(cam)
        b64 = self.image_to_base64(rgb, fmt="jpeg")
        return f"data:image/jpeg;base64,{b64}"
