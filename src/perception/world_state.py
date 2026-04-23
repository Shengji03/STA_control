from __future__ import annotations

from typing import Dict, List, Optional

import mujoco
import numpy as np

from .object_registry import ObjectSpec, RefSpec, SceneObjectRegistry


_JOINT_QPOS_WIDTH = {
    mujoco.mjtJoint.mjJNT_FREE: 7,
    mujoco.mjtJoint.mjJNT_BALL: 4,
    mujoco.mjtJoint.mjJNT_SLIDE: 1,
    mujoco.mjtJoint.mjJNT_HINGE: 1,
}

_JOINT_QVEL_WIDTH = {
    mujoco.mjtJoint.mjJNT_FREE: 6,
    mujoco.mjtJoint.mjJNT_BALL: 3,
    mujoco.mjtJoint.mjJNT_SLIDE: 1,
    mujoco.mjtJoint.mjJNT_HINGE: 1,
}


class WorldStateBuilder:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        registry: Optional[SceneObjectRegistry] = None,
        resolution: tuple[int, int] = (1280, 720),
    ) -> None:
        self.model = model
        self.data = data
        self.registry = registry or SceneObjectRegistry.empty()
        self.resolution = resolution

        self._body_ids = {
            name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in self._iter_names("body")
        }
        self._site_ids = {
            name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
            for name in self._iter_names("site")
        }
        self._joint_ids = {
            name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self._iter_names("joint")
        }
        self._light_ids = {
            name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, name)
            for name in self._iter_names("light")
        }
        self._camera_ids = {
            name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            for name in self._iter_names("camera")
        }

    def _iter_names(self, ref_type: str):
        if ref_type == "camera":
            for name in self.registry.default_cameras:
                yield name
        for spec in self.registry.objects.values():
            name = getattr(spec, ref_type)
            if name:
                yield name
            for keypoint in spec.keypoints.values():
                if keypoint.ref_type == ref_type:
                    yield keypoint.name

    def build(self, camera_names: Optional[List[str]] = None) -> Dict:
        world_state = {
            "sim_time": round(float(self.data.time), 4),
            "objects": {},
            "cameras": {},
        }

        selected_cameras = tuple(camera_names or self.registry.default_cameras)
        for obj_name, spec in self.registry.objects.items():
            world_state["objects"][obj_name] = self._build_object_state(spec, selected_cameras)

        for cam_name in selected_cameras:
            camera_state = self._build_camera_state(cam_name)
            if camera_state is not None:
                world_state["cameras"][cam_name] = camera_state

        return world_state

    def _build_object_state(
        self,
        spec: ObjectSpec,
        camera_names: tuple[str, ...],
    ) -> Dict:
        obj_state = {
            "kind": spec.kind,
        }
        if spec.metadata:
            obj_state["metadata"] = dict(spec.metadata)

        anchor_state = self._build_anchor_state(spec)
        if anchor_state is not None:
            obj_state.update(anchor_state)

        if spec.joint:
            joint_state = self._build_joint_state(spec.joint)
            if joint_state is not None:
                obj_state["joint"] = joint_state

        if spec.keypoints:
            keypoints = {}
            anchor_body_id = anchor_state.get("_body_id") if anchor_state else -1
            for key_name, ref in spec.keypoints.items():
                key_state = self._build_ref_state(ref)
                if key_state is None:
                    continue
                pos = np.array(key_state["pos"], dtype=np.float64)
                views = {}
                for cam_name in camera_names:
                    projection = self._project_point(cam_name, pos)
                    if projection is None:
                        continue
                    projection["visible"] = self._is_visible_from_camera(
                        cam_name, pos, anchor_body_id
                    )
                    views[cam_name] = projection
                if views:
                    key_state["views"] = views
                keypoints[key_name] = key_state
            if keypoints:
                obj_state["keypoints"] = keypoints

        if spec.light:
            light_state = self._build_light_state(spec.light)
            if light_state is not None:
                obj_state["light"] = light_state

        obj_state.pop("_body_id", None)
        return obj_state

    def _build_anchor_state(self, spec: ObjectSpec) -> Optional[Dict]:
        if spec.body:
            body_id = self._body_ids.get(spec.body, -1)
            if body_id >= 0:
                return {
                    "pose": {
                        "pos": self.data.xpos[body_id].round(6).tolist(),
                        "quat": self.data.xquat[body_id].round(6).tolist(),
                        "mat": self.data.xmat[body_id].reshape(3, 3).round(6).tolist(),
                    },
                    "_body_id": body_id,
                }

        if spec.site:
            site_state = self._build_ref_state(RefSpec("site", spec.site))
            if site_state is not None:
                return {
                    "pose": site_state,
                }

        if spec.camera:
            camera_state = self._build_camera_state(spec.camera)
            if camera_state is not None:
                return {
                    "pose": {
                        "pos": camera_state["pos"],
                        "mat": camera_state["mat"],
                    }
                }

        if spec.light:
            light_state = self._build_light_state(spec.light)
            if light_state is not None:
                return {
                    "pose": {
                        "pos": light_state["pos"],
                    }
                }

        return None

    def _build_ref_state(self, ref: RefSpec) -> Optional[Dict]:
        if ref.ref_type == "site":
            site_id = self._site_ids.get(ref.name, -1)
            if site_id < 0:
                return None
            return {
                "pos": self.data.site_xpos[site_id].round(6).tolist(),
                "mat": self.data.site_xmat[site_id].reshape(3, 3).round(6).tolist(),
            }

        if ref.ref_type == "body":
            body_id = self._body_ids.get(ref.name, -1)
            if body_id < 0:
                return None
            return {
                "pos": self.data.xpos[body_id].round(6).tolist(),
                "quat": self.data.xquat[body_id].round(6).tolist(),
                "mat": self.data.xmat[body_id].reshape(3, 3).round(6).tolist(),
            }

        return None

    def _build_joint_state(self, joint_name: str) -> Optional[Dict]:
        joint_id = self._joint_ids.get(joint_name, -1)
        if joint_id < 0:
            return None

        joint_type = self.model.jnt_type[joint_id]
        qpos_width = _JOINT_QPOS_WIDTH[joint_type]
        qvel_width = _JOINT_QVEL_WIDTH[joint_type]
        qpos_adr = self.model.jnt_qposadr[joint_id]
        qvel_adr = self.model.jnt_dofadr[joint_id]
        qpos = self.data.qpos[qpos_adr:qpos_adr + qpos_width]
        qvel = self.data.qvel[qvel_adr:qvel_adr + qvel_width]

        state = {
            "qpos": np.asarray(qpos, dtype=np.float64).round(6).tolist(),
            "qvel": np.asarray(qvel, dtype=np.float64).round(6).tolist(),
        }
        if qpos_width == 1:
            state["position"] = round(float(qpos[0]), 6)
        if qvel_width == 1:
            state["velocity"] = round(float(qvel[0]), 6)
        return state

    def _build_camera_state(self, camera_name: str) -> Optional[Dict]:
        cam_id = self._camera_ids.get(camera_name, -1)
        if cam_id < 0:
            cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
            self._camera_ids[camera_name] = cam_id
        if cam_id < 0:
            return None

        width, height = self.resolution
        return {
            "pos": self.data.cam_xpos[cam_id].round(6).tolist(),
            "mat": self.data.cam_xmat[cam_id].reshape(3, 3).round(6).tolist(),
            "fovy_deg": round(float(self.model.cam_fovy[cam_id]), 6),
            "resolution": [int(width), int(height)],
        }

    def _build_light_state(self, light_name: str) -> Optional[Dict]:
        light_id = self._light_ids.get(light_name, -1)
        if light_id < 0:
            return None

        return {
            "pos": self.data.light_xpos[light_id].round(6).tolist(),
            "dir": self.data.light_xdir[light_id].round(6).tolist(),
        }

    def _project_point(self, camera_name: str, point_world: np.ndarray) -> Optional[Dict]:
        cam_id = self._camera_ids.get(camera_name, -1)
        if cam_id < 0:
            cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
            self._camera_ids[camera_name] = cam_id
        if cam_id < 0:
            return None

        cam_pos = self.data.cam_xpos[cam_id]
        cam_rot = self.data.cam_xmat[cam_id].reshape(3, 3)
        point_cam = cam_rot.T @ (point_world - cam_pos)

        forward = -float(point_cam[2])
        if forward <= 1e-8:
            return {
                "in_frame": False,
                "pixel": None,
                "depth_m": None,
                "camera_xyz": point_cam.round(6).tolist(),
            }

        width, height = self.resolution
        fovy = np.deg2rad(float(self.model.cam_fovy[cam_id]))
        fy = 0.5 * height / np.tan(0.5 * fovy)
        fx = fy

        u = width * 0.5 + fx * float(point_cam[0]) / forward
        v = height * 0.5 - fy * float(point_cam[1]) / forward
        in_frame = (0.0 <= u < width) and (0.0 <= v < height)

        return {
            "in_frame": bool(in_frame),
            "pixel": [round(u, 3), round(v, 3)],
            "depth_m": round(forward, 6),
            "camera_xyz": point_cam.round(6).tolist(),
        }

    def _is_visible_from_camera(
        self,
        camera_name: str,
        point_world: np.ndarray,
        bodyexclude: int,
    ) -> bool:
        cam_id = self._camera_ids.get(camera_name, -1)
        if cam_id < 0:
            cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
            self._camera_ids[camera_name] = cam_id
        if cam_id < 0:
            return False

        cam_pos = self.data.cam_xpos[cam_id]
        direction = point_world - cam_pos
        distance = float(np.linalg.norm(direction))
        if distance <= 1e-8:
            return True

        geomid = np.array([-1], dtype=np.int32)
        geomgroup = np.ones(6, dtype=np.uint8)
        hit_dist = mujoco.mj_ray(
            self.model,
            self.data,
            cam_pos.astype(np.float64),
            (direction / distance).astype(np.float64),
            geomgroup,
            1,
            int(bodyexclude),
            geomid,
        )
        return hit_dist < 0 or hit_dist >= distance - 1e-3


def build_world_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    scene_path: Optional[str] = None,
    camera_names: Optional[List[str]] = None,
    resolution: tuple[int, int] = (1280, 720),
) -> Dict:
    registry = SceneObjectRegistry.from_scene_path(scene_path)
    builder = WorldStateBuilder(model, data, registry, resolution=resolution)
    return builder.build(camera_names=camera_names)
