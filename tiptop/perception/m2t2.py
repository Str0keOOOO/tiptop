import logging
import os
import time
from functools import cache

import aiohttp
import numpy as np
import requests
from jaxtyping import Float
from scipy.spatial.transform import Rotation

from tiptop.utils import ServerHealthCheckError

_log = logging.getLogger(__name__)

# M2T2's public action decoder constructs every raw grasp with this training
# gripper depth.  It is *not* a robot TCP measurement:
# ``raw_origin = contact_midpoint - 0.1034 * approach_direction``.
M2T2_TRAINING_GRIPPER_DEPTH_M = 0.1034


def _transform_from_frame_config(transform_cfg) -> np.ndarray:
    """Parse a named parent_from_child transform from TiPToP configuration."""
    required = ("parent_frame", "child_frame", "convention", "translation", "quaternion_wxyz")
    missing = [key for key in required if key not in transform_cfg]
    if missing:
        raise ValueError(f"M2T2 TCP transform is missing fields: {missing}")
    if transform_cfg.convention != "parent_from_child":
        raise ValueError(f"Unsupported M2T2 transform convention: {transform_cfg.convention!r}")
    translation = np.asarray(transform_cfg.translation, dtype=np.float64)
    quaternion_wxyz = np.asarray(transform_cfg.quaternion_wxyz, dtype=np.float64)
    if translation.shape != (3,) or not np.all(np.isfinite(translation)):
        raise ValueError(f"M2T2 transform translation must be three finite values, got {translation}")
    if quaternion_wxyz.shape != (4,) or not np.all(np.isfinite(quaternion_wxyz)):
        raise ValueError(f"M2T2 transform quaternion must be four finite values, got {quaternion_wxyz}")
    quaternion_norm = np.linalg.norm(quaternion_wxyz)
    if not np.isclose(quaternion_norm, 1.0, atol=1e-8):
        raise ValueError(f"M2T2 transform quaternion must be unit length, got norm={quaternion_norm}")
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_quat(quaternion_wxyz[[1, 2, 3, 0]]).as_matrix()
    matrix[:3, 3] = translation
    return matrix


@cache
def _legacy_m2t2_grasp_from_tcp() -> np.ndarray:
    """Original Panda-oriented transform retained only for non-Cobot robots."""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = [0.0, 0.0, M2T2_TRAINING_GRIPPER_DEPTH_M]
    transform[:3, :3] = Rotation.from_euler("xyz", np.array([np.pi, 0.0, -np.pi / 2.0])).as_matrix()
    return transform


def m2t2_grasp_from_tcp(robot_type: str | None = None) -> np.ndarray:
    """Return ``m2t2_grasp_from_tcp`` for the selected robot.

    The returned matrix maps the desired Cobot TCP into M2T2's raw grasp
    coordinates.  Therefore a raw prediction is converted with
    ``world_from_tcp_target = world_from_m2t2_grasp @ m2t2_grasp_from_tcp``.
    Cobot deliberately has a robot-specific configuration and never reuses
    the legacy Panda rotation.
    """
    from tiptop.config import tiptop_cfg

    cfg = tiptop_cfg()
    if robot_type is None:
        robot_type = str(cfg.robot.type)
    if robot_type != "cobot_magic":
        return _legacy_m2t2_grasp_from_tcp().copy()

    transform_cfg = cfg.robot.frames.m2t2_grasp_from_tcp
    if transform_cfg.parent_frame != "m2t2_grasp" or transform_cfg.child_frame != "tool_center_point":
        raise ValueError(
            "Cobot M2T2 transform must be explicitly m2t2_grasp_from_tool_center_point, "
            f"got {transform_cfg.parent_frame!r}_from_{transform_cfg.child_frame!r}"
        )
    transform = _transform_from_frame_config(transform_cfg)
    # This equality is a validation of M2T2's output convention, not an
    # assertion about the independent URDF TCP offset.
    if not np.isclose(transform[2, 3], M2T2_TRAINING_GRIPPER_DEPTH_M, atol=1e-8):
        raise ValueError(
            "Cobot m2t2_grasp_from_tcp must use M2T2's documented raw-grasp depth; "
            "change it only together with an M2T2 convention migration"
        )
    return transform


def m2t2_to_tiptop_transform() -> np.ndarray:
    """Deprecated compatibility alias for :func:`m2t2_grasp_from_tcp`."""
    return m2t2_grasp_from_tcp()


def _build_payload(
    scene_xyz: Float[np.ndarray, "n 3"],
    scene_rgb: Float[np.ndarray, "n 3"],
    grasp_threshold: float,
    num_points: int,
    num_runs: int,
    apply_bounds: bool,
) -> dict:
    return {
        "pointcloud": {
            "points": scene_xyz.tolist(),
            "rgb": scene_rgb.tolist(),
        },
        "num_points": num_points,
        "num_runs": num_runs,
        "mask_thresh": grasp_threshold,
        "apply_bounds": apply_bounds,
    }


def _process_m2t2_response(result: dict, num_grasps: int | None) -> dict:
    """Process M2T2 response and return structured grasp outputs."""
    grasps_list = result.get("grasps", [])
    confidences_list = result.get("grasp_confidence", [])
    contacts_list = result.get("grasp_contacts", [])
    outputs = {}

    for i, (grasps, confidences, contacts) in enumerate(zip(grasps_list, confidences_list, contacts_list)):
        label = f"object_{i}"
        if len(grasps) == 0:
            outputs[label] = {
                "poses": np.array([]).reshape(0, 4, 4),
                "confidences": np.array([]),
                "contacts": np.array([]),
            }
        else:
            poses = np.array(grasps)
            confs = np.array(confidences)
            conts = np.array(contacts)

            if num_grasps is not None and len(poses) > num_grasps:
                top_indices = np.argsort(confs)[-num_grasps:]
                poses = poses[top_indices]
                confs = confs[top_indices]
                conts = conts[top_indices]

            outputs[label] = {
                "poses": poses,
                "confidences": confs,
                "contacts": conts,
            }

    return outputs


def generate_grasps(
    server_url: str,
    scene_xyz: Float[np.ndarray, "n 3"],
    scene_rgb: Float[np.ndarray, "n 3"],
    grasp_threshold: float = 0.035,
    num_grasps: int = 200,
    num_points: int = 16384,
    num_runs: int = 5,
    apply_bounds: bool = True,
):
    """
    Generate grasps from point cloud using M2T2 server (synchronous version).
    Note: the coordinate frame of the grasps are in M2T2's convention.
    """
    start_time = time.perf_counter()
    payload = _build_payload(scene_xyz, scene_rgb, grasp_threshold, num_points, num_runs, apply_bounds)
    endpoint = os.path.join(server_url.rstrip("/"), "predict")

    _log.debug(f"Sending inference request to M2T2 server at {endpoint}")
    response = requests.post(endpoint, json=payload, timeout=500)
    result = response.json()

    outputs = _process_m2t2_response(result, num_grasps)
    duration = time.perf_counter() - start_time
    _log.info(f"M2T2 inference time={duration:.2f}s")
    return outputs


async def generate_grasps_async(
    session: aiohttp.ClientSession,
    server_url: str,
    scene_xyz: Float[np.ndarray, "n 3"],
    scene_rgb: Float[np.ndarray, "n 3"],
    grasp_threshold: float = 0.035,
    num_grasps: int = 200,
    num_points: int = 16384,
    num_runs: int = 5,
    apply_bounds: bool = True,
):
    """
    Generate grasps from point cloud using M2T2 server (async version).
    Note: the coordinate frame of the grasps are in M2T2's convention.
    """
    start_time = time.perf_counter()
    payload = _build_payload(scene_xyz, scene_rgb, grasp_threshold, num_points, num_runs, apply_bounds)
    endpoint = os.path.join(server_url.rstrip("/"), "predict")

    _log.debug(f"Sending inference request to M2T2 server at {endpoint}")
    async with session.post(endpoint, json=payload, timeout=aiohttp.ClientTimeout(total=30.0)) as response:
        result = await response.json()

    outputs = _process_m2t2_response(result, num_grasps)
    duration = time.perf_counter() - start_time
    _log.info(f"M2T2 inference time={duration:.2f}s")
    return outputs


async def check_health_status(session: aiohttp.ClientSession, server_url: str):
    """Calls the M2T2 server health status endpoint."""
    endpoint = os.path.join(server_url.rstrip("/"), "health")
    try:
        async with session.get(endpoint, timeout=5.0) as response:
            response.raise_for_status()
            health_data = await response.json()
            status = health_data["status"]

            if status != "healthy":
                _log.error(f"M2T2 health check failed at {server_url}")
                raise ServerHealthCheckError(f"{server_url} returned status: {status}")

            _log.info(f"✓ M2T2 server is healthy")
    except aiohttp.ClientError as e:
        _log.error(f"Health check failed for M2T2")
        raise ServerHealthCheckError(f"M2T2 is unreachable: {e}") from e
