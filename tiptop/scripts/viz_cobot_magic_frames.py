"""Inspect the complete Cobot wrist-camera/M2T2/TCP frame chain in Rerun.

The script is deliberately offline: it never opens a Cobot RPC connection and
never commands the robot.  Supply a saved raw M2T2 4x4 pose and, optionally,
the six-joint IK result from a planner run to inspect a real candidate.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import rerun as rr
import tyro

from cutamp.robots.cobot_magic import cobot_magic_home, load_cobot_magic_rerun
from cutamp.robots.cobot_magic_collision import load_cobot_magic_collision_urdf
from cutamp.robots.cobot_magic_frames import COBOT_MAGIC_GRIPPER_BASE_LINK, COBOT_MAGIC_TCP_LINK
from tiptop.config import load_tcp_from_camera, tiptop_cfg
from tiptop.perception.m2t2 import m2t2_grasp_from_tcp
from tiptop.viz_utils import get_gripper_mesh


_log = logging.getLogger(__name__)


def _load_pose(path: Path) -> np.ndarray:
    pose = np.asarray(np.load(path), dtype=np.float64)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise ValueError(f"Expected a finite (4, 4) raw M2T2 pose in {path}, got {pose.shape}")
    if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError(f"Raw M2T2 pose in {path} is not a homogeneous transform")
    return pose


def _log_frame(path: str, world_from_frame: np.ndarray, *, axis_length: float = 0.05) -> None:
    rr.log(
        path,
        rr.Transform3D(
            translation=world_from_frame[:3, 3],
            mat3x3=world_from_frame[:3, :3],
            axis_length=axis_length,
        ),
        static=True,
    )


def _log_proxy_mesh(path: str, world_from_proxy: np.ndarray, color: np.ndarray) -> None:
    """Log the existing lightweight M2T2 visual proxy without claiming it is a URDF mesh."""
    mesh = get_gripper_mesh()
    vertices = np.asarray(mesh.vertices)
    vertices_h = np.c_[vertices, np.ones(len(vertices))]
    world_vertices = (world_from_proxy @ vertices_h.T).T[:, :3]
    rr.log(
        path,
        rr.Mesh3D(
            vertex_positions=world_vertices,
            triangle_indices=np.asarray(mesh.triangles),
            vertex_colors=np.tile(color, (len(vertices), 1)),
        ),
        static=True,
    )


def _world_from_link(urdf, link: str) -> np.ndarray:
    return np.asarray(urdf.scene.graph.get(frame_to=link, frame_from=urdf.base_link)[0], dtype=np.float64)


def viz_cobot_magic_frames(
    raw_m2t2_pose: Path | None = None,
    ik_joints: list[float] | None = None,
    pointcloud: Path | None = None,
    output_rrd: Path = Path("/tmp/cobot_magic_frames.rrd"),
    spawn: bool = True,
) -> None:
    """Log Cobot mesh plus base, gripper, TCP, camera and grasp frames.

    ``raw_m2t2_pose`` is a ``.npy`` (4,4) ``world_from_m2t2_grasp`` matrix.
    ``ik_joints`` must be six arm values.  When no raw pose is supplied, the
    script synthesizes one from the displayed TCP so the complete chain is
    inspectable without an M2T2 server.
    """
    cfg = tiptop_cfg()
    if str(cfg.robot.type) != "cobot_magic":
        raise ValueError("viz_cobot_magic_frames only supports robot.type=cobot_magic")
    q_arm = np.asarray(cobot_magic_home if ik_joints is None else ik_joints, dtype=np.float64)
    if q_arm.shape != (6,):
        raise ValueError(f"ik_joints must contain only joint1..joint6, got {q_arm.shape}")

    rr.init("cobot_magic_frames", spawn=spawn)
    rr.save(str(output_rrd))

    # The actual Cobot URDF mesh and its mimic-driven visual gripper.
    robot_rr = load_cobot_magic_rerun(load_mesh=True)
    robot_rr.set_joint_positions(q_arm)
    urdf = load_cobot_magic_collision_urdf()
    urdf.update_cfg(np.asarray([*q_arm, 0.03], dtype=np.float64))

    world_from_base = np.eye(4)
    world_from_gripper_base = _world_from_link(urdf, COBOT_MAGIC_GRIPPER_BASE_LINK)
    world_from_tcp = _world_from_link(urdf, COBOT_MAGIC_TCP_LINK)
    tcp_from_camera = load_tcp_from_camera(str(cfg.cameras.hand.serial), "cobot_magic")
    world_from_camera = world_from_tcp @ tcp_from_camera

    # A raw M2T2 pose is converted once, at the explicit TCP boundary.
    m2t2_from_tcp = m2t2_grasp_from_tcp("cobot_magic")
    if raw_m2t2_pose is None:
        world_from_m2t2_grasp = world_from_tcp @ np.linalg.inv(m2t2_from_tcp)
        _log.info("No raw M2T2 pose provided; synthesized one whose TCP target is the displayed FK pose")
    else:
        world_from_m2t2_grasp = _load_pose(raw_m2t2_pose)
    world_from_tcp_target = world_from_m2t2_grasp @ m2t2_from_tcp

    _log_frame("world/base_link", world_from_base)
    _log_frame("robot/gripper_base", world_from_gripper_base)
    _log_frame("robot/tool_center_point", world_from_tcp)
    _log_frame("camera/optical_frame", world_from_camera)
    _log_frame("grasps/m2t2_raw", world_from_m2t2_grasp)
    _log_frame("grasps/tcp_target", world_from_tcp_target)
    # This is an actual FK result for the six joints displayed by the mesh.
    _log_frame("grasps/ik_result", world_from_tcp)

    _log_proxy_mesh("grasps/m2t2_raw/proxy_mesh", world_from_m2t2_grasp, np.array([255, 160, 0]))
    _log_proxy_mesh("grasps/tcp_target/proxy_mesh", world_from_tcp_target, np.array([0, 220, 255]))

    if pointcloud is not None:
        import open3d as o3d

        pcd = o3d.io.read_point_cloud(str(pointcloud))
        rr.log("world/object_pointcloud", rr.Points3D(positions=np.asarray(pcd.points), colors=np.asarray(pcd.colors)))

    _log.info("Saved frame-chain Rerun recording to %s", output_rrd)
    _log.info("Compare grasps/tcp_target against grasps/ik_result; they coincide for the synthesized default pose.")


def entrypoint() -> None:
    tyro.cli(viz_cobot_magic_frames)


if __name__ == "__main__":
    entrypoint()
