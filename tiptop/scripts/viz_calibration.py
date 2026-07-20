import logging
import time

import numpy as np
import rerun as rr
from curobo.types.base import TensorDeviceType
from cutamp.robots import (
    load_cobot_magic_container,
    load_fr3_franka_container,
    load_fr3_robotiq_container,
    load_panda_container,
    load_panda_robotiq_container,
    load_ur5_container,
)

from tiptop.config import load_tcp_from_camera, tiptop_cfg
from tiptop.perception.cameras import RemoteRealsenseCamera, get_hand_camera
from tiptop.perception.cameras.rs_camera import RealsenseIntrinsics, rs_infer_depth
from tiptop.perception.utils import depth_to_xyz
from tiptop.utils import get_robot_client, get_robot_rerun, patch_log_level, setup_logging

_log = logging.getLogger(__name__)


def _depth_for_visualization(frame, intrinsics: RealsenseIntrinsics | None) -> np.ndarray:
    """Return RGB-aligned metres without requiring the optional device depth stream.

    The remote Cobot RealSense contract intentionally sends RGB and the two IR
    images, not D435 onboard depth.  Reuse TiPToP's normal FoundationStereo
    path in that case; its output is already projected onto the RGB grid.
    """
    if frame.depth is not None:
        return frame.depth.copy()
    if intrinsics is None:
        raise RuntimeError(
            "Calibration visualization needs depth, but this camera does not provide device depth "
            "or RealSense IR intrinsics for FoundationStereo."
        )
    return rs_infer_depth(frame, intrinsics)


def viz_calibration(rr_spawn: bool = True, viz_freq: float = 5.0, max_time: float = 60.0):
    """
    Visualize hand camera calibration with robot in rerun.

    Args:
        rr_spawn: Spawn rerun viewer. You should only set to False if you're connecting to remote visualizer.
        viz_freq: Visualization loop frequency in Hz.
        max_time: Maximum visualization time in seconds before automatically stopping. Used to prevent the script from
            running too long and logging and crazy amount of data to rerun.
    """
    setup_logging()
    rr.init("viz_calibration", spawn=rr_spawn)
    rr.save("/tmp/viz_calibration.rrd")
    # Connect to robot
    client = get_robot_client()
    robot_rr = get_robot_rerun()

    # Setup wrist camera
    # Cobot_Magic follows TiPToP's RGB + IR stereo contract.  Do not request
    # the optional D435 depth fields from its RPC server.
    camera_is_remote_realsense = str(tiptop_cfg().cameras.hand.type) == "remote_realsense"
    cam = get_hand_camera(depth=not camera_is_remote_realsense)
    realsense_intrinsics = cam.get_intrinsics() if isinstance(cam, RemoteRealsenseCamera) else None
    tcp_from_camera = load_tcp_from_camera(cam.serial)

    cfg = tiptop_cfg()
    tensor_args = TensorDeviceType()
    with patch_log_level("curobo", logging.ERROR):
        if cfg.robot.type == "fr3_robotiq":
            robot_container = load_fr3_robotiq_container(tensor_args)
        elif cfg.robot.type == "fr3":
            robot_container = load_fr3_franka_container(tensor_args)
        elif cfg.robot.type == "panda":
            robot_container = load_panda_container(tensor_args)
        elif cfg.robot.type == "panda_robotiq":
            robot_container = load_panda_robotiq_container(tensor_args)
        elif cfg.robot.type == "ur5":
            robot_container = load_ur5_container(tensor_args)
        elif cfg.robot.type == "cobot_magic":
            robot_container = load_cobot_magic_container(tensor_args)
        else:
            raise ValueError(f"Unknown robot type: {cfg.robot.type}")

    start_time = time.perf_counter()
    sleep_time = 1.0 / viz_freq
    _log.warning("Do not keep this script running indefinitely! It logs a **lot** of data to rerun.")
    _log.info(f"Starting visualization loop at {viz_freq} Hz. Ctrl+C to exit.")

    try:
        while True:
            iter_start = time.perf_counter()
            time_elapsed = iter_start - start_time
            if time_elapsed >= max_time:
                _log.info(f"Max run time of {max_time}s reached. Exiting...")
                break

            rr.set_time("elapsed", duration=time_elapsed)

            # Get joint positions, camera pose, and camera frame
            frame = cam.read_camera()
            q_curr = client.get_joint_positions()
            q_curr_pt = tensor_args.to_device(q_curr)
            world_from_tcp = robot_container.kin_model.get_state(q_curr_pt).ee_pose.get_numpy_matrix()[0]
            world_from_cam = world_from_tcp @ tcp_from_camera

            # Read camera frame
            rgb = frame.rgb
            rgb_map = rgb / 255.0

            depth_m = _depth_for_visualization(frame, realsense_intrinsics)
            depth_m[depth_m > 5.0] = 0.0
            # rs_infer_depth returns depth on the RGB grid, whose intrinsics
            # are recorded in every Frame; no camera-specific property is
            # needed here.
            K = frame.intrinsics
            xyz_map = depth_to_xyz(depth_m, K)

            # Convert point cloud to world frame using camera transform
            xyz_hom = np.ones((xyz_map.shape[0], xyz_map.shape[1], 4))
            xyz_hom[:, :, :3] = xyz_map
            xyz_world = np.einsum("ij,hwj->hwi", world_from_cam, xyz_hom)[:, :, :3]
            valid_mask = (depth_m > 0) & ~np.isnan(xyz_world).any(axis=-1)
            xyz_valid, rgb_valid = xyz_world[valid_mask], rgb_map[valid_mask]

            # Log to rerun
            robot_rr.set_joint_positions(q_curr)
            rr.log(
                "world_from_cam",
                rr.Transform3D(translation=world_from_cam[:3, 3], mat3x3=world_from_cam[:3, :3], axis_length=0.05),
            )

            rr.log("rgb", rr.Image(rgb))
            rr.log("depth", rr.DepthImage(depth_m, meter=1.0))
            rr.log("pcd", rr.Points3D(positions=xyz_valid, colors=rgb_valid))

            # Sleep to maintain desired frequency
            iter_elapsed = time.perf_counter() - iter_start
            remaining_time = sleep_time - iter_elapsed
            if remaining_time > 0:
                time.sleep(remaining_time)
    except KeyboardInterrupt:
        _log.info(f"Detected keyboard interrupt. Exiting...")
    finally:
        client.close()


def viz_calibration_entrypoint():
    import tyro

    tyro.cli(viz_calibration)


if __name__ == "__main__":
    viz_calibration()
