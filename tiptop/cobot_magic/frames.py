"""Cobot Magic frame names and strict wrist-camera calibration loading."""

from __future__ import annotations

import numpy as np
from jaxtyping import Float

from cutamp.robots.cobot_magic_frames import (
    COBOT_MAGIC_BASE_LINK,
    COBOT_MAGIC_GRIPPER_BASE_LINK,
    COBOT_MAGIC_TCP_LINK,
    gripper_base_from_tcp,
    tcp_from_gripper_base,
)
from tiptop.config import CALIBRATION_CONVENTION, load_calibration


# FoundationStereo depth is reprojected to the RGB grid in rs_camera.py using
# K_color and T_color_from_ir.  Consequently depth_to_xyz() produces points in
# this color optical frame, not in the left-IR optical frame.
COBOT_MAGIC_CAMERA_FRAME = "camera_color_optical_frame"


def cobot_magic_calibration_metadata() -> dict[str, str]:
    """Metadata that every newly calibrated Cobot wrist camera must carry."""
    return {
        "parent_frame": COBOT_MAGIC_TCP_LINK,
        "child_frame": COBOT_MAGIC_CAMERA_FRAME,
        "convention": CALIBRATION_CONVENTION,
    }


def load_cobot_magic_tcp_from_camera(serial: str) -> Float[np.ndarray, "4 4"]:
    """Load strictly validated ``tool_center_point_from_camera`` extrinsics."""
    return load_calibration(
        serial,
        parent_frame=COBOT_MAGIC_TCP_LINK,
        child_frame=COBOT_MAGIC_CAMERA_FRAME,
    )


__all__ = [
    "COBOT_MAGIC_BASE_LINK",
    "COBOT_MAGIC_GRIPPER_BASE_LINK",
    "COBOT_MAGIC_TCP_LINK",
    "COBOT_MAGIC_CAMERA_FRAME",
    "cobot_magic_calibration_metadata",
    "gripper_base_from_tcp",
    "tcp_from_gripper_base",
    "load_cobot_magic_tcp_from_camera",
]
