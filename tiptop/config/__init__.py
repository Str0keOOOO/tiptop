import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float
from omegaconf import DictConfig, OmegaConf
from scipy.spatial.transform import Rotation

config_dir = Path(__file__).parent
config_assets_dir = config_dir / "assets"
calib_info_path = config_assets_dir / "calibration_info.json"

_cached_cfg: DictConfig | None = None
_cached_cfg_path: Path | None = None

CALIBRATION_CONVENTION = "parent_from_child"


@dataclass(frozen=True)
class CameraCalibration:
    """A rigid camera transform with explicit frame semantics.

    ``parent_from_child`` maps camera-coordinate points into the parent frame.
    Legacy calibration entries can still be read only when no frame assertion
    is requested; all Cobot Magic live paths request and enforce the metadata.
    """

    matrix: Float[np.ndarray, "4 4"]
    parent_frame: str | None
    child_frame: str | None
    convention: str | None


def set_tiptop_cfg_from_file(cfg_path: Path) -> DictConfig:
    """Load and cache the TiPToP config from a specific file. Call before any tiptop_cfg() usage."""
    global _cached_cfg, _cached_cfg_path
    cfg = OmegaConf.load(cfg_path)
    _cached_cfg = cfg
    _cached_cfg_path = Path(cfg_path)
    return cfg


def tiptop_cfg() -> DictConfig:
    """Return the cached TiPToP config, loading the default config file on first call."""
    if _cached_cfg is None:
        return set_tiptop_cfg_from_file(config_dir / "tiptop.yml")
    return _cached_cfg


def get_tiptop_cfg_path() -> Path:
    """Return the source path of the currently-cached config. Loads the default config if not yet cached."""
    if _cached_cfg_path is None:
        tiptop_cfg()
    assert _cached_cfg_path is not None
    return _cached_cfg_path


def load_calibration_info():
    if not os.path.exists(calib_info_path):
        raise FileNotFoundError(f"{calib_info_path} not found.")
    with open(calib_info_path, "r") as f:
        calibration_info = json.load(f)
    return calibration_info


def load_calibration_record(cam_key: str) -> CameraCalibration:
    """Load a calibration record without accepting or inferring its frames."""
    calibration_dict = load_calibration_info()
    if cam_key not in calibration_dict:
        raise ValueError(f"{cam_key} not found in {calib_info_path}")

    entry = calibration_dict[cam_key]
    pose_vec = entry["pose"]
    if len(pose_vec) != 6:
        raise ValueError(f"Calibration pose for {cam_key} must have six values, got {len(pose_vec)}")
    xyz, rpy = pose_vec[:3], pose_vec[3:]
    cam2frame = np.eye(4)
    cam2frame[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    cam2frame[:3, 3] = xyz
    if not np.all(np.isfinite(cam2frame)):
        raise ValueError(f"Calibration matrix for {cam_key} contains non-finite values")
    return CameraCalibration(
        matrix=cam2frame,
        parent_frame=entry.get("parent_frame"),
        child_frame=entry.get("child_frame"),
        convention=entry.get("convention"),
    )


def load_calibration(
    cam_key: str,
    *,
    parent_frame: str | None = None,
    child_frame: str | None = None,
) -> Float[np.ndarray, "4 4"]:
    """Load ``parent_from_child`` calibration, optionally enforcing both frames.

    A caller that provides either frame must provide both.  This deliberately
    rejects bare legacy entries rather than guessing which robot link was used
    during calibration.
    """
    if (parent_frame is None) != (child_frame is None):
        raise ValueError("parent_frame and child_frame must be provided together")
    record = load_calibration_record(cam_key)
    if parent_frame is not None:
        if record.convention != CALIBRATION_CONVENTION:
            raise ValueError(
                f"Calibration {cam_key} convention must be {CALIBRATION_CONVENTION!r}, "
                f"got {record.convention!r}; re-calibrate or migrate it explicitly"
            )
        if record.parent_frame != parent_frame or record.child_frame != child_frame:
            raise ValueError(
                f"Calibration {cam_key} frames are {record.parent_frame!r}_from_{record.child_frame!r}, "
                f"expected {parent_frame!r}_from_{child_frame!r}; re-calibrate or migrate it explicitly"
            )
    return record.matrix


def load_tcp_from_camera(cam_key: str, robot_type: str | None = None) -> Float[np.ndarray, "4 4"]:
    """Load the wrist-camera extrinsics in the planning TCP frame.

    Cobot Magic has an explicit TCP and therefore rejects unlabelled
    calibration data.  Legacy robot paths retain their existing calibration
    loading behaviour until they are migrated separately.
    """
    if robot_type is None:
        robot_type = str(tiptop_cfg().robot.type)
    if robot_type == "cobot_magic":
        from tiptop.cobot_magic.frames import load_cobot_magic_tcp_from_camera

        return load_cobot_magic_tcp_from_camera(cam_key)
    return load_calibration(cam_key)


def update_calibration_info(
    cam_key: str,
    pose: np.ndarray,
    *,
    parent_frame: str | None = None,
    child_frame: str | None = None,
    convention: str = CALIBRATION_CONVENTION,
):
    """Update calibration info with new camera pose.

    Args:
        cam_key: Camera identifier (e.g., "16779706_left")
        pose: 6DOF pose vector [x, y, z, roll, pitch, yaw]
    """
    import time

    # Load existing calibration info or create empty dict
    if os.path.exists(calib_info_path):
        calibration_dict = load_calibration_info()
    else:
        calibration_dict = {}

    if (parent_frame is None) != (child_frame is None):
        raise ValueError("parent_frame and child_frame must be provided together")
    entry = {
        "pose": pose.tolist() if isinstance(pose, np.ndarray) else list(pose),
        "timestamp": time.time(),
    }
    if parent_frame is not None:
        if convention != CALIBRATION_CONVENTION:
            raise ValueError(f"Unsupported calibration convention: {convention!r}")
        entry.update(
            parent_frame=parent_frame,
            child_frame=child_frame,
            convention=convention,
        )
    # Update with new pose and timestamp
    calibration_dict[cam_key] = entry

    # Write back to file
    with open(calib_info_path, "w") as f:
        json.dump(calibration_dict, f, indent=2)

    print(f"Updated calibration for {cam_key}")
