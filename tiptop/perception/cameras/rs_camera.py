import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, is_dataclass
from functools import cache
from typing import Any, Sequence

import aiohttp
import cv2
import numpy as np
from jaxtyping import Float, UInt8, UInt16

from tiptop.cobot_magic.rpc_client import ZmqRpcClient
from tiptop.config import tiptop_cfg
from tiptop.perception.cameras.frame import Frame

_log = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class RealsenseFrame(Frame):
    """Frame from RealSense which also includes the IR stereo pair."""

    ir_left: UInt8[np.ndarray, "h w"] | None = None  # IR left uint8
    ir_right: UInt8[np.ndarray, "h w"] | None = None  # IR right uint8
    depth_raw: UInt16[np.ndarray, "h w"] | None = None  # Raw depth uint16 millimeters


@dataclass(frozen=True)
class RealsenseIntrinsics:
    """Intrinsics for RealSense camera."""

    K_color: Float[np.ndarray, "3 3"]  # Color camera matrix
    K_ir: Float[np.ndarray, "3 3"]  # IR camera matrix
    baseline_ir: float  # Meters (IR baseline)
    T_color_from_ir: Float[np.ndarray, "4 4"]  # Transform from IR to color
    distortion_color: Float[np.ndarray, "5"]  # Color camera distortion coefficients


class RealsenseCamera:
    def __init__(
        self,
        serial: str | None = None,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        enable_depth: bool = False,
        enable_ir: bool = True,
    ):
        import pyrealsense2 as rs

        start_time = time.perf_counter()
        self._enable_depth = enable_depth
        self._enable_ir = enable_ir

        # Enable streams
        config = rs.config()
        if serial is not None:
            config.enable_device(serial)
            _log.info(f"Configuring RealSense camera {serial}")
        else:
            _log.info(f"Configuring RealSense camera (first available)")

        config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        if enable_depth:
            config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        if enable_ir:
            config.enable_stream(rs.stream.infrared, 1, width, height, rs.format.y8, fps)
            config.enable_stream(rs.stream.infrared, 2, width, height, rs.format.y8, fps)

        # Start pipeline
        self.pipeline = rs.pipeline()
        self._profile = self.pipeline.start(config)
        for _ in range(30):
            self.pipeline.wait_for_frames()

        # Get camera serial number
        device = self._profile.get_device()
        self.serial = device.get_info(rs.camera_info.serial_number)

        # Cache the intrinsics call
        self.get_intrinsics()

        init_dur = time.perf_counter() - start_time
        _log.info(f"Realsense camera (s/n: {self.serial}) initialization complete, took {init_dur:.2f}s")

    @cache
    def get_intrinsics(self) -> RealsenseIntrinsics:
        import pyrealsense2 as rs

        # Color intrinsics
        color_profile = self._profile.get_stream(rs.stream.color)
        color_intr = color_profile.as_video_stream_profile().get_intrinsics()
        K_color = np.array(
            [
                [color_intr.fx, 0, color_intr.ppx],
                [0, color_intr.fy, color_intr.ppy],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        distortion_color = np.array(color_intr.coeffs, dtype=np.float32)

        # IR intrinsics and extrinsics
        if not self._enable_ir:
            raise ValueError("IR streams must be enabled to get intrinsics")

        ir_left_profile = self._profile.get_stream(rs.stream.infrared, 1)
        ir_right_profile = self._profile.get_stream(rs.stream.infrared, 2)
        ir_intr = ir_left_profile.as_video_stream_profile().get_intrinsics()
        K_ir = np.array(
            [
                [ir_intr.fx, 0, ir_intr.ppx],
                [0, ir_intr.fy, ir_intr.ppy],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )

        # Baseline between IR cameras
        extr = ir_left_profile.get_extrinsics_to(ir_right_profile)
        baseline = np.linalg.norm(extr.translation)

        # Extrinsics from IR1 to color
        extr_color = ir_left_profile.get_extrinsics_to(color_profile)
        T_color_from_ir = np.eye(4, dtype=np.float32)
        T_color_from_ir[:3, :3] = np.array(extr_color.rotation).reshape(3, 3).T
        T_color_from_ir[:3, 3] = np.array(extr_color.translation)

        return RealsenseIntrinsics(
            K_color=K_color,
            K_ir=K_ir,
            baseline_ir=baseline,
            T_color_from_ir=T_color_from_ir,
            distortion_color=distortion_color,
        )

    def read_camera(self) -> RealsenseFrame:
        import pyrealsense2 as rs

        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        rgb = np.asanyarray(color_frame.get_data())
        timestamp = frames.get_timestamp()

        # IR streams required for RealsenseFrame
        if not self._enable_ir:
            raise ValueError("IR streams must be enabled for RealsenseFrame")

        ir_left_frame = frames.get_infrared_frame(1)
        ir_right_frame = frames.get_infrared_frame(2)
        ir_left = np.asanyarray(ir_left_frame.get_data())
        ir_right = np.asanyarray(ir_right_frame.get_data())

        # Optional depth
        depth_float = None
        depth_raw = None
        if self._enable_depth:
            # Get raw depth
            depth_frame = frames.get_depth_frame()
            depth_raw = np.asanyarray(depth_frame.get_data())

            # Get aligned depth and convert mm to m
            align = rs.align(rs.stream.color)
            aligned_frames = align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            depth_float = (np.asanyarray(aligned_depth_frame.get_data()) / 1000.0).astype(np.float32)

        intrinsics = self.get_intrinsics()
        return RealsenseFrame(
            serial=self.serial,
            timestamp=timestamp,
            rgb=rgb,
            intrinsics=intrinsics.K_color,
            depth=depth_float,
            ir_left=ir_left,
            ir_right=ir_right,
            depth_raw=depth_raw,
        )

    def close(self):
        """Stop the camera pipeline."""
        self.pipeline.stop()


def _depth_ir_to_color(
    depth_ir: Float[np.ndarray, "h w"],
    K_ir: Float[np.ndarray, "3 3"],
    T_color_from_ir: Float[np.ndarray, "4 4"],
    K_color: Float[np.ndarray, "3 3"],
    color_size: tuple[int, int],
) -> np.ndarray:
    """
    Warp IR depth (meters) onto color pixel grid using forward projection.
    Uses 4-neighbor splatting with z-buffer min, then fills small holes via min-filter.

    Thanks to Wenlong Huang for this.
    """
    Hc, Wc = color_size
    Hi, Wi = depth_ir.shape
    assert Hc > 0 and Wc > 0 and Hi > 0 and Wi > 0, "invalid image sizes for depth warp"

    fx_i, fy_i = float(K_ir[0, 0]), float(K_ir[1, 1])
    cx_i, cy_i = float(K_ir[0, 2]), float(K_ir[1, 2])
    fx_c, fy_c = float(K_color[0, 0]), float(K_color[1, 1])
    cx_c, cy_c = float(K_color[0, 2]), float(K_color[1, 2])

    u, v = np.meshgrid(np.arange(Wi, dtype=np.float32), np.arange(Hi, dtype=np.float32))
    z = depth_ir.astype(np.float32)
    valid = (z > 0.0) & np.isfinite(z)
    if not np.any(valid):
        return np.zeros((Hc, Wc), dtype=np.float32)

    # Unproject IR pixels to 3D
    x_i = (u[valid] - cx_i) / max(fx_i, 1e-6) * z[valid]
    y_i = (v[valid] - cy_i) / max(fy_i, 1e-6) * z[valid]
    pts_ir = np.stack([x_i, y_i, z[valid]], axis=0)

    # Transform to color frame
    R = T_color_from_ir[:3, :3].astype(np.float32)
    t = T_color_from_ir[:3, 3].astype(np.float32).reshape(3, 1)
    pts_c = R @ pts_ir + t
    Xc, Yc, Zc = pts_c[0], pts_c[1], pts_c[2]
    valid_c = Zc > 1e-6
    if not np.any(valid_c):
        return np.zeros((Hc, Wc), dtype=np.float32)
    Xc, Yc, Zc = Xc[valid_c], Yc[valid_c], Zc[valid_c]

    # Project to color image
    uc_f = fx_c * (Xc / Zc) + cx_c
    vc_f = fy_c * (Yc / Zc) + cy_c
    x0 = np.floor(uc_f).astype(np.int32)
    y0 = np.floor(vc_f).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    depth_color = np.full((Hc, Wc), np.inf, dtype=np.float32)

    def splat(ix: np.ndarray, iy: np.ndarray, zvals: np.ndarray) -> None:
        inb = (ix >= 0) & (ix < Wc) & (iy >= 0) & (iy < Hc)
        if not np.any(inb):
            return
        np.minimum.at(depth_color, (iy[inb], ix[inb]), zvals[inb])

    # Splat to 4 neighbors to reduce gaps
    splat(x0, y0, Zc)
    splat(x1, y0, Zc)
    splat(x0, y1, Zc)
    splat(x1, y1, Zc)

    # Fill holes with iterative erosion (min-filter)
    # This handles larger holes from FoundationStereo by propagating valid depth values
    holes = np.isinf(depth_color)
    if np.any(holes):
        depth_color[holes] = 0.0
        kernel = np.ones((3, 3), np.uint8)
        max_iterations = 5  # Fill holes up to ~5 pixels wide
        for _ in range(max_iterations):
            holes_mask = depth_color <= 0.0
            if not np.any(holes_mask):
                break
            # Use large sentinel value for unfilled regions, erode to get min of neighbors
            sentinel = np.where(depth_color > 0.0, depth_color, 65535.0).astype(np.float32)
            min_neigh = cv2.erode(sentinel, kernel)

            # Only fill pixels that have at least one valid neighbor (not all sentinels)
            newly_filled = holes_mask & (min_neigh < 65000.0)
            depth_color[newly_filled] = min_neigh[newly_filled]

        # Clean up any remaining unfilled holes
        depth_color[depth_color > 65000.0] = 0.0

    return depth_color


def _prepare_ir_stereo(
    frame: RealsenseFrame,
) -> tuple[UInt8[np.ndarray, "h w 3"], UInt8[np.ndarray, "h w 3"], tuple[int, int]]:
    """Prepare IR stereo images for FoundationStereo inference."""
    rgb_size = frame.rgb.shape[:2]
    ir_size = frame.ir_left.shape[:2]
    if rgb_size != ir_size:
        raise NotImplementedError("We don't currently support different color and IR resolutions")

    # Convert IR to RGB (FoundationStereo expects 3-channel input)
    ir_left, ir_right = frame.ir_left, frame.ir_right
    ir_left_rgb = np.stack([ir_left, ir_left, ir_left], axis=-1)
    ir_right_rgb = np.stack([ir_right, ir_right, ir_right], axis=-1)

    return ir_left_rgb, ir_right_rgb, rgb_size


def rs_infer_depth(
    frame: RealsenseFrame,
    intrinsics: RealsenseIntrinsics,
) -> Float[np.ndarray, "h w"]:
    """Estimate depth from Realsense frame and intrinsics using FoundationStereo. Synchronous version."""
    from tiptop.perception.foundation_stereo import infer_depth

    ir_left_rgb, ir_right_rgb, rgb_size = _prepare_ir_stereo(frame)
    cfg = tiptop_cfg()
    K_ir = intrinsics.K_ir
    depth = infer_depth(
        cfg.perception.foundation_stereo.url,
        ir_left_rgb,
        ir_right_rgb,
        fx=K_ir[0, 0],
        fy=K_ir[1, 1],
        cx=K_ir[0, 2],
        cy=K_ir[1, 2],
        baseline=intrinsics.baseline_ir,
    )
    depth_aligned = _depth_ir_to_color(depth, K_ir, intrinsics.T_color_from_ir, intrinsics.K_color, color_size=rgb_size)
    return depth_aligned


async def rs_infer_depth_async(
    session: aiohttp.ClientSession,
    frame: RealsenseFrame,
    intrinsics: RealsenseIntrinsics,
) -> Float[np.ndarray, "h w"]:
    """Estimate depth from Realsense frame and intrinsics using FoundationStereo. Async version."""
    from tiptop.perception.foundation_stereo import infer_depth_async

    ir_left_rgb, ir_right_rgb, rgb_size = _prepare_ir_stereo(frame)
    cfg = tiptop_cfg()
    K_ir = intrinsics.K_ir
    depth = await infer_depth_async(
        session,
        cfg.perception.foundation_stereo.url,
        ir_left_rgb,
        ir_right_rgb,
        fx=K_ir[0, 0],
        fy=K_ir[1, 1],
        cx=K_ir[0, 2],
        cy=K_ir[1, 2],
        baseline=intrinsics.baseline_ir,
    )
    depth_aligned = _depth_ir_to_color(depth, K_ir, intrinsics.T_color_from_ir, intrinsics.K_color, color_size=rgb_size)
    return depth_aligned


def _required_remote_matrix(result: dict[str, Any], field: str, shape: tuple[int, ...]) -> np.ndarray:
    if field not in result:
        raise RuntimeError(f"Remote camera response is missing {field}")
    array = np.asarray(result[field])
    if array.dtype != np.float32:
        raise RuntimeError(f"Remote camera {field} has dtype {array.dtype}, expected float32")
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise RuntimeError(f"Remote camera {field} must be a finite matrix with shape {shape}")
    return np.ascontiguousarray(array)


def _required_remote_ir(result: dict[str, Any], field: str, rgb_shape: tuple[int, int]) -> np.ndarray:
    if field not in result:
        raise RuntimeError(f"Remote camera response is missing {field}")
    array = np.asarray(result[field])
    if array.dtype != np.uint8:
        raise RuntimeError(f"Remote camera {field} has dtype {array.dtype}, expected uint8")
    if array.shape != rgb_shape:
        raise RuntimeError(f"Remote camera {field} has shape {array.shape}, expected {rgb_shape}")
    return np.ascontiguousarray(array)


def _required_remote_depth(
    result: dict[str, Any], field: str, image_shape: tuple[int, int], dtype: np.dtype[Any]
) -> np.ndarray:
    if field not in result:
        raise RuntimeError(f"Remote camera response is missing {field}")
    array = np.asarray(result[field])
    if array.dtype != dtype:
        raise RuntimeError(f"Remote camera {field} has dtype {array.dtype}, expected {dtype}")
    if array.shape != image_shape or not np.all(np.isfinite(array)):
        raise RuntimeError(f"Remote camera {field} must be finite with shape {image_shape}")
    return np.ascontiguousarray(array)


class RemoteRealsenseCamera(ZmqRpcClient):
    """RealSense camera client that receives the same data contract over RPC."""

    def __init__(
        self,
        serial: str,
        host: str,
        port: int,
        enable_depth: bool = False,
        request_timeout_ms: int = 30_000,
    ):
        if not serial:
            raise ValueError("serial must be non-empty")
        self.serial = str(serial)
        self._enable_depth = enable_depth
        super().__init__(
            host=host,
            port=port,
            request_timeout_ms=request_timeout_ms,
            max_message_bytes=128 * 1024 * 1024,
        )
        _log.info("Configured remote RealSense %s through %s for FoundationStereo", self.serial, self.endpoint)

    @cache
    def get_intrinsics(self) -> RealsenseIntrinsics:
        result = self._request("get_intrinsics", {"serial": self.serial})
        if not isinstance(result, dict):
            raise RuntimeError("Invalid get_intrinsics response")
        if str(result.get("serial")) != self.serial:
            raise RuntimeError(f"Remote camera returned intrinsics for {result.get('serial')!r}, expected {self.serial!r}")

        K_color = _required_remote_matrix(result, "K_color", (3, 3))
        distortion_color = _required_remote_matrix(result, "distortion_color", (5,))
        K_ir = _required_remote_matrix(result, "K_ir", (3, 3))
        T_color_from_ir = _required_remote_matrix(result, "T_color_from_ir", (4, 4))
        if "baseline_ir" not in result:
            raise RuntimeError("Remote camera response is missing baseline_ir")
        baseline_ir = float(result["baseline_ir"])
        if not np.isfinite(baseline_ir) or baseline_ir <= 0.0:
            raise RuntimeError("Remote camera baseline_ir must be finite and positive")
        return RealsenseIntrinsics(
            K_color=K_color,
            K_ir=K_ir,
            baseline_ir=baseline_ir,
            T_color_from_ir=T_color_from_ir,
            distortion_color=distortion_color,
        )

    def list_cameras(self) -> list[dict[str, Any]]:
        result = self._request("list_cameras", {})
        if not isinstance(result, dict) or not isinstance(result.get("cameras"), list):
            raise RuntimeError("Invalid list_cameras response")
        return result["cameras"]

    def read_camera(self) -> RealsenseFrame:
        """Read an RPC frame with the same data contract as RealsenseCamera.read_camera()."""
        result = self._request("read_camera", {"serial": self.serial})
        if not isinstance(result, dict):
            raise RuntimeError("Invalid read_camera response")
        if str(result.get("serial")) != self.serial:
            raise RuntimeError(f"Remote camera returned frame for {result.get('serial')!r}, expected {self.serial!r}")

        rgb = np.asarray(result.get("rgb"))
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise RuntimeError(f"Remote camera rgb must be uint8 [H, W, 3], got {rgb.dtype} {rgb.shape}")
        rgb = np.ascontiguousarray(rgb)
        image_shape = rgb.shape[:2]

        timestamp = float(result.get("timestamp"))
        if not np.isfinite(timestamp):
            raise RuntimeError("Remote camera timestamp must be finite")

        ir_left = _required_remote_ir(result, "ir_left", image_shape)
        ir_right = _required_remote_ir(result, "ir_right", image_shape)
        depth = None
        depth_raw = None
        if self._enable_depth:
            depth = _required_remote_depth(result, "depth", image_shape, np.dtype(np.float32))
            depth_raw = _required_remote_depth(result, "depth_raw", image_shape, np.dtype(np.uint16))

        intrinsics = self.get_intrinsics()
        return RealsenseFrame(
            serial=self.serial,
            timestamp=timestamp,
            rgb=rgb,
            intrinsics=intrinsics.K_color,
            depth=depth,
            ir_left=ir_left,
            ir_right=ir_right,
            depth_raw=depth_raw,
        )


def _remote_camera_json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def remote_realsense_health_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check a forwarded Cobot Magic RealSense RPC endpoint")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=15556)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--timeout-ms", type=int, default=5_000)
    args = parser.parse_args(argv)

    camera = RemoteRealsenseCamera(
        serial=args.serial,
        host=args.host,
        port=args.port,
        request_timeout_ms=args.timeout_ms,
    )
    try:
        frame = camera.read_camera()
        result = {
            "ping": camera.ping(),
            "health": camera.health(),
            "cameras": camera.list_cameras(),
            "intrinsics": camera.get_intrinsics(),
            "frame": {
                "rgb_shape": frame.rgb.shape,
                "ir_left_shape": frame.ir_left.shape,
                "ir_right_shape": frame.ir_right.shape,
                "timestamp": frame.timestamp,
            },
        }
        print(json.dumps(result, default=_remote_camera_json_default, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"Camera health check failed: {exc}", file=sys.stderr)
        return 1
    finally:
        camera.close()
