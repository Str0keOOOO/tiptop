"""GPU-server client for a Cobot Magic controller exposed over ZeroMQ RPC."""

from __future__ import annotations

from functools import cache
from typing import Any, Sequence

import numpy as np

from tiptop.cobot_magic.rpc_client import ZmqRpcClient


def _normalized_unit_interval(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite value in [0, 1], got {value}")
    return value


class CobotMagicClient(ZmqRpcClient):
    """RobotClient-compatible RPC client with no ROS dependency."""

    def __init__(
        self,
        host: str,
        port: int,
        dof: int,
        request_timeout_ms: int = 30_000,
        trajectory_timeout_ms: int = 300_000,
    ):
        if int(dof) <= 0:
            raise ValueError(f"dof must be positive, got {dof}")
        if int(trajectory_timeout_ms) <= 0:
            raise ValueError("trajectory_timeout_ms must be positive")
        self.dof = int(dof)
        self.trajectory_timeout_ms = int(trajectory_timeout_ms)
        super().__init__(
            host=host,
            port=port,
            request_timeout_ms=request_timeout_ms,
            max_message_bytes=64 * 1024 * 1024,
        )

    def get_joint_positions(self) -> list[float]:
        result = self._request("get_joint_positions", {})
        if not isinstance(result, dict):
            raise RuntimeError("Invalid get_joint_positions response")
        q = np.asarray(result.get("joint_positions"), dtype=np.float64)
        if q.shape != (self.dof,):
            raise RuntimeError(f"Expected {self.dof} joint positions, got shape {q.shape}")
        if not np.all(np.isfinite(q)):
            raise RuntimeError("Received non-finite joint positions")
        return q.tolist()

    def _gripper_command(self, op: str, speed: float, force: float) -> dict[str, Any]:
        try:
            result = self._request(
                op,
                {
                    "speed": _normalized_unit_interval(speed, "speed"),
                    "force": _normalized_unit_interval(force, "force"),
                },
            )
            if not isinstance(result, dict) or result.get("success") is not True:
                raise RuntimeError("Remote gripper command did not report success")
            return {"success": True}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def open_gripper(self, speed: float = 1.0, force: float = 0.1) -> dict[str, Any]:
        return self._gripper_command("open_gripper", speed=speed, force=force)

    def close_gripper(self, speed: float = 1.0, force: float = 0.1) -> dict[str, Any]:
        return self._gripper_command("close_gripper", speed=speed, force=force)

    def execute_joint_impedance_path(
        self,
        joint_confs: Sequence[Sequence[float]] | np.ndarray,
        joint_vels: Sequence[Sequence[float]] | np.ndarray,
        durations: Sequence[float] | np.ndarray,
    ) -> dict[str, Any]:
        """Send one complete trajectory; the control-rate loop stays on the upper computer."""
        try:
            joint_confs_array = np.ascontiguousarray(joint_confs, dtype=np.float64)
            joint_vels_array = np.ascontiguousarray(joint_vels, dtype=np.float64)
            durations_array = np.ascontiguousarray(durations, dtype=np.float64)

            if joint_confs_array.shape == (0,):
                return {"success": True}
            if joint_confs_array.ndim != 2 or joint_confs_array.shape[1] != self.dof:
                raise ValueError(
                    f"Expected joint_confs shape [N, {self.dof}], got {joint_confs_array.shape}"
                )
            if joint_vels_array.shape != joint_confs_array.shape:
                raise ValueError(
                    f"Velocity shape {joint_vels_array.shape} does not match position shape {joint_confs_array.shape}"
                )
            if durations_array.shape != (len(joint_confs_array),):
                raise ValueError(
                    f"Expected {len(joint_confs_array)} durations, got shape {durations_array.shape}"
                )
            if not np.all(np.isfinite(joint_confs_array)):
                raise ValueError("Trajectory contains non-finite joint positions")
            if not np.all(np.isfinite(joint_vels_array)):
                raise ValueError("Trajectory contains non-finite joint velocities")
            if not np.all(np.isfinite(durations_array)) or np.any(durations_array <= 0.0):
                raise ValueError("Trajectory durations must be finite and positive")
            result = self._request(
                "execute_joint_impedance_path",
                {
                    "joint_confs": joint_confs_array,
                    "joint_vels": joint_vels_array,
                    "durations": durations_array,
                },
                timeout_ms=self.trajectory_timeout_ms,
            )
            if not isinstance(result, dict) or result.get("success") is not True:
                error = result.get("error", "Remote trajectory execution failed") if isinstance(result, dict) else "Invalid response"
                return {"success": False, "error": str(error)}
            return {"success": True}
        except Exception as exc:
            return {"success": False, "error": str(exc)}


@cache
def get_cobot_magic_client() -> CobotMagicClient:
    """Build the RPC client from the active GPU-server TiPToP configuration."""
    from tiptop.config import tiptop_cfg

    robot_cfg = tiptop_cfg().robot

    def get_cfg(name: str, default: Any) -> Any:
        return robot_cfg.get(name, default) if hasattr(robot_cfg, "get") else getattr(robot_cfg, name, default)

    return CobotMagicClient(
        host=str(robot_cfg.host),
        port=int(robot_cfg.port),
        dof=int(robot_cfg.dof),
        request_timeout_ms=int(get_cfg("request_timeout_ms", 30_000)),
        trajectory_timeout_ms=int(get_cfg("trajectory_timeout_ms", 300_000)),
    )
