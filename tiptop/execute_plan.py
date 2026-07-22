import logging
import time
from collections.abc import Mapping

from cutamp.robots.utils import RerunRobot
from tiptop.cobot_magic.cobot_magic_client import CobotMagicClient
from tiptop.config import tiptop_cfg
from tiptop.utils import RobotClient, get_robot_client

_log = logging.getLogger(__name__)


class ExecutionFailure(Exception):
    """Failure in executing plan on robot."""


def execute_cutamp_plan(
    cutamp_plan: list[dict],
    client: RobotClient | None = None,
    *,
    rerun_robot: RerunRobot | None = None,
    rerun_gripper_joint_name: str | None = None,
    rerun_gripper_positions: Mapping[str, float] | None = None,
) -> None:
    """Execute the plan from cuTAMP on the real robot.

    ``rerun_*`` is optional and updates a visual-only gripper joint after a
    successful independent gripper RPC.  Trajectory waypoints are deliberately
    left untouched, so the physical arm RPC receives exactly cuTAMP's arm DOF.
    """
    if client is None:
        client = get_robot_client()

    rerun_args = (rerun_robot, rerun_gripper_joint_name, rerun_gripper_positions)
    if any(arg is not None for arg in rerun_args) and not all(arg is not None for arg in rerun_args):
        raise ValueError("rerun_robot, rerun_gripper_joint_name, and rerun_gripper_positions must be set together")

    start_time = time.perf_counter()
    for step, action_dict in enumerate(cutamp_plan):
        action_start_time = time.perf_counter()
        action_type = action_dict["type"]
        action_label = action_dict["label"]

        # Form log message
        msg = f"Executing step {step + 1}/{len(cutamp_plan)}: {action_label}. Action type: {action_dict['type']}"
        if action_type == "gripper":
            msg += f" ({action_dict['action']})"
        elif action_type == "trajectory":
            msg += f" ({len(action_dict['plan'].position)} waypoints)"
        else:
            raise ValueError(f"Unknown action type in cuTAMP plan: {action_dict['type']}")
        _log.info(msg)

        # Now execute the actions
        if action_type == "gripper":
            action = action_dict["action"]
            if action == "open":
                result = client.open_gripper(speed=1.0)
            elif action == "close":
                result = client.close_gripper(speed=1.0)
            else:
                raise ValueError(f"Unknown gripper action: {action}")

        elif action_type == "trajectory":
            # Extract joint position and velocity waypoints for the trajectory
            waypoints = action_dict["plan"].position.cpu().numpy()
            velocities = action_dict["plan"].velocity.cpu().numpy()
            expected_dof = int(tiptop_cfg().robot.dof)
            if waypoints.ndim != 2:
                raise ValueError(f"Arm trajectory positions must be 2D (N, {expected_dof}), got {waypoints.shape}")
            if waypoints.shape[1] != expected_dof:
                raise ValueError(
                    f"Arm trajectory must contain exactly {expected_dof} joints, got shape {waypoints.shape}; "
                    "Rerun-only gripper joints must never reach the arm RPC"
                )
            if velocities.shape != waypoints.shape:
                raise ValueError(
                    f"Arm trajectory velocities must match positions, got {velocities.shape} vs {waypoints.shape}"
                )
            timings = [action_dict["dt"]] * len(waypoints)
            result = client.execute_joint_impedance_path(
                joint_confs=waypoints, joint_vels=velocities, durations=timings
            )

        else:
            raise ValueError(f"Unexpected action type in cuTAMP plan: {action_dict['type']}")

        # Raise error if execution failed
        if result is None:
            raise RuntimeError("Fatal error: result should not be None")
        if not result["success"]:
            error = str(result.get("error", "robot command failed"))
            # A failed Cobot Magic motion may have reached the bridge even if
            # its response was lost. Ask its explicit stop operation once;
            # ZmqRpcClient never replays either command.
            if isinstance(client, CobotMagicClient):
                stop_result = client.stop()
                if not stop_result["success"]:
                    error = f"{error}; subsequent Cobot Magic stop failed: {stop_result.get('error')}"
            raise ExecutionFailure(error)

        if action_type == "gripper" and rerun_robot is not None:
            try:
                gripper_position = rerun_gripper_positions[action]
            except KeyError as exc:
                raise ValueError(f"No Rerun joint position configured for gripper action: {action}") from exc
            rerun_robot.set_joint_position(rerun_gripper_joint_name, gripper_position)

        action_duration = time.perf_counter() - action_start_time
        _log.debug(f"Executing {action_type} action took {action_duration:.2f}s")

    # Now we're done executing plan open-loop without any failures on the controller side
    duration = time.perf_counter() - start_time
    _log.info(f"Real robot execution took {duration:.2f}s")
