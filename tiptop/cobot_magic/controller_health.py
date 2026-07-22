"""Health-check CLI for the controller endpoint forwarded to this GPU server."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Sequence

import numpy as np

from tiptop.cobot_magic.cobot_magic_client import CobotMagicClient
from tiptop.cobot_magic.protocol import DEFAULT_MAX_MESSAGE_BYTES
from tiptop.config import tiptop_cfg


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check a forwarded Cobot Magic controller RPC endpoint")
    parser.add_argument("--host", help="Controller tunnel host; defaults to robot.controller_host")
    parser.add_argument("--port", type=int, help="Controller tunnel port; defaults to robot.controller_port")
    parser.add_argument("--dof", type=int, help="Arm DOF; defaults to robot.dof")
    parser.add_argument("--timeout-ms", type=int, help="RPC timeout; defaults to robot.request_timeout_ms")
    parser.add_argument("--max-message-bytes", type=int, help="RPC size limit; defaults to robot.max_message_bytes")
    args = parser.parse_args(argv)

    robot_cfg = tiptop_cfg().robot

    def configured(name: str, default: Any) -> Any:
        return robot_cfg.get(name, default)

    host = args.host or configured("controller_host", configured("host", "127.0.0.1"))
    port = args.port if args.port is not None else configured("controller_port", configured("port", 15555))
    dof = args.dof if args.dof is not None else robot_cfg.dof
    timeout_ms = args.timeout_ms if args.timeout_ms is not None else configured("request_timeout_ms", 5_000)
    max_message_bytes = (
        args.max_message_bytes
        if args.max_message_bytes is not None
        else configured("max_message_bytes", DEFAULT_MAX_MESSAGE_BYTES)
    )

    client = CobotMagicClient(
        host,
        port,
        dof,
        request_timeout_ms=timeout_ms,
        max_message_bytes=max_message_bytes,
    )
    try:
        result = {"ping": client.ping(), "health": client.health()}
        print(json.dumps(result, default=_json_default, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"Controller health check failed: {exc}", file=sys.stderr)
        return 1
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
