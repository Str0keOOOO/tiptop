"""Health-check CLI for the controller endpoint forwarded to this GPU server."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Sequence

import numpy as np

from tiptop.cobot_magic.cobot_magic_client import CobotMagicClient


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check a forwarded Cobot Magic controller RPC endpoint")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=15555)
    parser.add_argument("--dof", type=int, default=6)
    parser.add_argument("--timeout-ms", type=int, default=5_000)
    args = parser.parse_args(argv)

    client = CobotMagicClient(args.host, args.port, args.dof, request_timeout_ms=args.timeout_ms)
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
