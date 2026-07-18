"""Cobot Magic GPU-server integration."""

from __future__ import annotations

from typing import Any

__all__ = ["CobotMagicClient", "get_cobot_magic_client"]


def __getattr__(name: str) -> Any:
    """Avoid importing the optional ZeroMQ dependency for unrelated robot types."""
    if name in __all__:
        from tiptop.cobot_magic.cobot_magic_client import CobotMagicClient, get_cobot_magic_client

        return {"CobotMagicClient": CobotMagicClient, "get_cobot_magic_client": get_cobot_magic_client}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
