"""Versioned MessagePack protocol shared by the Cobot Magic RPC endpoints."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import msgpack
import msgpack_numpy

msgpack_numpy.patch()

PROTOCOL_VERSION = "1.0"
DEFAULT_MAX_MESSAGE_BYTES = 128 * 1024 * 1024


class ProtocolError(ValueError):
    """Raised when an RPC message does not satisfy the protocol schema."""


def _require_nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProtocolError(f"{field} must be a non-empty string")
    return value


def _require_message_size(payload: bytes, max_message_bytes: int) -> None:
    if max_message_bytes <= 0:
        raise ValueError("max_message_bytes must be positive")
    if len(payload) > max_message_bytes:
        raise ProtocolError(
            f"RPC message is {len(payload)} bytes, exceeding the {max_message_bytes}-byte limit"
        )


def pack_message(message: Mapping[str, Any], max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES) -> bytes:
    """Encode an RPC message while retaining NumPy array dtype and shape."""
    if not isinstance(message, Mapping):
        raise ProtocolError("RPC message must be a dictionary")
    try:
        payload = msgpack.packb(dict(message), use_bin_type=True, default=msgpack_numpy.encode)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"Unable to encode RPC message: {exc}") from exc
    _require_message_size(payload, max_message_bytes)
    return payload


def unpack_message(payload: bytes, max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES) -> dict[str, Any]:
    """Decode an RPC message and reject oversize or non-mapping payloads."""
    if not isinstance(payload, bytes):
        raise ProtocolError("RPC payload must be bytes")
    _require_message_size(payload, max_message_bytes)
    try:
        value = msgpack.unpackb(payload, raw=False, object_hook=msgpack_numpy.decode)
    except (msgpack.ExtraData, msgpack.FormatError, msgpack.StackError, ValueError, TypeError) as exc:
        raise ProtocolError(f"Unable to decode RPC message: {exc}") from exc
    if not isinstance(value, dict):
        raise ProtocolError("RPC message must decode to a dictionary")
    return value


def validate_request(request: Mapping[str, Any]) -> None:
    """Validate the fields common to every client request."""
    if not isinstance(request, Mapping):
        raise ProtocolError("RPC request must be a dictionary")
    if request.get("protocol_version") != PROTOCOL_VERSION:
        raise ProtocolError(
            f"Unsupported RPC protocol version: {request.get('protocol_version')!r}; "
            f"expected {PROTOCOL_VERSION!r}"
        )
    _require_nonempty_string(request.get("request_id"), "request_id")
    _require_nonempty_string(request.get("op"), "op")
    if not isinstance(request.get("params"), dict):
        raise ProtocolError("params must be a dictionary")


def validate_response(response: Mapping[str, Any]) -> None:
    """Validate the fields common to every server response."""
    if not isinstance(response, Mapping):
        raise ProtocolError("RPC response must be a dictionary")
    if response.get("protocol_version") != PROTOCOL_VERSION:
        raise ProtocolError(
            f"RPC protocol version mismatch: {response.get('protocol_version')!r}; "
            f"expected {PROTOCOL_VERSION!r}"
        )
    _require_nonempty_string(response.get("request_id"), "request_id")
    success = response.get("success")
    if not isinstance(success, bool):
        raise ProtocolError("success must be a boolean")
    if success:
        if response.get("error") is not None:
            raise ProtocolError("successful RPC response must have error=None")
        if "result" not in response:
            raise ProtocolError("successful RPC response is missing result")
        return

    error = response.get("error")
    if not isinstance(error, Mapping):
        raise ProtocolError("failed RPC response must include an error dictionary")
    _require_nonempty_string(error.get("code"), "error.code")
    _require_nonempty_string(error.get("message"), "error.message")
    if not isinstance(error.get("retryable"), bool):
        raise ProtocolError("error.retryable must be a boolean")


def make_success(request_id: str, result: Any) -> dict[str, Any]:
    """Build a successful RPC response with the protocol envelope."""
    _require_nonempty_string(request_id, "request_id")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": request_id,
        "success": True,
        "result": result,
        "error": None,
    }


def make_error(
    request_id: str,
    code: str,
    message: str,
    retryable: bool = False,
) -> dict[str, Any]:
    """Build a failed RPC response with a machine-readable error code."""
    _require_nonempty_string(request_id, "request_id")
    _require_nonempty_string(code, "code")
    _require_nonempty_string(message, "message")
    if not isinstance(retryable, bool):
        raise ProtocolError("retryable must be a boolean")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": request_id,
        "success": False,
        "result": None,
        "error": {
            "code": code,
            "message": message,
            "retryable": retryable,
        },
    }
