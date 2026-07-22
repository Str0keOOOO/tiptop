"""Thread-safe ZeroMQ REQ client used by Cobot Magic server-side adapters."""

from __future__ import annotations

import threading
import uuid
from typing import Any

import zmq

from tiptop.cobot_magic.protocol import (
    DEFAULT_MAX_MESSAGE_BYTES,
    PROTOCOL_VERSION,
    ProtocolError,
    pack_message,
    unpack_message,
    validate_response,
)


class RpcRemoteError(RuntimeError):
    """An application error returned by the remote RPC server."""

    def __init__(self, code: str, message: str, retryable: bool):
        self.code = code
        self.retryable = retryable
        super().__init__(f"{code}: {message}")


class ZmqRpcClient:
    """ZeroMQ REQ/REP transport that reconnects after failures but never retries requests.

    Reconnecting only resets the local REQ socket after a timeout or malformed
    response.  It deliberately does not resend the request: a controller
    command may already have reached the robot when its response is lost.
    """

    def __init__(
        self,
        host: str,
        port: int,
        request_timeout_ms: int = 30_000,
        max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
    ):
        if not host:
            raise ValueError("host must be non-empty")
        if not 1 <= int(port) <= 65535:
            raise ValueError(f"port must be in [1, 65535], got {port}")
        if int(request_timeout_ms) <= 0:
            raise ValueError("request_timeout_ms must be positive")
        if int(max_message_bytes) <= 0:
            raise ValueError("max_message_bytes must be positive")

        self.host = str(host)
        self.port = int(port)
        self.request_timeout_ms = int(request_timeout_ms)
        self.max_message_bytes = int(max_message_bytes)
        self._context = zmq.Context.instance()
        self._lock = threading.Lock()
        self._socket: zmq.Socket | None = None
        self._connect()

    @property
    def endpoint(self) -> str:
        return f"tcp://{self.host}:{self.port}"

    def _connect(self) -> None:
        self._close_socket()
        socket = self._context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.SNDTIMEO, self.request_timeout_ms)
        socket.setsockopt(zmq.RCVTIMEO, self.request_timeout_ms)
        socket.setsockopt(zmq.MAXMSGSIZE, self.max_message_bytes)
        socket.connect(self.endpoint)
        self._socket = socket

    def _close_socket(self) -> None:
        socket, self._socket = self._socket, None
        if socket is not None:
            socket.close(linger=0)

    def close(self) -> None:
        """Close the local socket; the next request reconnects lazily."""
        with self._lock:
            self._close_socket()

    def _request(self, op: str, params: dict[str, Any], timeout_ms: int | None = None) -> Any:
        if not isinstance(op, str) or not op:
            raise ValueError("op must be a non-empty string")
        if not isinstance(params, dict):
            raise TypeError("params must be a dictionary")
        if timeout_ms is not None and int(timeout_ms) <= 0:
            raise ValueError("timeout_ms must be positive")

        request_id = str(uuid.uuid4())
        request = {
            "protocol_version": PROTOCOL_VERSION,
            "request_id": request_id,
            "op": op,
            "params": params,
        }

        with self._lock:
            if self._socket is None:
                self._connect()
            socket = self._socket
            assert socket is not None
            response_timeout_ms = self.request_timeout_ms if timeout_ms is None else int(timeout_ms)
            socket.setsockopt(zmq.RCVTIMEO, response_timeout_ms)
            try:
                socket.send(pack_message(request, max_message_bytes=self.max_message_bytes))
                response = unpack_message(socket.recv(), max_message_bytes=self.max_message_bytes)
                validate_response(response)
            except zmq.Again as exc:
                self._connect()
                raise TimeoutError(f"Cobot Magic RPC timed out: op={op}") from exc
            except (zmq.ZMQError, ProtocolError) as exc:
                self._connect()
                raise RuntimeError(f"Cobot Magic RPC transport failure: op={op}: {exc}") from exc
            finally:
                if self._socket is socket:
                    socket.setsockopt(zmq.RCVTIMEO, self.request_timeout_ms)

        if response["request_id"] != request_id:
            raise RuntimeError(
                f"Mismatched RPC request_id for {op}: expected {request_id!r}, "
                f"received {response['request_id']!r}"
            )
        if response["success"]:
            return response["result"]

        error = response["error"]
        assert isinstance(error, dict)
        raise RpcRemoteError(error["code"], error["message"], error["retryable"])

    def ping(self) -> Any:
        return self._request("ping", {})

    def health(self) -> Any:
        return self._request("health", {})
