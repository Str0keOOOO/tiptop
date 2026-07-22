"""Strict multipart client for the OmniGround grounding service."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import math
import time
from functools import cache
from pathlib import Path
from typing import Any

import aiohttp
from PIL import Image

from tiptop.config import tiptop_cfg

_log = logging.getLogger(__name__)
_PROMPTS_DIR = Path(__file__).parent / "prompts"
_GENERATE_ENDPOINTS = {"/generate", "/v1/generate"}


class OmniGroundError(RuntimeError):
    """A malformed, failed, or contract-incompatible OmniGround request."""


class OmniGroundHttpError(OmniGroundError):
    """An OmniGround HTTP error with its structured server error when available."""

    def __init__(self, status: int, code: str | None, message: str):
        self.status = status
        self.code = code
        detail = f"{code}: {message}" if code else message
        super().__init__(f"OmniGround request failed with HTTP {status}: {detail}")


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise OmniGroundError(f"OmniGround {field} must be a non-empty, trimmed string")
    return value


def _finite_nonnegative_number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise OmniGroundError(f"OmniGround {field} must be a number greater than or equal to zero")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise OmniGroundError(f"OmniGround {field} must be a number greater than or equal to zero") from exc
    if not math.isfinite(number) or number < 0.0:
        raise OmniGroundError(f"OmniGround {field} must be a finite number greater than or equal to zero")
    return number


def _validate_box(box: Any, index: int) -> list[int]:
    if not isinstance(box, list) or len(box) != 4:
        raise OmniGroundError(f"OmniGround bboxes[{index}].box_2d must be [ymin, xmin, ymax, xmax]")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in box):
        raise OmniGroundError(f"OmniGround bboxes[{index}].box_2d must contain only integers")
    ymin, xmin, ymax, xmax = box
    if not all(0 <= value <= 1000 for value in box):
        raise OmniGroundError(f"OmniGround bboxes[{index}].box_2d values must be normalized to 0..1000")
    if ymin >= ymax or xmin >= xmax:
        raise OmniGroundError(f"OmniGround bboxes[{index}].box_2d must satisfy ymin < ymax and xmin < xmax")
    return [ymin, xmin, ymax, xmax]


def parse_grounding_result(payload: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Strictly parse OmniGround's direct ``bboxes``/``predicates`` response.

    No legacy result envelope, text field, or Markdown code fence is accepted.
    The second return value is TiPToP's internal predicate representation.
    """

    if not isinstance(payload, dict):
        raise OmniGroundError("OmniGround success response must be a JSON object")
    expected_fields = {"bboxes", "predicates"}
    actual_fields = set(payload)
    if actual_fields != expected_fields:
        raise OmniGroundError(
            "OmniGround success response must contain exactly bboxes and predicates; "
            f"received {sorted(actual_fields)}"
        )
    bboxes_value = payload["bboxes"]
    predicates_value = payload["predicates"]
    if not isinstance(bboxes_value, list) or not isinstance(predicates_value, list):
        raise OmniGroundError("OmniGround bboxes and predicates must both be lists")

    bboxes: list[dict[str, Any]] = []
    labels: set[str] = set()
    for index, bbox in enumerate(bboxes_value):
        if not isinstance(bbox, dict) or set(bbox) != {"box_2d", "label"}:
            raise OmniGroundError(f"OmniGround bboxes[{index}] must contain exactly box_2d and label")
        label = _nonempty_string(bbox["label"], f"bboxes[{index}].label")
        if label in labels:
            raise OmniGroundError(f"OmniGround bbox labels must be unique; duplicate {label!r}")
        labels.add(label)
        bboxes.append({"box_2d": _validate_box(bbox["box_2d"], index), "label": label})

    grounded_atoms: list[dict[str, Any]] = []
    for index, predicate in enumerate(predicates_value):
        if not isinstance(predicate, dict) or set(predicate) != {"name", "args"}:
            raise OmniGroundError(f"OmniGround predicates[{index}] must contain exactly name and args")
        name = _nonempty_string(predicate["name"], f"predicates[{index}].name")
        args = predicate["args"]
        if not isinstance(args, list) or not args:
            raise OmniGroundError(f"OmniGround predicates[{index}].args must be a non-empty list")
        validated_args = [_nonempty_string(argument, f"predicates[{index}].args") for argument in args]
        unknown_labels = sorted(set(validated_args).difference(labels))
        if unknown_labels:
            raise OmniGroundError(
                f"OmniGround predicates[{index}] references unknown bbox labels: {unknown_labels}"
            )
        grounded_atoms.append({"predicate": name, "args": validated_args})
    return bboxes, grounded_atoms


def _endpoint(base_url: str, endpoint: str) -> str:
    if not isinstance(base_url, str) or not base_url:
        raise ValueError("perception.vlm.url must be a non-empty URL")
    if endpoint not in _GENERATE_ENDPOINTS:
        raise ValueError("perception.vlm.endpoint must be /generate or /v1/generate")
    return f"{base_url.rstrip('/')}{endpoint}"


def _encode_image(image: Image.Image) -> bytes:
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")
    return image_bytes.getvalue()


def _error_from_response(status: int, body: bytes) -> OmniGroundHttpError:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        text = body.decode("utf-8", errors="replace")
        return OmniGroundHttpError(status, None, text or "non-JSON error response")
    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict) and isinstance(error.get("code"), str) and isinstance(error.get("message"), str):
        return OmniGroundHttpError(status, error["code"], error["message"])
    return OmniGroundHttpError(status, None, str(payload))


class OmniGroundClient:
    """Client for OmniGround's multipart ``POST /generate`` contract."""

    def __init__(
        self,
        *,
        url: str,
        endpoint: str,
        model_id: str,
        timeout_seconds: float,
        temperature: float | None = None,
    ) -> None:
        self.endpoint = _endpoint(url, endpoint)
        self.model_id = _nonempty_string(model_id, "model_id")
        if not math.isfinite(float(timeout_seconds)) or float(timeout_seconds) <= 0.0:
            raise ValueError("perception.vlm.timeout_seconds must be finite and positive")
        self.timeout_seconds = float(timeout_seconds)
        self.temperature = None if temperature is None else _finite_nonnegative_number(temperature, "temperature")

    async def generate_async(
        self,
        image: Image.Image,
        prompt: str,
        *,
        temperature: float | None = None,
        session: aiohttp.ClientSession | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if not isinstance(image, Image.Image):
            raise TypeError("OmniGround image must be a PIL.Image.Image")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("OmniGround prompt must be a non-empty string")
        selected_temperature = (
            self.temperature if temperature is None else _finite_nonnegative_number(temperature, "temperature")
        )
        data = aiohttp.FormData()
        data.add_field("image", _encode_image(image), filename="image.png", content_type="image/png")
        # Do not strip, wrap, or otherwise alter the TiPToP prompt.
        data.add_field("prompt", prompt, content_type="text/plain; charset=utf-8")
        data.add_field("model_id", self.model_id, content_type="text/plain; charset=utf-8")
        if selected_temperature is not None:
            data.add_field("temperature", str(selected_temperature), content_type="text/plain; charset=utf-8")

        async def _post(active_session: aiohttp.ClientSession) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
            started_at = time.perf_counter()
            try:
                async with active_session.post(
                    self.endpoint,
                    data=data,
                    timeout=aiohttp.ClientTimeout(total=self.timeout_seconds),
                ) as response:
                    body = await response.read()
                    if not 200 <= response.status < 300:
                        raise _error_from_response(response.status, body)
            except asyncio.TimeoutError as exc:
                raise OmniGroundError(f"OmniGround request timed out after {self.timeout_seconds:g} seconds") from exc
            except aiohttp.ClientError as exc:
                raise OmniGroundError(f"OmniGround request transport failure: {exc}") from exc
            try:
                payload = json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, ValueError) as exc:
                raise OmniGroundError("OmniGround success response must be JSON, not text or Markdown") from exc
            _log.info("OmniGround inference time=%.2fs", time.perf_counter() - started_at)
            return parse_grounding_result(payload)

        if session is not None:
            return await _post(session)
        async with aiohttp.ClientSession() as owned_session:
            return await _post(owned_session)

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        *,
        temperature: float | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        return asyncio.run(self.generate_async(image, prompt, temperature=temperature))


@cache
def omniground_client() -> OmniGroundClient:
    """Build the configured OmniGround client; model_id is intentionally required."""

    vlm_cfg = tiptop_cfg().perception.vlm
    model_id = vlm_cfg.get("model_id")
    if model_id is None:
        raise ValueError("perception.vlm.model_id is required for OmniGround")
    return OmniGroundClient(
        url=vlm_cfg.url,
        endpoint=vlm_cfg.endpoint,
        model_id=model_id,
        timeout_seconds=float(vlm_cfg.timeout_seconds),
        temperature=vlm_cfg.get("temperature", None),
    )


@cache
def load_prompt(prompt_name: str) -> str:
    """Load the prompt template without trimming or adding transport wrappers."""

    return (_PROMPTS_DIR / f"{prompt_name}.txt").read_text()


def _detect_prompt(task_instruction: str) -> str:
    if not isinstance(task_instruction, str) or not task_instruction:
        raise ValueError("task_instruction must be a non-empty string")
    return load_prompt("detect_and_translate").format(task_instruction=task_instruction)


def detect_and_translate(
    image: Image.Image,
    task_instruction: str,
    client: OmniGroundClient | None = None,
    temperature: float | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Detect objects and task predicates in one OmniGround request."""

    return (client or omniground_client()).generate(image, _detect_prompt(task_instruction), temperature=temperature)


async def detect_and_translate_async(
    image: Image.Image,
    task_instruction: str,
    client: OmniGroundClient | None = None,
    temperature: float | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Asynchronously detect objects and task predicates in one OmniGround request."""

    return await (client or omniground_client()).generate_async(
        image,
        _detect_prompt(task_instruction),
        temperature=temperature,
    )
