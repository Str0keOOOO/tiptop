"""Multipart client for the OmniGround grounding service."""

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


class OmniGroundError(RuntimeError):
    """An OmniGround request or response error."""


def parse_grounding_result(payload: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert OmniGround's direct response to TiPToP boxes and predicates."""
    if not isinstance(payload, dict) or set(payload) != {"bboxes", "predicates"}:
        raise OmniGroundError("OmniGround response must contain exactly bboxes and predicates")

    bbox_values = payload["bboxes"]
    predicate_values = payload["predicates"]
    if not isinstance(bbox_values, list) or not isinstance(predicate_values, list):
        raise OmniGroundError("OmniGround bboxes and predicates must be lists")

    bboxes: list[dict[str, Any]] = []
    labels: set[str] = set()
    for index, bbox in enumerate(bbox_values):
        if not isinstance(bbox, dict) or set(bbox) != {"box_2d", "label"}:
            raise OmniGroundError(f"bboxes[{index}] must contain exactly box_2d and label")
        box, label = bbox["box_2d"], bbox["label"]
        if not isinstance(label, str) or not label.strip() or label != label.strip():
            raise OmniGroundError(f"bboxes[{index}].label must be a non-empty, trimmed string")
        if label in labels:
            raise OmniGroundError(f"OmniGround labels must be unique; duplicate {label!r}")
        if (
            not isinstance(box, list)
            or len(box) != 4
            or any(isinstance(value, bool) or not isinstance(value, int) for value in box)
            or not all(0 <= value <= 1000 for value in box)
            or box[0] >= box[2]
            or box[1] >= box[3]
        ):
            raise OmniGroundError(f"bboxes[{index}].box_2d must be valid integer [ymin, xmin, ymax, xmax]")
        labels.add(label)
        bboxes.append({"box_2d": box, "label": label})

    grounded_atoms: list[dict[str, Any]] = []
    for index, predicate in enumerate(predicate_values):
        if not isinstance(predicate, dict) or set(predicate) != {"name", "args"}:
            raise OmniGroundError(f"predicates[{index}] must contain exactly name and args")
        name, args = predicate["name"], predicate["args"]
        if not isinstance(name, str) or not name.strip():
            raise OmniGroundError(f"predicates[{index}].name must be a non-empty string")
        if not isinstance(args, list) or not args or not all(isinstance(arg, str) and arg.strip() for arg in args):
            raise OmniGroundError(f"predicates[{index}].args must be a non-empty list of labels")
        unknown_labels = set(args).difference(labels)
        if unknown_labels:
            raise OmniGroundError(f"predicates[{index}] references unknown labels: {sorted(unknown_labels)}")
        grounded_atoms.append({"predicate": name, "args": args})
    return bboxes, grounded_atoms


class OmniGroundClient:
    """Client for OmniGround's multipart ``POST /generate`` contract."""

    def __init__(
        self,
        *,
        url: str,
        endpoint: str,
        timeout_seconds: float,
        temperature: float | None = None,
    ) -> None:
        if not url:
            raise ValueError("perception.omniground.url is required")
        if endpoint not in {"/generate", "/v1/generate"}:
            raise ValueError("perception.omniground.endpoint must be /generate or /v1/generate")
        timeout_seconds = float(timeout_seconds)
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
            raise ValueError("perception.omniground.timeout_seconds must be finite and positive")

        self.url = f"{url.rstrip('/')}{endpoint}"
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature

    async def generate_async(
        self,
        image: Image.Image,
        prompt: str,
        *,
        temperature: float | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL image")
        if not prompt:
            raise ValueError("prompt must be non-empty")

        selected_temperature = self.temperature if temperature is None else temperature
        if selected_temperature is not None:
            selected_temperature = float(selected_temperature)
            if not math.isfinite(selected_temperature) or selected_temperature < 0.0:
                raise ValueError("temperature must be finite and non-negative")

        image_bytes = io.BytesIO()
        image.save(image_bytes, format="PNG")
        data = aiohttp.FormData()
        data.add_field("image", image_bytes.getvalue(), filename="image.png", content_type="image/png")
        data.add_field("prompt", prompt, content_type="text/plain; charset=utf-8")
        if selected_temperature is not None:
            data.add_field("temperature", str(selected_temperature))

        started_at = time.perf_counter()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.url,
                    data=data,
                    timeout=aiohttp.ClientTimeout(total=self.timeout_seconds),
                ) as response:
                    status = response.status
                    body = await response.read()
        except asyncio.TimeoutError as exc:
            raise OmniGroundError(f"OmniGround timed out after {self.timeout_seconds:g} seconds") from exc
        except aiohttp.ClientError as exc:
            raise OmniGroundError(f"OmniGround transport error: {exc}") from exc

        if not 200 <= status < 300:
            detail = body.decode("utf-8", errors="replace") or "empty response"
            raise OmniGroundError(f"OmniGround HTTP {status}: {detail}")
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise OmniGroundError("OmniGround success response must be JSON") from exc

        _log.info("OmniGround inference took %.2fs", time.perf_counter() - started_at)
        return parse_grounding_result(payload)

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
    """Build the configured OmniGround client."""
    cfg = tiptop_cfg().perception.omniground
    return OmniGroundClient(
        url=cfg.url,
        endpoint=cfg.endpoint,
        timeout_seconds=cfg.timeout_seconds,
        temperature=cfg.get("temperature"),
    )


@cache
def load_prompt(prompt_name: str) -> str:
    """Load a prompt template from the prompts directory."""
    return (_PROMPTS_DIR / f"{prompt_name}.txt").read_text()


def detect_and_translate(
    image: Image.Image,
    task_instruction: str,
    client: OmniGroundClient | None = None,
    temperature: float | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Detect objects and task predicates in one OmniGround request."""
    prompt = load_prompt("detect_and_translate").format(task_instruction=task_instruction)
    return (client or omniground_client()).generate(image, prompt, temperature=temperature)


async def detect_and_translate_async(
    image: Image.Image,
    task_instruction: str,
    client: OmniGroundClient | None = None,
    temperature: float | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Asynchronously detect objects and task predicates in one OmniGround request."""
    prompt = load_prompt("detect_and_translate").format(task_instruction=task_instruction)
    return await (client or omniground_client()).generate_async(image, prompt, temperature=temperature)
