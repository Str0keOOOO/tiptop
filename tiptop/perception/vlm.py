"""Client for TiPToP's image-and-text VLM HTTP service.

The configured service receives the image and TiPToP prompt as multipart form
data and returns the generated JSON text.
"""

import asyncio
import io
import json
import logging
import time
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import aiohttp
from PIL import Image

from tiptop.config import tiptop_cfg

_log = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"


@dataclass(frozen=True)
class VLMResponse:
    """Minimal response object matching the ``response.text`` call-site shape."""

    text: str


class types:
    """Small compatibility namespace for the Gemini generation config shape."""

    @dataclass(frozen=True)
    class ThinkingConfig:
        thinking_budget: int | None = None

    @dataclass(frozen=True)
    class GenerateContentConfig:
        temperature: float | None = None
        thinking_config: "types.ThinkingConfig | None" = None


class _AsyncModels:
    def __init__(self, client: "VLMClient"):
        self._client = client

    async def generate_content(
        self,
        model: str,
        contents: list[Image.Image | str],
        config: types.GenerateContentConfig | None = None,
    ) -> VLMResponse:
        image, prompt = _image_and_prompt(contents)
        return await self._client._generate_content(
            image,
            prompt,
            model,
            temperature=config.temperature if config else None,
        )


class _AsyncClient:
    def __init__(self, client: "VLMClient"):
        self.models = _AsyncModels(client)


class _Models:
    def __init__(self, client: "VLMClient"):
        self._client = client

    def generate_content(
        self,
        model: str,
        contents: list[Image.Image | str],
        config: types.GenerateContentConfig | None = None,
    ) -> VLMResponse:
        return asyncio.run(self._client.aio.models.generate_content(model, contents, config))


class VLMClient:
    """Local adapter exposing the Gemini client calls TiPToP already uses."""

    def __init__(self):
        self.aio = _AsyncClient(self)
        self.models = _Models(self)

    async def _generate_content(
        self,
        image: Image.Image,
        prompt: str,
        model_id: str,
        temperature: float | None,
    ) -> VLMResponse:
        async with aiohttp.ClientSession() as session:
            return await _post_generate_content(session, image, prompt, model_id, temperature)





@cache
def load_prompt(prompt_name: str) -> str:
    """Load a prompt template from the prompts directory."""
    return (_PROMPTS_DIR / f"{prompt_name}.txt").read_text().strip()


@cache
def gemini_client() -> VLMClient:
    """Return the local VLM client adapter; no Gemini SDK is used."""
    return VLMClient()

def load_json(response_text: str) -> list | dict:
    """Extract JSON string from code fencing if present."""
    cleaned_text = response_text.strip()
    if cleaned_text.startswith("```json"):
        cleaned_text = cleaned_text.replace("```json", "").replace("```", "")
    elif cleaned_text.startswith("```"):
        cleaned_text = cleaned_text.replace("```", "")

    try:
        results = json.loads(cleaned_text)
    except json.decoder.JSONDecodeError:
        _log.error(f"Invalid JSON: {cleaned_text}")
        raise
    return results


def _parse_response(response_text: str) -> tuple[list, list]:
    """Parse VLM response text into TiPToP bboxes and grounded atoms."""
    try:
        result = load_json(response_text)
    except Exception:
        raise ValueError(f"VLM returned a non-JSON response; check for a discrepancy in your image: {response_text}")
    bboxes = result.get("bboxes", [])
    grounded_atoms = [
        {"predicate": spec["name"], "args": spec["args"]}
        for spec in result.get("predicates", [])
        if spec.get("name") and spec.get("args")
    ]
    return bboxes, grounded_atoms


def _endpoint(server_url: str, path: str) -> str:
    return f"{server_url.rstrip('/')}/{path.lstrip('/')}"


def _encode_image(image: Image.Image) -> bytes:
    """Encode a PIL image as PNG for a portable multipart request."""
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")
    return image_bytes.getvalue()


def _image_and_prompt(contents: list[Image.Image | str]) -> tuple[Image.Image, str]:
    """Extract TiPToP's single image and prompt from SDK-style contents."""
    if len(contents) != 2 or not isinstance(contents[0], Image.Image) or not isinstance(contents[1], str):
        raise ValueError("VLM contents must be [PIL.Image.Image, prompt]")
    return contents[0], contents[1]


async def _post_generate_content(
    session: aiohttp.ClientSession,
    image: Image.Image,
    prompt: str,
    model_id: str = "gemini-robotics-er-1.6-preview",
    temperature: float | None = None,
) -> VLMResponse:
    """Send an image and prompt to the configured VLM."""
    vlm_cfg = tiptop_cfg().perception.vlm
    request_endpoint = _endpoint(vlm_cfg.url, vlm_cfg.endpoint)
    data = aiohttp.FormData()
    data.add_field("image", _encode_image(image), filename="image.png", content_type="image/png")
    data.add_field("prompt", prompt, content_type="text/plain; charset=utf-8")
    data.add_field("model_id", model_id, content_type="text/plain; charset=utf-8")
    if temperature is not None:
        data.add_field("temperature", str(temperature), content_type="text/plain; charset=utf-8")

    start_time = time.perf_counter()
    _log.debug("Sending VLM inference request to %s", request_endpoint)
    async with session.post(
        request_endpoint,
        data=data,
        timeout=aiohttp.ClientTimeout(total=vlm_cfg.timeout_seconds),
    ) as response:
        response_text = await response.text()
        if not 200 <= response.status < 300:
            raise RuntimeError(
                f"VLM request failed with status code {response.status}. Response: {response_text}"
            )

    _log.info("VLM inference time=%.2fs", time.perf_counter() - start_time)
    return VLMResponse(text=response_text)


def detect_and_translate(
    image: Image.Image,
    task_instruction: str,
    client: VLMClient | None = None,
    model_id: str = "gemini-robotics-er-1.6-preview",
    temperature: float | None = None,
) -> tuple[list[dict], list[dict]]:
    """Detect objects and translate task in a single VLM API call."""
    client = client or gemini_client()
    prompt = load_prompt("detect_and_translate").format(task_instruction=task_instruction)
    response = client.models.generate_content(
        model=model_id,
        contents=[image, prompt],
        config=types.GenerateContentConfig(
            temperature=temperature, thinking_config=types.ThinkingConfig(thinking_budget=0)
        ),
    )
    return _parse_response(response.text)


async def detect_and_translate_async(
    image: Image.Image,
    task_instruction: str,
    client: VLMClient | None = None,
    model_id: str = "gemini-robotics-er-1.6-preview",
    temperature: float | None = None,
) -> tuple[list[dict], list[dict]]:
    """Asynchronously detect objects and translate task in a single VLM API call."""
    client = client or gemini_client()
    prompt = load_prompt("detect_and_translate").format(task_instruction=task_instruction)
    response = await client.aio.models.generate_content(
        model=model_id,
        contents=[image, prompt],
        config=types.GenerateContentConfig(
            temperature=temperature, thinking_config=types.ThinkingConfig(thinking_budget=0)
        ),
    )
    return _parse_response(response.text)
