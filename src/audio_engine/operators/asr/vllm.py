from __future__ import annotations

import json
import mimetypes
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from uuid import uuid4


def _encode_multipart(
    fields: dict[str, str],
    files: dict[str, tuple[str, bytes, str]],
) -> tuple[bytes, str]:
    boundary = f"----AudioEngineBoundary{uuid4().hex}"
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.extend(
            (
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                f"{value}\r\n".encode(),
            )
        )
    for name, (filename, data, content_type) in files.items():
        parts.extend(
            (
                f"--{boundary}\r\n".encode(),
                (
                    f'Content-Disposition: form-data; name="{name}"; '
                    f'filename="{filename}"\r\n'
                ).encode(),
                f"Content-Type: {content_type}\r\n\r\n".encode(),
                data,
                b"\r\n",
            )
        )
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def transcription_url(api_base: str) -> str:
    """Build a vLLM transcription URL from either a host or an OpenAI `/v1` base."""
    base = api_base.rstrip("/")
    if base.endswith("/audio/transcriptions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/audio/transcriptions"
    return f"{base}/v1/audio/transcriptions"


def call_vllm_transcription(audio_path: str, settings: dict[str, Any]) -> dict[str, Any]:
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")

    # vLLM's OpenAI-compatible transcription schema rejects unknown multipart
    # fields. Only send fields defined by that endpoint; generation-only options
    # such as max_completion_tokens belong to /v1/chat/completions.
    fields = {
        "model": str(settings["model"]),
        "language": str(settings.get("language", "zh")),
        "temperature": str(settings.get("temperature", 0)),
    }
    for key in ("prompt", "response_format"):
        if settings.get(key) is not None:
            fields[key] = str(settings[key])

    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    body, content_type = _encode_multipart(
        fields,
        {"file": (path.name, path.read_bytes(), mime_type)},
    )
    headers = {"Content-Type": content_type}
    if settings.get("api_key"):
        headers["Authorization"] = f"Bearer {settings['api_key']}"

    request = urllib.request.Request(
        transcription_url(str(settings["api_base"])),
        data=body,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=float(settings.get("timeout", 120))) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"vLLM ASR HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"vLLM ASR 请求失败: {exc.reason}") from exc

    if not isinstance(payload, dict) or "text" not in payload:
        raise RuntimeError(f"vLLM ASR 返回格式无效: {payload!r}")
    return {
        "text": str(payload["text"]),
        "language": payload.get("language") or settings.get("language"),
        "extra": {"raw_response": payload},
    }
