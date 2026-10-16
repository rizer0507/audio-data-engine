from __future__ import annotations

import base64
import json
import mimetypes
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from uuid import uuid4

_ASR_TEXT_TAG = "<asr_text>"
_LANG_PREFIX = "language "


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


def chat_completions_url(api_base: str) -> str:
    """Build a vLLM chat-completions URL from either a host or an OpenAI `/v1` base."""
    base = api_base.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    if base.endswith("/audio/transcriptions"):
        root = base[: -len("/audio/transcriptions")]
        return f"{root}/chat/completions"
    return f"{base}/v1/chat/completions"


def _chat_template_kwargs(settings: dict[str, Any]) -> dict[str, Any]:
    """Build kwargs for Qwen chat-template (force_language → Chinese)."""
    raw = settings.get("chat_template_kwargs")
    kwargs: dict[str, Any] = dict(raw) if isinstance(raw, dict) else {}
    force_language = settings.get("force_language")
    if force_language and "force_language" not in kwargs:
        kwargs["force_language"] = str(force_language)
    if "force_language" not in kwargs:
        kwargs["force_language"] = "Chinese"
    return kwargs


def _audio_data_url(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "audio/wav"
    if not mime_type.startswith("audio/"):
        mime_type = "audio/wav"
    audio_b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{audio_b64}"


def _parse_qwen_chat_content(content: str) -> tuple[str, str]:
    """Split optional `language X<asr_text>` prefix from chat content."""
    text = str(content or "").strip()
    if not text:
        return "", ""
    if _ASR_TEXT_TAG not in text:
        return "", text
    meta, body = text.split(_ASR_TEXT_TAG, 1)
    language = ""
    for line in meta.splitlines():
        line = line.strip()
        if line.lower().startswith(_LANG_PREFIX):
            language = line[len(_LANG_PREFIX) :].strip()
            break
    return language, body.strip()


def _http_json(
    url: str,
    *,
    body: bytes,
    headers: dict[str, str],
    timeout: float,
) -> dict[str, Any]:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"vLLM ASR HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"vLLM ASR 请求失败: {exc.reason}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"vLLM ASR 返回格式无效: {payload!r}")
    return payload


def call_vllm_chat_asr(audio_path: str, settings: dict[str, Any]) -> dict[str, Any]:
    """Call vLLM `/v1/chat/completions` for Qwen3-ASR with forced Chinese."""
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")

    context = settings.get("context")
    if context is None:
        context = settings.get("prompt") or ""
    payload = {
        "messages": [
            {
                "role": "system",
                "content": f"{context}",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio_url",
                        "audio_url": {
                            "url": _audio_data_url(path),
                        },
                    }
                ],
            },
        ],
        "chat_template_kwargs": _chat_template_kwargs(settings),
        "temperature": float(settings.get("temperature", 0)),
    }
    # model 对 served-model-name 路由必需；不改变上面与业务对齐的 messages/kwargs/temperature。
    if settings.get("model"):
        payload = {"model": str(settings["model"]), **payload}

    headers = {"Content-Type": "application/json"}
    if settings.get("api_key"):
        headers["Authorization"] = f"Bearer {settings['api_key']}"

    response = _http_json(
        chat_completions_url(str(settings["api_base"])),
        body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        timeout=float(settings.get("timeout", 120)),
    )
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"vLLM chat ASR 返回格式无效: {response!r}") from exc

    language, text = _parse_qwen_chat_content(str(content or ""))
    return {
        "text": text,
        "language": language
        or settings.get("chat_template_kwargs", {}).get("force_language")
        or settings.get("force_language")
        or settings.get("language"),
        "extra": {"raw_response": response},
    }


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

    payload = _http_json(
        transcription_url(str(settings["api_base"])),
        body=body,
        headers=headers,
        timeout=float(settings.get("timeout", 120)),
    )
    if "text" not in payload:
        raise RuntimeError(f"vLLM ASR 返回格式无效: {payload!r}")
    return {
        "text": str(payload["text"]),
        "language": payload.get("language") or settings.get("language"),
        "extra": {"raw_response": payload},
    }
