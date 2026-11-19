from __future__ import annotations

import base64
import json
import mimetypes
import re
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from uuid import uuid4

_ASR_TEXT_TAG = "<asr_text>"
_LANG_PREFIX = "language "
_SOPHON_LAUGHTER = re.compile(
    r"<laughter\s*>(?:[\s哈呵嘿嘻，,.!！。~…]|ha|he|hi)*</laughter\s*>", re.IGNORECASE
)
_SOPHON_TAG = re.compile(
    r"\[[^\]]*(?:\]|$)|【[^】]*(?:】|$)|<\s*/?\s*(?:[^\W\d]|[|!])[^<>]*(?:>|$)"
    r"|\(\s*(?:笑|笑声|大笑|轻笑|叹气|呼吸|咳嗽|喘息|哭声|掌声|噪声|背景音乐|听不清|无语音"
    r"|开心|悲伤|愤怒|惊讶|兴奋|重读|laughter|sigh|breath|quick_breath|cough|noise|music|silence"
    r"|happy|sad|angry|excited|surprised|emphasis)\s*\)",
    re.IGNORECASE,
)


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


_OMNI_DEFAULT_PROMPT = (
    "这段音频是电话贷款营销业务场景中客户的回答，大部分客户会表示不用了、不需要等否定意图，输出ASR转译文本。"
)


def _sophon_chat_payload(path: Path, settings: dict[str, Any], context: str) -> dict[str, Any]:
    prompt = settings.get("system_prompt") or (
        "逐字转写音频中的人声，保留语气词和重复。只输出转写文本。"
        "没有可辨认的人声时输出空文本。"
    )
    if context:
        prompt += f"\n以下词表仅供参考，以实际听到的语音为准。\n{context}"
    return {
        "messages": [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "<|audio|>"},
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": base64.b64encode(path.read_bytes()).decode("ascii"),
                            "format": path.suffix.lower().lstrip(".") or "wav",
                        },
                    },
                ],
            },
        ],
        "modalities": ["text"],
        "stream": False,
        "language": settings.get("language") or "zh",
        "max_tokens": int(settings.get("max_new_tokens", 256)),
    }


def _omni_chat_payload(path: Path, settings: dict[str, Any]) -> dict[str, Any]:
    prompt = settings.get("prompt") or _OMNI_DEFAULT_PROMPT
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "audio_url", "audio_url": {"url": _audio_data_url(path)}},
                    {"type": "text", "text": str(prompt)},
                ],
            }
        ],
        "max_tokens": int(settings.get("max_new_tokens", 20480)),
        "stream": False,
    }


def call_vllm_chat_asr(audio_path: str, settings: dict[str, Any]) -> dict[str, Any]:
    """Call vLLM `/v1/chat/completions` with the configured ASR request format."""
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")

    context = settings.get("context")
    if context is None:
        context = settings.get("prompt") or ""
    chat_format = settings.get("chat_format")
    sophon = chat_format == "sophon"
    omni = chat_format == "omni"
    if sophon:
        payload = _sophon_chat_payload(path, settings, str(context))
    elif omni:
        payload = _omni_chat_payload(path, settings)
    else:
        payload = {
            "messages": [
                {"role": "system", "content": f"{context}"},
                {
                    "role": "user",
                    "content": [
                        {"type": "audio_url", "audio_url": {"url": _audio_data_url(path)}}
                    ],
                },
            ],
            "chat_template_kwargs": _chat_template_kwargs(settings),
        }
    default_temperature = 0.1 if omni else 0
    payload["temperature"] = float(settings.get("temperature", default_temperature))
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

    if sophon:
        raw_text = str(content or "")
        extra = {"raw_response": response}
        text = raw_text.strip()
        if settings.get("strip_sophon_tags"):
            extra["raw_text"] = raw_text
            text = unicodedata.normalize("NFKC", text)
            text = _SOPHON_LAUGHTER.sub("", text)
            text = _SOPHON_TAG.sub("", text).strip()
            if not any(char.isalnum() for char in text):
                text = ""
        return {
            "text": text,
            "language": payload["language"],
            "extra": extra,
        }

    if omni:
        return {
            "text": str(content or "").strip(),
            "language": settings.get("language") or "zh",
            "extra": {"raw_response": response},
        }

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
