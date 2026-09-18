"""Short-audio probes for stage-1 family adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from audio_engine.operators.asr.vllm import call_vllm_transcription


@dataclass(frozen=True)
class ProbeResult:
    ok: bool
    family: str
    audio_path: str
    text: str
    detail: dict[str, Any]


def require_probe_audio(path: str | Path | None) -> Path:
    if path is None or not str(path).strip():
        raise ValueError("未指定探针音频；请传 --audio 或在 runtime 配置 probe.audio_path")
    audio = Path(path).expanduser().resolve()
    if not audio.is_file():
        raise FileNotFoundError(f"探针音频不存在: {audio}")
    return audio


def probe_vllm_transcription(
    *,
    family: str,
    audio_path: str | Path,
    api_base: str,
    model: str,
    api_key: str = "dummy",
    language: str = "zh",
    timeout_s: float = 120.0,
) -> ProbeResult:
    audio = require_probe_audio(audio_path)
    result = call_vllm_transcription(
        str(audio),
        {
            "api_base": api_base,
            "api_key": api_key,
            "model": model,
            "language": language,
            "temperature": 0,
            "timeout": timeout_s,
        },
    )
    text = str(result.get("text") or "").strip()
    return ProbeResult(
        ok=bool(text),
        family=family,
        audio_path=str(audio),
        text=text,
        detail={"language": result.get("language"), "raw": result.get("extra")},
    )
