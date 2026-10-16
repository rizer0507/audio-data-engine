"""Input resolution and output validation for tempo augmentation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import soundfile as sf

from audio_engine.core.manifest import file_sha256, probe_audio
from audio_engine.core.sample import Sample


class TempoAudioError(RuntimeError):
    """Per-row audio problem (missing key, decode failure, invalid output)."""


def resolve_input_audio(
    sample: Sample,
    *,
    input_key: str,
    allow_raw_fallback: bool,
) -> Path:
    audio = sample.audio or {}
    if input_key in audio and audio[input_key]:
        return Path(str(audio[input_key]))
    if input_key == "raw" and allow_raw_fallback and sample.source_path:
        return Path(sample.source_path)
    raise TempoAudioError(
        f"audio key {input_key!r} missing for sample {sample.id}; "
        "refusing to silently fall back to another key"
    )


def read_audio_info(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise TempoAudioError(f"input audio not found: {path}")
    if path.stat().st_size <= 0:
        raise TempoAudioError(f"input audio is empty: {path}")
    # Reject bare PCM without explicit params at operator level.
    suffix = path.suffix.lower()
    if suffix in {".pcm", ".raw"}:
        raise TempoAudioError(
            f"bare PCM input not supported without explicit conversion: {path}"
        )
    info = probe_audio(path)
    if not info.get("valid"):
        raise TempoAudioError(
            f"cannot decode input audio {path}: {info.get('error') or 'unknown'}"
        )
    try:
        sf_info = sf.info(str(path))
    except Exception as exc:  # noqa: BLE001
        raise TempoAudioError(f"cannot open input audio {path}: {exc}") from exc
    return {
        "path": path,
        "sample_rate": int(sf_info.samplerate),
        "channels": int(sf_info.channels),
        "duration": float(sf_info.duration),
        "frames": int(sf_info.frames),
        "sha256": file_sha256(path),
    }


def validate_output_wav(
    path: Path,
    *,
    expected_sr: int,
    expected_channels: int,
    expected_duration: float | None = None,
    duration_tol: float | None = None,
) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise TempoAudioError(f"output audio missing: {path}")
    if path.stat().st_size <= 0:
        raise TempoAudioError(f"output audio empty: {path}")
    try:
        info = sf.info(str(path))
    except Exception as exc:  # noqa: BLE001
        raise TempoAudioError(f"output audio not decodable {path}: {exc}") from exc
    if info.frames <= 0:
        raise TempoAudioError(f"output audio has no frames: {path}")
    if int(info.samplerate) != int(expected_sr):
        raise TempoAudioError(
            f"output sample_rate {info.samplerate} != expected {expected_sr}"
        )
    if int(info.channels) != int(expected_channels):
        raise TempoAudioError(
            f"output channels {info.channels} != expected {expected_channels}"
        )
    # Prefer PCM_16 / WAV
    subtype = str(getattr(info, "subtype", "") or "")
    if subtype and "PCM_16" not in subtype.upper() and "PCM_S16" not in subtype.upper():
        # soundfile may report PCM_16; accept common aliases.
        if "16" not in subtype:
            raise TempoAudioError(f"output subtype {subtype!r} is not PCM_16")
    duration = float(info.duration)
    if expected_duration is not None and duration_tol is not None:
        if abs(duration - expected_duration) > duration_tol:
            raise TempoAudioError(
                f"output duration {duration:.4f}s outside tolerance of "
                f"expected {expected_duration:.4f}s (±{duration_tol:.4f}s)"
            )
    return {
        "path": path,
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "duration": duration,
        "frames": int(info.frames),
        "subtype": subtype,
        "sha256": file_sha256(path),
    }


def duration_tolerance(expected: float) -> float:
    return max(0.05, 0.03 * abs(expected))
