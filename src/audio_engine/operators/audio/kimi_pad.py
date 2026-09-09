"""Kimi-vLLM exclusive duration-bucket pad. Not a shared cleaning / training step."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from audio_engine.core.artifacts import atomic_path, derived_audio_path
from audio_engine.core.operator import BaseOperator, OperatorConfig
from audio_engine.core.registry import register_operator
from audio_engine.core.sample import Sample

# Tail-silence pad targets in seconds. d > last bucket is not truncated.
KIMI_PAD_BUCKETS: tuple[int, ...] = (3, 6, 10, 15, 30)
KIMI_PAD_OVER_SECONDS = 30.0
DEFAULT_INPUT_KEY = "resampled_16k"
DEFAULT_OUTPUT_KEY = "kimi_padded_16k"


@dataclass(frozen=True)
class KimiPadPlan:
    """How one WAV should be padded for Kimi-vLLM inference."""

    mode: str  # padded | passthrough | over_30s
    target_s: int | None
    source_duration: float
    sample_rate: int
    frames: int
    channels: int
    subtype: str | None
    needs_write: bool


def resolve_kimi_pad_target(
    duration: float,
    *,
    buckets: tuple[int, ...] = KIMI_PAD_BUCKETS,
    over_seconds: float = KIMI_PAD_OVER_SECONDS,
) -> int | None:
    """Return the bucket upper bound in seconds, or None when duration exceeds the last bucket."""
    if duration < 0:
        raise ValueError(f"duration must be >= 0, got {duration}")
    if duration > over_seconds:
        return None
    for target in buckets:
        if duration <= target:
            return int(target)
    return None


def _audio_duration_seconds(path: Path) -> tuple[float, int, int, int, str | None]:
    info = sf.info(str(path))
    sr = int(info.samplerate)
    frames = int(info.frames)
    channels = int(info.channels or 1)
    subtype = getattr(info, "subtype", None)
    if sr <= 0:
        raise ValueError(f"invalid sample rate in {path}: {sr}")
    return frames / float(sr), sr, frames, channels, subtype


def plan_kimi_pad(
    path: Path,
    *,
    duration: float | None = None,
    buckets: tuple[int, ...] = KIMI_PAD_BUCKETS,
    over_seconds: float = KIMI_PAD_OVER_SECONDS,
) -> KimiPadPlan:
    """Decide pad vs passthrough from original duration; never truncate >30s audio."""
    wav_duration, sr, frames, channels, subtype = _audio_duration_seconds(path)
    source_duration = float(duration) if duration is not None else wav_duration
    target_s = resolve_kimi_pad_target(
        source_duration, buckets=buckets, over_seconds=over_seconds
    )
    if target_s is None:
        return KimiPadPlan(
            mode="over_30s",
            target_s=None,
            source_duration=source_duration,
            sample_rate=sr,
            frames=frames,
            channels=channels,
            subtype=subtype,
            needs_write=False,
        )
    target_frames = int(round(target_s * sr))
    if frames >= target_frames:
        return KimiPadPlan(
            mode="passthrough",
            target_s=target_s,
            source_duration=source_duration,
            sample_rate=sr,
            frames=frames,
            channels=channels,
            subtype=subtype,
            needs_write=False,
        )
    return KimiPadPlan(
        mode="padded",
        target_s=target_s,
        source_duration=source_duration,
        sample_rate=sr,
        frames=frames,
        channels=channels,
        subtype=subtype,
        needs_write=True,
    )


def pad_wav_file(src: Path, dst: Path, plan: KimiPadPlan) -> Path:
    """Write tail-silence padded WAV. Sampling rate and channel count are preserved."""
    if not plan.needs_write or plan.target_s is None:
        return src
    data, sr = sf.read(str(src), always_2d=False)
    if sr != plan.sample_rate:
        raise ValueError(f"sample rate changed while reading {src}: {sr} != {plan.sample_rate}")
    target_frames = int(round(plan.target_s * sr))
    current = int(data.shape[0])
    if current >= target_frames:
        return src
    pad_width = target_frames - current
    if data.ndim == 1:
        padded = np.concatenate([data, np.zeros(pad_width, dtype=data.dtype)])
    else:
        padded = np.concatenate(
            [data, np.zeros((pad_width, data.shape[1]), dtype=data.dtype)]
        )
    dst = Path(dst)
    with atomic_path(dst) as tmp:
        write_kwargs: dict[str, Any] = {}
        if plan.subtype:
            write_kwargs["subtype"] = plan.subtype
        sf.write(str(tmp), padded, sr, **write_kwargs)
    return dst


def _bucket_params(config: OperatorConfig) -> tuple[tuple[int, ...], float]:
    raw_buckets = config.params.get("buckets", KIMI_PAD_BUCKETS)
    buckets = tuple(int(item) for item in raw_buckets)
    if not buckets:
        raise ValueError("kimi duration pad buckets must not be empty")
    over_seconds = float(config.params.get("over_seconds", KIMI_PAD_OVER_SECONDS))
    return buckets, over_seconds


@register_operator
class KimiDurationPadOperator(BaseOperator):
    """Pad resampled_16k into kimi_padded_16k for Kimi-vLLM only.

    Does not overwrite resampled_16k or Sample.duration. Samples already at the
    bucket cap alias the original path. Audio longer than 30s is passed through
    and labelled over_30s instead of being truncated.
    """

    name = "kimi_duration_pad"
    version = "1.0.0"
    category = "audio"

    def compute_cache_key(self, sample: Sample, config: OperatorConfig) -> str:
        buckets, over_seconds = _bucket_params(config)
        input_key = config.params.get("input_audio_key", DEFAULT_INPUT_KEY)
        input_path = Path(sample.audio_path(input_key))
        plan = plan_kimi_pad(
            input_path,
            duration=sample.duration,
            buckets=buckets,
            over_seconds=over_seconds,
        )
        params = dict(config.params)
        params["resolved_kimi_pad_target_s"] = plan.target_s
        params["resolved_kimi_pad_mode"] = plan.mode
        params["resolved_kimi_pad_buckets"] = list(buckets)
        params["resolved_kimi_pad_over_seconds"] = over_seconds
        return super().compute_cache_key(
            sample, config.model_copy(update={"params": params})
        )

    def _execute(self, sample: Sample, config: OperatorConfig) -> dict[str, Any]:
        input_key = config.params.get("input_audio_key", DEFAULT_INPUT_KEY)
        output_key = config.params.get("output_audio_key", DEFAULT_OUTPUT_KEY)
        buckets, over_seconds = _bucket_params(config)
        input_path = Path(sample.audio_path(input_key)).resolve()
        plan = plan_kimi_pad(
            input_path,
            duration=sample.duration,
            buckets=buckets,
            over_seconds=over_seconds,
        )

        if plan.needs_write:
            stem_suffix = f"_pad{plan.target_s}s"
            out_path = derived_audio_path(
                config.output_dir,
                "kimi_padded_16k",
                sample,
                stem_suffix=stem_suffix,
            )
            output_path = pad_wav_file(input_path, out_path, plan).resolve()
        else:
            output_path = input_path

        return {
            "audio": {output_key: str(output_path)},
            "labels": {
                "kimi_pad_mode": plan.mode,
                "kimi_pad_target_s": plan.target_s,
            },
            "quality": {
                "kimi_pad": plan.mode,
                "kimi_pad_source_duration": plan.source_duration,
                "kimi_pad_sample_rate": plan.sample_rate,
            },
            "lineage_entry": {
                "operator": self.full_name,
                "version": self.version,
                "params": {
                    **dict(config.params),
                    "resolved_kimi_pad_target_s": plan.target_s,
                    "resolved_kimi_pad_mode": plan.mode,
                },
                "input_key": input_key,
                "output_key": output_key,
                "output_path": str(output_path),
            },
        }
