from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy import signal

from audio_engine.core.artifacts import atomic_path, derived_audio_path
from audio_engine.core.audio_normalize import pcm_array_to_float32, take_channel_0, write_pcm16_wav
from audio_engine.core.operator import BaseOperator, OperatorConfig
from audio_engine.core.registry import register_operator
from audio_engine.core.sample import Sample

DEFAULT_SAMPLE_RATE = 16000
_RESAMPLE_SUBDIR_PREFIX = "resample_v2_"


def resample_audio(data: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    """Resample a 1-D or 2-D signal. No-op when rates already match."""
    if source_sr == target_sr:
        return data
    if source_sr <= 0 or target_sr <= 0:
        raise ValueError(f"invalid sample rate: source={source_sr} target={target_sr}")
    n_out = int(len(data) * target_sr / source_sr)
    if n_out < 1:
        n_out = 1
    if data.ndim > 1:
        return np.column_stack(
            [signal.resample(data[:, ch], n_out) for ch in range(data.shape[1])]
        )
    return signal.resample(data, n_out)


def _original_source_rate(sample: Sample, fallback: int | None) -> int | None:
    """Prefer labels/quality original rate over the already-normalized WAV header."""
    for bag in (sample.labels, sample.quality):
        value = bag.get("source_sample_rate")
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return int(fallback) if fallback is not None else None


@register_operator
class ResampleOperator(BaseOperator):
    """Resample to target rate; passthrough when already at target (no full decode)."""

    name = "resample"
    version = "1.2.0"
    category = "audio"

    def compute_cache_key(self, sample: Sample, config: OperatorConfig) -> str:
        input_key = str(config.params.get("input_audio_key", "raw"))
        input_path = sample.audio.get(input_key) or ""
        payload = {
            "sha256": sample.sha256,
            "sample_id": sample.id,
            "operator": self.full_name,
            "version": self.version,
            "params": config.params,
            "input_audio_path": str(input_path),
            "audio_scene": sample.labels.get("audio_scene"),
            "labels_source_sample_rate": sample.labels.get("source_sample_rate"),
            "recipe": "resample_v2",
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode()).hexdigest()

    def _resolve_sr(self, sample: Sample, input_path: Path) -> int | None:
        """Prefer known sample metadata; fall back to header probe (no sample decode)."""
        if sample.sample_rate:
            return int(sample.sample_rate)
        try:
            return int(sf.info(str(input_path)).samplerate)
        except Exception:
            return None

    def _execute(self, sample: Sample, config: OperatorConfig) -> dict[str, Any]:
        input_key = config.params.get("input_audio_key", "raw")
        output_key = config.params.get("output_audio_key", "resampled_16k")
        target_sr = int(config.params.get("sample_rate", DEFAULT_SAMPLE_RATE))

        input_path = Path(sample.audio_path(input_key))
        current_sr = self._resolve_sr(sample, input_path)
        original_sr = _original_source_rate(sample, current_sr)

        # Already at target and known mono — alias path (pcm_to_wav already ch0).
        known_ch = sample.channels
        if (
            current_sr is not None
            and current_sr == target_sr
            and (known_ch is None or int(known_ch) == 1)
        ):
            return {
                "audio": {output_key: str(input_path.resolve())},
                "sample_rate": target_sr,
                "channels": 1,
                "duration": sample.duration,
                "labels": {"resampled": False},
                "quality": {
                    "resample": "passthrough",
                    # Keep original capture rate; do not overwrite with normalized 16 kHz.
                    "source_sample_rate": original_sr if original_sr is not None else current_sr,
                    "target_sample_rate": target_sr,
                },
                "lineage_entry": {
                    "operator": self.full_name,
                    "version": self.version,
                    "params": dict(config.params),
                    "input_key": input_key,
                    "output_key": output_key,
                    "output_path": str(input_path.resolve()),
                },
            }

        data, sr = sf.read(str(input_path), always_2d=True)
        mono, source_channels, channel_policy = take_channel_0(data)
        if sr == target_sr and channel_policy == "mono_passthrough":
            out_path = input_path
            to_write = mono
            resampled_flag = False
            quality_status = "passthrough"
        else:
            float_mono = pcm_array_to_float32(mono)
            to_write = float_mono if sr == target_sr else resample_audio(float_mono, sr, target_sr)
            out_path = derived_audio_path(
                config.output_dir, f"{_RESAMPLE_SUBDIR_PREFIX}{target_sr // 1000}k", sample
            )
            with atomic_path(out_path) as tmp:
                write_pcm16_wav(str(tmp), to_write, target_sr)
            resampled_flag = sr != target_sr
            quality_status = "converted" if resampled_flag else "channel_extract"

        duration = len(to_write) / target_sr if target_sr else sample.duration
        return {
            "audio": {output_key: str(out_path.resolve())},
            "sample_rate": target_sr,
            "channels": 1,
            "duration": duration,
            "labels": {"resampled": resampled_flag},
            "quality": {
                "resample": quality_status,
                "source_sample_rate": original_sr if original_sr is not None else int(sr),
                "target_sample_rate": target_sr,
                "source_channels": source_channels,
                "selected_channel": 0,
                "channel_policy": channel_policy,
            },
            "lineage_entry": {
                "operator": self.full_name,
                "version": self.version,
                "params": dict(config.params),
                "input_key": input_key,
                "output_key": output_key,
                "output_path": str(out_path.resolve()),
            },
        }
