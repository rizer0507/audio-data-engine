from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from audio_engine.core.artifacts import atomic_path, derived_audio_path
from audio_engine.core.manifest import probe_audio
from audio_engine.core.operator import BaseOperator, OperatorConfig
from audio_engine.core.registry import register_operator
from audio_engine.core.sample import Sample

# Already a container format — no pcm header decode needed.
_CONTAINER_EXTS = {".wav", ".flac", ".ogg", ".mp3", ".m4a"}
_PCM_EXTS = {".pcm", ".raw"}


@register_operator
class PcmToWavOperator(BaseOperator):
    """Convert raw PCM to WAV; passthrough when input is already a container format."""

    name = "pcm_to_wav"
    version = "1.1.0"
    category = "audio"

    def _execute(self, sample: Sample, config: OperatorConfig) -> dict[str, Any]:
        input_key = config.params.get("input_audio_key", "raw")
        output_key = config.params.get("output_audio_key", "pcm_to_wav")
        sample_rate = int(config.params.get("sample_rate", 8000))
        channels = int(config.params.get("channels", 1))
        dtype = config.params.get("dtype", "int16")

        input_path = Path(sample.audio_path(input_key))
        suffix = input_path.suffix.lower()

        # Already wav/flac/... — requirements met, no conversion.
        if suffix in _CONTAINER_EXTS:
            meta = probe_audio(input_path)
            updates: dict[str, Any] = {
                "audio": {output_key: str(input_path.resolve())},
                "labels": {"pcm_converted": False},
                "quality": {"pcm_to_wav": "passthrough"},
                "lineage_entry": {
                    "operator": self.full_name,
                    "version": self.version,
                    "params": dict(config.params),
                    "input_key": input_key,
                    "output_key": output_key,
                    "output_path": str(input_path.resolve()),
                },
            }
            if meta.get("valid"):
                updates["sample_rate"] = meta.get("sample_rate")
                updates["channels"] = meta.get("channels")
                updates["duration"] = meta.get("duration")
            return updates

        if suffix not in _PCM_EXTS:
            raise ValueError(
                f"audio.pcm_to_wav: unsupported format '{suffix}' for {input_path}"
            )

        output_path = derived_audio_path(config.output_dir, "pcm_to_wav", sample)

        raw = np.fromfile(input_path, dtype=dtype)
        if channels > 1:
            raw = raw.reshape(-1, channels)
        with atomic_path(output_path) as tmp:
            sf.write(str(tmp), raw, sample_rate, subtype="PCM_16")

        duration = len(raw) / sample_rate / (channels if channels > 1 else 1)
        return {
            "audio": {output_key: str(output_path.resolve())},
            "sample_rate": sample_rate,
            "channels": channels,
            "duration": duration,
            "labels": {"pcm_converted": True},
            "quality": {"pcm_to_wav": "converted"},
            "lineage_entry": {
                "operator": self.full_name,
                "version": self.version,
                "params": dict(config.params),
                "input_key": input_key,
                "output_key": output_key,
                "output_path": str(output_path.resolve()),
            },
        }
