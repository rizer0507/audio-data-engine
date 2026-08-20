from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from audio_engine.core.operator import BaseOperator, OperatorConfig
from audio_engine.core.registry import register_operator
from audio_engine.core.sample import Sample


@register_operator
class PcmToWavOperator(BaseOperator):
    name = "pcm_to_wav"
    version = "1.0.0"
    category = "audio"

    def _execute(self, sample: Sample, config: OperatorConfig) -> dict[str, Any]:
        input_key = config.params.get("input_audio_key", "raw")
        output_key = config.params.get("output_audio_key", "pcm_to_wav")
        sample_rate = int(config.params.get("sample_rate", 8000))
        channels = int(config.params.get("channels", 1))
        dtype = config.params.get("dtype", "int16")

        input_path = Path(sample.audio_path(input_key))
        out_dir = config.output_dir / "pcm_to_wav"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"{sample.id}.wav"

        if input_path.suffix.lower() in {".wav", ".flac", ".ogg"}:
            return {
                "audio": {output_key: str(input_path.resolve())},
                "lineage_entry": {
                    "operator": self.full_name,
                    "version": self.version,
                    "params": dict(config.params),
                    "input_key": input_key,
                    "output_key": output_key,
                    "output_path": str(input_path.resolve()),
                },
            }

        raw = np.fromfile(input_path, dtype=dtype)
        if channels > 1:
            raw = raw.reshape(-1, channels)
        sf.write(str(output_path), raw, sample_rate, subtype="PCM_16")

        duration = len(raw) / sample_rate / (channels if channels > 1 else 1)
        return {
            "audio": {output_key: str(output_path.resolve())},
            "sample_rate": sample_rate,
            "channels": channels,
            "duration": duration,
            "lineage_entry": {
                "operator": self.full_name,
                "version": self.version,
                "params": dict(config.params),
                "input_key": input_key,
                "output_key": output_key,
                "output_path": str(output_path.resolve()),
            },
        }
