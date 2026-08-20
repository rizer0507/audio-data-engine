from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy import signal

from audio_engine.core.operator import BaseOperator, OperatorConfig
from audio_engine.core.registry import register_operator
from audio_engine.core.sample import Sample


@register_operator
class ResampleOperator(BaseOperator):
    name = "resample"
    version = "1.0.0"
    category = "audio"

    def _execute(self, sample: Sample, config: OperatorConfig) -> dict[str, Any]:
        input_key = config.params.get("input_audio_key", "raw")
        output_key = config.params.get("output_audio_key", "resampled_16k")
        target_sr = int(config.params.get("sample_rate", 16000))

        input_path = Path(sample.audio_path(input_key))
        data, sr = sf.read(str(input_path), always_2d=False)
        if sr == target_sr:
            out_path = input_path
        else:
            if data.ndim > 1:
                resampled = np.column_stack(
                    [signal.resample(data[:, ch], int(len(data) * target_sr / sr)) for ch in range(data.shape[1])]
                )
            else:
                resampled = signal.resample(data, int(len(data) * target_sr / sr))

            out_dir = config.output_dir / f"resample_{target_sr // 1000}k"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{sample.id}.wav"
            sf.write(str(out_path), resampled, target_sr)

        duration = len(data) / sr if sr else sample.duration
        return {
            "audio": {output_key: str(out_path.resolve())},
            "sample_rate": target_sr,
            "duration": duration,
            "lineage_entry": {
                "operator": self.full_name,
                "version": self.version,
                "params": dict(config.params),
                "input_key": input_key,
                "output_key": output_key,
                "output_path": str(out_path.resolve()),
            },
        }
