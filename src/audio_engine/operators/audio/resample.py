from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy import signal

from audio_engine.core.artifacts import atomic_path, derived_audio_path
from audio_engine.core.operator import BaseOperator, OperatorConfig
from audio_engine.core.registry import register_operator
from audio_engine.core.sample import Sample


@register_operator
class ResampleOperator(BaseOperator):
    """Resample to target rate; passthrough when already at target (no full decode)."""

    name = "resample"
    version = "1.1.0"
    category = "audio"

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
        target_sr = int(config.params.get("sample_rate", 16000))

        input_path = Path(sample.audio_path(input_key))
        current_sr = self._resolve_sr(sample, input_path)

        # Already at target — requirements met, alias path, do not decode/write.
        if current_sr is not None and current_sr == target_sr:
            return {
                "audio": {output_key: str(input_path.resolve())},
                "sample_rate": target_sr,
                "duration": sample.duration,
                "labels": {"resampled": False},
                "quality": {"resample": "passthrough", "source_sample_rate": current_sr},
                "lineage_entry": {
                    "operator": self.full_name,
                    "version": self.version,
                    "params": dict(config.params),
                    "input_key": input_key,
                    "output_key": output_key,
                    "output_path": str(input_path.resolve()),
                },
            }

        data, sr = sf.read(str(input_path), always_2d=False)
        if sr == target_sr:
            out_path = input_path
            resampled_flag = False
            quality_status = "passthrough"
        else:
            if data.ndim > 1:
                resampled = np.column_stack(
                    [
                        signal.resample(data[:, ch], int(len(data) * target_sr / sr))
                        for ch in range(data.shape[1])
                    ]
                )
            else:
                resampled = signal.resample(data, int(len(data) * target_sr / sr))

            out_path = derived_audio_path(
                config.output_dir, f"resample_{target_sr // 1000}k", sample
            )
            with atomic_path(out_path) as tmp:
                sf.write(str(tmp), resampled, target_sr)
            resampled_flag = True
            quality_status = "converted"

        duration = len(data) / sr if sr else sample.duration
        return {
            "audio": {output_key: str(out_path.resolve())},
            "sample_rate": target_sr,
            "duration": duration,
            "labels": {"resampled": resampled_flag},
            "quality": {
                "resample": quality_status,
                "source_sample_rate": sr,
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
