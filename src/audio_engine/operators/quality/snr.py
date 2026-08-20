from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from audio_engine.core.operator import BaseOperator, OperatorConfig
from audio_engine.core.registry import register_operator
from audio_engine.core.sample import Sample


@register_operator
class SnrOperator(BaseOperator):
    name = "snr"
    version = "1.0.0"
    category = "quality"

    def _execute(self, sample: Sample, config: OperatorConfig) -> dict[str, Any]:
        input_key = config.params.get("input_audio_key", "raw")
        frame_ms = int(config.params.get("frame_ms", 30))

        input_path = Path(sample.audio_path(input_key))
        data, sr = sf.read(str(input_path), always_2d=False)
        mono = data.mean(axis=1) if data.ndim > 1 else data

        frame_len = max(1, int(sr * frame_ms / 1000))
        n_frames = max(1, len(mono) // frame_len)
        frames = mono[: n_frames * frame_len].reshape(n_frames, frame_len)
        energy = (frames**2).mean(axis=1)
        sorted_energy = np.sort(energy)
        noise_frames = max(1, int(n_frames * 0.1))
        signal_power = sorted_energy[noise_frames:].mean() if n_frames > noise_frames else energy.mean()
        noise_power = sorted_energy[:noise_frames].mean()
        snr = 10 * np.log10((signal_power + 1e-10) / (noise_power + 1e-10))

        return {
            "quality": {"snr": round(float(snr), 2)},
            "lineage_entry": {
                "operator": self.full_name,
                "version": self.version,
                "params": dict(config.params),
                "input_key": input_key,
            },
        }
