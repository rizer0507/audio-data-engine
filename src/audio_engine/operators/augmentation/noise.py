from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from audio_engine.core.artifacts import atomic_path, derived_audio_path
from audio_engine.core.operator import BaseOperator, OperatorConfig
from audio_engine.core.registry import register_operator
from audio_engine.core.sample import Sample


@register_operator
class AddNoiseOperator(BaseOperator):
    name = "add_noise"
    version = "1.0.0"
    category = "augmentation"

    def _execute(self, sample: Sample, config: OperatorConfig) -> dict[str, Any]:
        input_key = config.params.get("input_audio_key", "raw")
        output_key = config.params.get("output_audio_key", "aug_noise")
        snr_db = float(config.params.get("snr", 5))
        seed = int(config.params.get("seed", 42))
        noise_id = config.params.get("noise_id", "white")

        rng = np.random.default_rng(seed)
        input_path = Path(sample.audio_path(input_key))
        data, sr = sf.read(str(input_path), always_2d=False)
        mono = data.mean(axis=1) if data.ndim > 1 else data

        noise = rng.standard_normal(len(mono)).astype(np.float32)
        signal_power = np.mean(mono**2) + 1e-10
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = noise * np.sqrt(noise_power / (np.mean(noise**2) + 1e-10))
        augmented = mono + noise

        output_path = derived_audio_path(
            config.output_dir, "augment/noise", sample, stem_suffix="_aug_noise"
        )
        with atomic_path(output_path) as tmp:
            sf.write(str(tmp), augmented, sr)

        return {
            "audio": {output_key: str(output_path.resolve())},
            "lineage_entry": {
                "operator": self.full_name,
                "version": self.version,
                "params": {"snr": snr_db, "noise_id": noise_id, "seed": seed},
                "input_key": input_key,
                "output_key": output_key,
                "output_path": str(output_path.resolve()),
            },
        }
