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
class SpeedPerturbOperator(BaseOperator):
    name = "speed_perturb"
    version = "1.0.0"
    category = "augmentation"

    def _execute(self, sample: Sample, config: OperatorConfig) -> dict[str, Any]:
        input_key = config.params.get("input_audio_key", "raw")
        output_key = config.params.get("output_audio_key", "aug_speed")
        factor = float(config.params.get("factor", 1.1))

        input_path = Path(sample.audio_path(input_key))
        data, sr = sf.read(str(input_path), always_2d=False)
        mono = data.mean(axis=1) if data.ndim > 1 else data
        new_len = max(1, int(len(mono) / factor))
        perturbed = signal.resample(mono, new_len)

        out_dir = config.output_dir / "augment" / "speed"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"{sample.id}_sp{factor:.2f}.wav"
        sf.write(str(output_path), perturbed.astype(np.float32), sr)

        return {
            "audio": {output_key: str(output_path.resolve())},
            "duration": len(perturbed) / sr,
            "lineage_entry": {
                "operator": self.full_name,
                "version": self.version,
                "params": {"factor": factor},
                "input_key": input_key,
                "output_key": output_key,
                "output_path": str(output_path.resolve()),
            },
        }
