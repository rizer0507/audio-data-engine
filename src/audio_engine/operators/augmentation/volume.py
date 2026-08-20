from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from audio_engine.core.operator import BaseOperator, OperatorConfig
from audio_engine.core.registry import register_operator
from audio_engine.core.sample import Sample


@register_operator
class VolumePerturbOperator(BaseOperator):
    name = "volume_perturb"
    version = "1.0.0"
    category = "augmentation"

    def _execute(self, sample: Sample, config: OperatorConfig) -> dict[str, Any]:
        input_key = config.params.get("input_audio_key", "raw")
        output_key = config.params.get("output_audio_key", "aug_volume")
        gain_db = float(config.params.get("gain_db", 3))

        input_path = Path(sample.audio_path(input_key))
        data, sr = sf.read(str(input_path), always_2d=False)
        scaled = data * (10 ** (gain_db / 20))
        scaled = np.clip(scaled, -1.0, 1.0)

        out_dir = config.output_dir / "augment" / "volume"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"{sample.id}_vol{gain_db:+.0f}db.wav"
        sf.write(str(output_path), scaled, sr)

        return {
            "audio": {output_key: str(output_path.resolve())},
            "lineage_entry": {
                "operator": self.full_name,
                "version": self.version,
                "params": {"gain_db": gain_db},
                "input_key": input_key,
                "output_key": output_key,
                "output_path": str(output_path.resolve()),
            },
        }
