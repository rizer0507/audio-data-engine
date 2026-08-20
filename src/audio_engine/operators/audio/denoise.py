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
class DenoiseOperator(BaseOperator):
    """Lightweight spectral gating denoise; swap for DNS/DeepFilterNet in production."""

    name = "denoise"
    version = "1.0.0"
    category = "audio"

    def _execute(self, sample: Sample, config: OperatorConfig) -> dict[str, Any]:
        input_key = config.params.get("input_audio_key", "raw")
        output_key = config.params.get("output_audio_key", "denoised")
        model = config.params.get("model", "spectral_gate")
        gate_db = float(config.params.get("gate_db", -40))

        input_path = Path(sample.audio_path(input_key))
        data, sr = sf.read(str(input_path), always_2d=False)
        mono = data.mean(axis=1) if data.ndim > 1 else data

        spectrum = np.fft.rfft(mono)
        magnitude = np.abs(spectrum)
        threshold = 10 ** (gate_db / 20) * magnitude.max()
        cleaned = np.fft.irfft(np.where(magnitude > threshold, spectrum, 0))
        cleaned = cleaned[: len(mono)].astype(np.float32)

        output_path = derived_audio_path(config.output_dir, f"denoise/{model}", sample)
        with atomic_path(output_path) as tmp:
            sf.write(str(tmp), cleaned, sr)

        return {
            "audio": {output_key: str(output_path.resolve())},
            "lineage_entry": {
                "operator": self.full_name,
                "version": self.version,
                "params": dict(config.params),
                "input_key": input_key,
                "output_key": output_key,
                "output_path": str(output_path.resolve()),
            },
        }
