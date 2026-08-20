from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from audio_engine.core.operator import BaseOperator, OperatorConfig
from audio_engine.core.registry import register_operator
from audio_engine.core.sample import Sample


@register_operator
class VadOperator(BaseOperator):
    """Simple energy-based VAD; replace with Silero/WebRTC in production."""

    name = "vad"
    version = "1.0.0"
    category = "audio"

    def _execute(self, sample: Sample, config: OperatorConfig) -> dict[str, Any]:
        input_key = config.params.get("input_audio_key", "raw")
        threshold = float(config.params.get("energy_threshold", 0.01))
        frame_ms = int(config.params.get("frame_ms", 30))

        input_path = Path(sample.audio_path(input_key))
        data, sr = sf.read(str(input_path), always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)

        frame_len = max(1, int(sr * frame_ms / 1000))
        n_frames = max(1, len(data) // frame_len)
        frames = data[: n_frames * frame_len].reshape(n_frames, frame_len)
        energy = (frames**2).mean(axis=1)
        speech_frames = int((energy > threshold).sum())
        speech_ratio = round(speech_frames / n_frames, 4)

        return {
            "quality": {"speech_ratio": speech_ratio, "vad_method": "energy"},
            "lineage_entry": {
                "operator": self.full_name,
                "version": self.version,
                "params": dict(config.params),
                "input_key": input_key,
            },
        }
