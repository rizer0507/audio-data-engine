from __future__ import annotations

from pathlib import Path
from typing import Any

from audio_engine.core.manifest import probe_audio
from audio_engine.core.operator import BaseOperator, OperatorConfig
from audio_engine.core.registry import register_operator
from audio_engine.core.sample import Sample


@register_operator
class ProbeOperator(BaseOperator):
    """Re-probe audio metadata and mark broken samples."""

    name = "probe"
    version = "1.0.0"
    category = "quality"

    def _execute(self, sample: Sample, config: OperatorConfig) -> dict[str, Any]:
        input_key = config.params.get("input_audio_key", "raw")
        path = Path(sample.audio_path(input_key))
        meta = probe_audio(path)
        broken = not meta.get("valid", False)

        updates: dict[str, Any] = {
            "labels": {"broken": broken},
            "quality": {
                "probe_valid": not broken,
                "probe_error": meta.get("error", ""),
            },
            "lineage_entry": {
                "operator": self.full_name,
                "version": self.version,
                "params": {"input_audio_key": input_key},
                "input_key": input_key,
            },
        }
        if not broken:
            updates["sample_rate"] = meta.get("sample_rate")
            updates["channels"] = meta.get("channels")
            updates["duration"] = meta.get("duration")
        return updates
