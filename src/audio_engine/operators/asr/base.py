from __future__ import annotations

from abc import abstractmethod
from typing import Any

from audio_engine.core.operator import BaseOperator, OperatorConfig
from audio_engine.core.sample import Sample


class BaseASROperator(BaseOperator):
    category = "asr"

    @abstractmethod
    def transcribe(self, sample: Sample, config: OperatorConfig) -> dict[str, Any]:
        """Return transcript dict with text, model, version, etc."""

    def _execute(self, sample: Sample, config: OperatorConfig) -> dict[str, Any]:
        input_key = config.params.get("input_audio_key", "raw")
        result = self.transcribe(sample, config)
        return {
            "transcripts": {self.name: result},
            "lineage_entry": {
                "operator": self.full_name,
                "version": self.version,
                "params": dict(config.params),
                "input_key": input_key,
            },
        }

    def _mock_transcript(self, sample: Sample, config: OperatorConfig) -> str:
        return f"[mock:{self.name}:{sample.id}]"
