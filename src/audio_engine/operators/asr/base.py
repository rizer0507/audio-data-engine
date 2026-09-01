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

    def should_skip(self, sample: Sample, config: OperatorConfig) -> bool:
        transcript_key = config.params.get("transcript_key")
        if transcript_key:
            return not config.force and str(transcript_key) in sample.transcripts
        return super().should_skip(sample, config)

    def _execute(self, sample: Sample, config: OperatorConfig) -> dict[str, Any]:
        input_key = config.params.get("input_audio_key", "raw")
        result = self.transcribe(sample, config)
        transcript_key = str(config.params.get("transcript_key", self.name))
        return {
            "transcripts": {transcript_key: result},
            "lineage_entry": {
                "operator": self.full_name,
                "version": self.version,
                "params": dict(config.params),
                "input_key": input_key,
            },
        }

    def _mock_transcript(self, sample: Sample, config: OperatorConfig) -> str:
        return f"[mock:{self.name}:{sample.id}]"
