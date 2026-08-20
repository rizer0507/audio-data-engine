from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from audio_engine.core.operator import OperatorConfig
from audio_engine.core.registry import register_operator
from audio_engine.core.sample import Sample
from audio_engine.operators.asr.base import BaseASROperator


def _load_asr_config(config_path: str | None) -> dict[str, Any]:
    if not config_path:
        return {}
    path = Path(config_path)
    if path.exists():
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {}


@register_operator
class SenseVoiceOperator(BaseASROperator):
    name = "sensevoice"
    version = "1.0.0"

    def transcribe(self, sample: Sample, config: OperatorConfig) -> dict[str, Any]:
        asr_cfg = _load_asr_config(config.params.get("config_path", "configs/asr/sensevoice.yaml"))
        model = config.params.get("model", asr_cfg.get("model", "sensevoice-small"))
        model_version = config.params.get("model_version", asr_cfg.get("version", "20260820"))

        if config.mock or config.params.get("mock"):
            text = self._mock_transcript(sample, config)
        else:
            text = self._call_model(sample, config, asr_cfg)

        return {"text": text, "model": model, "version": model_version}

    def _call_model(
        self,
        sample: Sample,
        config: OperatorConfig,
        asr_cfg: dict[str, Any],
    ) -> str:
        # Hook for FunASR / SenseVoice local inference
        try:
            from funasr import AutoModel  # type: ignore

            model_name = asr_cfg.get("model_path", model := asr_cfg.get("model", "iic/SenseVoiceSmall"))
            _ = model
            _model = AutoModel(model=model_name)
            input_key = config.params.get("input_audio_key", "raw")
            result = _model.generate(input=sample.audio_path(input_key))
            if isinstance(result, list) and result:
                return result[0].get("text", "")
            return str(result)
        except ImportError:
            return self._mock_transcript(sample, config)
        except Exception:
            return self._mock_transcript(sample, config)
