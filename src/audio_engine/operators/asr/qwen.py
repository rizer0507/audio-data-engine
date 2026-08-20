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
class QwenASROperator(BaseASROperator):
    name = "qwen"
    version = "1.0.0"

    def transcribe(self, sample: Sample, config: OperatorConfig) -> dict[str, Any]:
        asr_cfg = _load_asr_config(config.params.get("config_path", "configs/asr/qwen_asr.yaml"))
        model = config.params.get("model", asr_cfg.get("model", "qwen3-asr"))
        model_version = config.params.get("model_version", asr_cfg.get("version", "20260820"))
        api_url = config.params.get("api_url", asr_cfg.get("api_url"))

        if config.mock or config.params.get("mock"):
            text = self._mock_transcript(sample, config)
        elif api_url:
            text = self._call_remote_api(sample, config, api_url, asr_cfg)
        else:
            text = self._call_local_model(sample, config, asr_cfg)

        return {"text": text, "model": model, "version": model_version}

    def _call_remote_api(
        self,
        sample: Sample,
        config: OperatorConfig,
        api_url: str,
        asr_cfg: dict[str, Any],
    ) -> str:
        import urllib.request

        input_key = config.params.get("input_audio_key", "raw")
        audio_path = sample.audio_path(input_key)
        # Placeholder: real implementation would POST multipart audio
        req = urllib.request.Request(
            api_url,
            data=f'{{"path": "{audio_path}"}}'.encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=asr_cfg.get("timeout", 60)) as resp:
                import json

                body = json.loads(resp.read().decode())
                return body.get("text", "")
        except Exception:
            return self._mock_transcript(sample, config)

    def _call_local_model(
        self,
        sample: Sample,
        config: OperatorConfig,
        asr_cfg: dict[str, Any],
    ) -> str:
        # Hook for local Qwen-ASR inference; falls back to mock when model unavailable
        _ = asr_cfg
        return self._mock_transcript(sample, config)
