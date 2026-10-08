"""quality.silero_vad — full-pass Silero speech evidence (no re-segmentation)."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from audio_engine.core.operator import BaseOperator, OperatorConfig
from audio_engine.core.quality.silero_vad import SileroVadConfig, SileroVadSession, load_silero_vad_config
from audio_engine.core.registry import register_operator
from audio_engine.core.sample import Sample


def _load_vad_config(params: dict[str, Any]) -> SileroVadConfig:
    path = params.get("config_path") or params.get("silero_vad_config")
    loaded = load_silero_vad_config(path) if path else SileroVadConfig()
    # Inline param overrides (thresholds stay config-driven, not hardcoded).
    overrides = {
        k: v
        for k, v in params.items()
        if k
        not in {
            "config_path",
            "silero_vad_config",
            "selection_config",
            "config",
        }
    }
    if overrides:
        data = {
            "policy_version": loaded.policy_version,
            "input_audio_key": loaded.input_audio_key,
            "sample_rate": loaded.sample_rate,
            "onnx": loaded.onnx,
            "opset_version": loaded.opset_version,
            "model_path": loaded.model_path,
            "min_speech_ratio_present": loaded.min_speech_ratio_present,
            "max_speech_ratio_absent": loaded.max_speech_ratio_absent,
            "min_speech_duration_ms": loaded.min_speech_duration_ms,
            "get_speech_timestamps": loaded.get_speech_timestamps,
        }
        data.update(overrides)
        return SileroVadConfig.from_mapping(data)
    return loaded


@register_operator
class SileroVadOperator(BaseOperator):
    """Write vad_* / speech_* fields into sample.quality for every sample."""

    name = "silero_vad"
    version = "1.0.0"
    category = "quality"

    def __init__(self) -> None:
        super().__init__()
        self._session: SileroVadSession | None = None
        self._session_key: str | None = None

    def compute_cache_key(self, sample: Sample, config: OperatorConfig) -> str:
        params = dict(config.params)
        vad_cfg = _load_vad_config(params)
        payload = {
            "audio": sample.sha256,
            "vad": {
                "policy_version": vad_cfg.policy_version,
                "sample_rate": vad_cfg.sample_rate,
                "onnx": vad_cfg.onnx,
                "min_speech_ratio_present": vad_cfg.min_speech_ratio_present,
                "max_speech_ratio_absent": vad_cfg.max_speech_ratio_absent,
                "min_speech_duration_ms": vad_cfg.min_speech_duration_ms,
            },
            "operator": self.full_name,
            "version": self.version,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode()
        ).hexdigest()

    def _get_session(self, cfg: SileroVadConfig) -> SileroVadSession:
        key = json.dumps(
            {
                "policy": cfg.policy_version,
                "onnx": cfg.onnx,
                "model_path": cfg.model_path,
                "opset": cfg.opset_version,
            },
            sort_keys=True,
        )
        if self._session is None or self._session_key != key:
            self._session = SileroVadSession(cfg)
            self._session_key = key
        return self._session

    def _execute(self, sample: Sample, config: OperatorConfig) -> dict[str, Any]:
        params = dict(config.params)
        # 037 on-demand: skip non-candidates without forging present/absent.
        if params.get("respect_candidates"):
            labels = sample.labels if isinstance(sample.labels, dict) else {}
            cand = labels.get("silero_vad_v2_4_candidate")
            required = False
            if isinstance(cand, dict):
                required = bool(cand.get("required"))
            elif cand is True:
                required = True
            quality = dict(sample.quality or {})
            if not required:
                from audio_engine.core.selection_v3.types import ACOUSTIC_STATUS_NOT_REQUIRED

                if quality.get("vad_status") != ACOUSTIC_STATUS_NOT_REQUIRED:
                    quality.update(
                        {
                            "vad_status": ACOUSTIC_STATUS_NOT_REQUIRED,
                            "vad_speech_state": None,
                            "speech_ratio": None,
                            "vad_error": None,
                        }
                    )
                return {
                    "quality": quality,
                    "lineage": {
                        "operator": self.full_name,
                        "version": self.version,
                        "params": {"respect_candidates": True, "skipped": True},
                    },
                }

        vad_cfg = _load_vad_config(params)
        input_key = str(vad_cfg.input_audio_key or "resampled_16k")
        try:
            audio_path = sample.audio_path(input_key)
        except Exception:  # noqa: BLE001
            audio_path = (sample.audio or {}).get(input_key) or sample.source_path
        session = self._get_session(vad_cfg)
        fields = session.analyze(audio_path)
        quality = dict(sample.quality or {})
        quality.update(fields)
        return {
            "quality": quality,
            "lineage": {
                "operator": self.full_name,
                "version": self.version,
                "params": {
                    "input_audio_key": input_key,
                    "vad_config_version": vad_cfg.policy_version,
                    "vad_status": fields.get("vad_status"),
                },
            },
        }
