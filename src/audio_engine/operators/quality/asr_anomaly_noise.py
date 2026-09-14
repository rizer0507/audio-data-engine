"""quality.asr_anomaly_noise — score only the ASR-anomaly subset (023).

Does not initialize DNSMOS when the anomaly subset is empty. A missing model
fails the diagnosis rows only; normal Chinese rows stay not_required and are
not dropped. This operator is not the research full-batch sidecar.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from audio_engine.core.operator import ManifestOperator, OperatorConfig
from audio_engine.core.registry import register_operator
from audio_engine.core.sample import Sample
from audio_engine.core.selection_v3.config import SelectionV3Config
from audio_engine.core.selection_v3.noise_trigger import (
    TRIGGER_VERSION,
    ScoreResult,
    diagnose_samples,
)
from audio_engine.core.selection_v3.types import (
    DNSMOS_STATUS_FAILED,
    DNSMOS_STATUS_SUCCESS,
    DNSMOS_STATUS_UNSUPPORTED,
)


class _DnsMosSubsetScorer:
    """Lazy DNSMOS session. Constructing this object does not load ONNX."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.session = None
        self.calls: list[str] = []
        self.model_id = str(cfg.get("model_path") or "dnsmos")
        self.preprocess_version = str(cfg.get("preprocess_version") or "p835_official_nonpersonalized_v2")

    def score(self, *, sample: Sample, audio_path: str, audio_sha256: str) -> ScoreResult:
        if self.session is None:
            model_path = Path(str(self.cfg.get("model_path") or "")).expanduser()
            if not model_path.is_file():
                return ScoreResult(
                    status=DNSMOS_STATUS_FAILED,
                    error=f"scoring_model_missing:{model_path}",
                    model_digest="",
                    preprocess_version=self.preprocess_version,
                )
            from audio_engine.core.quality.dnsmos_p835 import DnsmosP835Session

            providers = self.cfg.get("providers")
            if isinstance(providers, str):
                providers = [providers]
            self.session = DnsmosP835Session(model_path, providers=list(providers) if providers else None)
        path = Path(audio_path) if audio_path else None
        if path is None or not path.is_file():
            return ScoreResult(
                status=DNSMOS_STATUS_FAILED,
                error=f"audio_unreadable:{audio_path}",
                model_digest=getattr(self.session, "model_digest", ""),
                preprocess_version=self.preprocess_version,
            )
        self.calls.append(audio_sha256 or str(sample.id))
        try:
            import soundfile as sf

            data, sr = sf.read(str(path), always_2d=False)
            scores = self.session.score_array(data, int(sr))
        except Exception as exc:  # noqa: BLE001
            message = str(exc).lower()
            status = DNSMOS_STATUS_UNSUPPORTED if "unsupported" in message else DNSMOS_STATUS_FAILED
            return ScoreResult(
                status=status,
                error=str(exc),
                model_digest=getattr(self.session, "model_digest", ""),
                preprocess_version=self.preprocess_version,
                unsupported_reason=str(exc) if status == DNSMOS_STATUS_UNSUPPORTED else None,
            )
        if scores.status != DNSMOS_STATUS_SUCCESS or None in {scores.sig, scores.bak, scores.ovrl}:
            return ScoreResult(
                status=DNSMOS_STATUS_FAILED,
                error=scores.error or "incomplete_scores",
                model_digest=scores.model_digest,
                preprocess_version=self.preprocess_version,
            )
        return ScoreResult(
            status=DNSMOS_STATUS_SUCCESS,
            sig=scores.sig,
            bak=scores.bak,
            ovrl=scores.ovrl,
            model_digest=scores.model_digest,
            preprocess_version=self.preprocess_version,
        )


def _load_mapping(path: str | None, params: dict[str, Any]) -> dict[str, Any]:
    loaded: dict[str, Any] = {}
    if path:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"config must be a mapping: {path}")
        loaded = dict(raw)
    return {**loaded, **params}


@register_operator
class AsrAnomalyNoiseOperator(ManifestOperator):
    """Route after ASR, score the anomaly subset once, backfill by audio identity."""

    name = "asr_anomaly_noise"
    version = "1.0.0"
    category = "quality"

    def run(self, samples: list[Sample], config: OperatorConfig) -> list[Sample]:
        params = dict(config.params)
        selection_path = params.get("config_path") or params.get("selection_config")
        if not selection_path:
            raise ValueError("quality.asr_anomaly_noise requires selection_config")
        selection = SelectionV3Config.from_yaml(selection_path)
        selection.noise_policy = "asr_anomaly_noise_v1"
        dnsmos_cfg = _load_mapping(params.get("dnsmos_config"), {})
        calibrated = bool(dnsmos_cfg.get("calibrated", False))
        if calibrated:
            raise ValueError("asr_anomaly_noise_v1 refuses calibrated=true; thresholds are not hearing-calibrated")
        scorer = _DnsMosSubsetScorer(dnsmos_cfg)
        updated = [sample.model_copy(deep=True) for sample in samples]
        report = diagnose_samples(
            updated,
            selection,
            scorer,
            calibrated=False,
            threshold_version=str(dnsmos_cfg.get("quality_policy_version") or "uncalibrated"),
        )
        for sample in updated:
            sample.add_lineage(
                self.full_name,
                self.version,
                {
                    "policy": TRIGGER_VERSION,
                    "calls": report["calls"],
                    "not_required": report["not_required"],
                    "status": (sample.labels.get("noise_diagnosis") or {}).get("status"),
                },
            )
        return updated
