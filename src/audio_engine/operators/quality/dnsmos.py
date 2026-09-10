"""quality.dnsmos — DNSMOS P.835 sidecar (produces noise_risk for selection_v3)."""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
from dataclasses import asdict
from typing import Any, ClassVar

import soundfile as sf
import yaml

from audio_engine.core.operator import BaseOperator, OperatorConfig
from audio_engine.core.quality.dnsmos_p835 import (
    DnsmosP835Session,
    DnsmosScores,
    derive_noise_band_explicit,
    scores_to_quality_dict,
    DNSMOS_PREPROCESS_VERSION,
)
from audio_engine.core.registry import register_operator
from audio_engine.core.sample import Sample
from audio_engine.core.selection_v3.types import (
    DNSMOS_STATUS_FAILED,
    DNSMOS_STATUS_SUCCESS,
    DNSMOS_STATUS_UNSUPPORTED,
    QUALITY_POLICY_VERSION,
)


def _load_dnsmos_config(params: dict[str, Any]) -> dict[str, Any]:
    cfg_path = params.get("config_path") or params.get("config")
    loaded: dict[str, Any] = {}
    if cfg_path:
        raw = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"dnsmos config must be a mapping: {cfg_path}")
        loaded = dict(raw)
    # Explicit params override file
    merged = {**loaded, **{k: v for k, v in params.items() if k not in {"config", "config_path"}}}
    return merged


@register_operator
class DnsmosOperator(BaseOperator):
    """Score resampled_16k audio with fixed DNSMOS P.835; write quality.* fields.

    Startup fails if the ONNX model path is missing (unless ``risk_only``).
    Per-sample inference failures are isolated: status=failed, noise_band=unknown,
    never a default clean score.
    """

    name = "dnsmos"
    version = "1.0.0"
    category = "quality"

    _session: ClassVar[DnsmosP835Session | None] = None
    _session_key: ClassVar[str | None] = None
    _startup_validated: ClassVar[bool] = False

    def _get_session(self, cfg: dict[str, Any]) -> DnsmosP835Session:
        model_path = Path(str(cfg.get("model_path") or "")).expanduser()
        if not str(model_path):
            raise ValueError("quality.dnsmos requires model_path in config")
        from audio_engine.core.quality.dnsmos_p835 import file_digest
        key = str(model_path.resolve()) + (file_digest(model_path) if model_path.is_file() else "") + str(cfg.get("providers"))
        if DnsmosOperator._session is None or DnsmosOperator._session_key != key:
            providers = cfg.get("providers")
            if isinstance(providers, str):
                providers = [providers]
            DnsmosOperator._session = DnsmosP835Session(
                model_path, providers=list(providers) if providers else None
            )
            DnsmosOperator._session_key = key
            DnsmosOperator._startup_validated = True
        return DnsmosOperator._session

    def compute_cache_key(self, sample: Sample, config: OperatorConfig) -> str:
        # Include model + preprocess version so cache invalidates on vendor change.
        cfg = _load_dnsmos_config(dict(config.params))
        model_path = str(cfg.get("model_path") or "")
        digest = ""
        path = Path(model_path)
        if path.exists():
            from audio_engine.core.quality.dnsmos_p835 import file_digest

            digest = file_digest(path)[:16]
        payload = {"audio": sample.sha256, "config": cfg, "model_digest": digest,
                   "preprocess": DNSMOS_PREPROCESS_VERSION}
        if cfg.get("risk_only") or cfg.get("recompute_risk_only"):
            payload["scores"] = sample.quality
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()

    def _execute(self, sample: Sample, config: OperatorConfig) -> dict[str, Any]:
        cfg = _load_dnsmos_config(dict(config.params))
        input_key = str(cfg.get("input_audio_key") or "resampled_16k")
        risk_only = bool(cfg.get("risk_only") or cfg.get("recompute_risk_only"))
        calibrated = bool(cfg.get("calibrated", False))
        thresholds = cfg.get("thresholds") or {}
        clean_bak = float(thresholds.get("clean_bak", 3.5))
        clean_ovrl = float(thresholds.get("clean_ovrl", 3.2))
        moderate_bak = float(thresholds.get("moderate_bak", thresholds.get("t_bak", 3.0)))
        moderate_ovrl = float(thresholds.get("moderate_ovrl", thresholds.get("t_ovrl", 2.8)))
        policy_version = str(cfg.get("quality_policy_version") or QUALITY_POLICY_VERSION)
        policy_digest = hashlib.sha256(json.dumps({"thresholds": thresholds, "calibrated": calibrated,
            "version": policy_version}, sort_keys=True).encode()).hexdigest()

        if risk_only:
            existing = sample.quality or {}
            if existing.get("dnsmos_preprocess_version") != DNSMOS_PREPROCESS_VERSION:
                raise ValueError("stored DNSMOS scores use a different scoring algorithm; rescore before risk-only update")
            scores = DnsmosScores(
                sig=existing.get("dnsmos_sig"),
                bak=existing.get("dnsmos_bak"),
                ovrl=existing.get("dnsmos_ovrl"),
                status=str(existing.get("dnsmos_status") or DNSMOS_STATUS_FAILED),
                model_digest=str(existing.get("dnsmos_model_digest") or ""),
                preprocess_version=str(
                    existing.get("dnsmos_preprocess_version")
                    or "p835_primary_repeat_pad_v1"
                ),
            )
            risk = derive_noise_band_explicit(
                bak=scores.bak,
                ovrl=scores.ovrl,
                clean_bak=clean_bak,
                clean_ovrl=clean_ovrl,
                moderate_bak=moderate_bak,
                moderate_ovrl=moderate_ovrl,
                calibrated=calibrated,
                status=scores.status,
            )
            risk.quality_policy_version = policy_version
            quality = scores_to_quality_dict(scores, risk)
            quality["dnsmos_policy_digest"] = policy_digest
            return {
                "quality": quality,
                "lineage_entry": {
                    "operator": self.full_name,
                    "version": self.version,
                    "params": {"risk_only": True, "calibrated": calibrated},
                    "input_key": input_key,
                },
            }

        # Startup fail-fast: missing model / bad config
        session = self._get_session(cfg)

        try:
            audio_path = Path(sample.audio_path(input_key))
        except Exception as exc:  # noqa: BLE001
            scores = DnsmosScores(
                sig=None,
                bak=None,
                ovrl=None,
                status=DNSMOS_STATUS_UNSUPPORTED,
                error=f"missing_audio_key:{input_key}:{exc}",
                model_digest=session.model_digest,
            )
            risk = derive_noise_band_explicit(
                bak=None,
                ovrl=None,
                clean_bak=clean_bak,
                clean_ovrl=clean_ovrl,
                moderate_bak=moderate_bak,
                moderate_ovrl=moderate_ovrl,
                calibrated=calibrated,
                status=scores.status,
            )
            risk.quality_policy_version = policy_version
            return {
                "quality": scores_to_quality_dict(scores, risk),
                "lineage_entry": {
                    "operator": self.full_name,
                    "version": self.version,
                    "params": dict(cfg),
                    "input_key": input_key,
                },
            }

        if not audio_path.exists():
            scores = DnsmosScores(
                sig=None,
                bak=None,
                ovrl=None,
                status=DNSMOS_STATUS_FAILED,
                error=f"audio_not_found:{audio_path}",
                model_digest=session.model_digest,
            )
        else:
            try:
                from audio_engine.core.quality.dnsmos_p835 import file_digest
                from audio_engine.core.artifacts import atomic_write_json
                score_key = hashlib.sha256((file_digest(audio_path) + session.model_digest + DNSMOS_PREPROCESS_VERSION).encode()).hexdigest()
                score_path = config.cache_dir / "dnsmos_raw_scores" / f"{score_key}.json"
                if score_path.is_file():
                    scores = DnsmosScores(**json.loads(score_path.read_text(encoding="utf-8")))
                else:
                    data, sr = sf.read(str(audio_path), always_2d=False)
                    scores = session.score_array(data, int(sr))
                    if scores.status == DNSMOS_STATUS_SUCCESS:
                        atomic_write_json(score_path, asdict(scores))
            except Exception as exc:  # noqa: BLE001
                scores = DnsmosScores(
                    sig=None,
                    bak=None,
                    ovrl=None,
                    status=DNSMOS_STATUS_FAILED,
                    error=str(exc),
                    model_digest=session.model_digest,
                )

        if scores.status == DNSMOS_STATUS_SUCCESS:
            # Ensure we never invent clean defaults on partial outputs
            if scores.bak is None or scores.ovrl is None or scores.sig is None:
                scores.status = DNSMOS_STATUS_FAILED
                scores.error = scores.error or "incomplete_scores"

        risk = derive_noise_band_explicit(
            bak=scores.bak,
            ovrl=scores.ovrl,
            clean_bak=clean_bak,
            clean_ovrl=clean_ovrl,
            moderate_bak=moderate_bak,
            moderate_ovrl=moderate_ovrl,
            calibrated=calibrated,
            status=scores.status,
        )
        risk.quality_policy_version = policy_version
        quality = scores_to_quality_dict(scores, risk)
        quality["dnsmos_policy_digest"] = policy_digest
        return {
            "quality": quality,
            "lineage_entry": {
                "operator": self.full_name,
                "version": self.version,
                "params": {
                    "model_path": str(cfg.get("model_path")),
                    "model_digest": scores.model_digest,
                    "calibrated": calibrated,
                    "thresholds": {
                        "clean_bak": clean_bak,
                        "clean_ovrl": clean_ovrl,
                        "moderate_bak": moderate_bak,
                        "moderate_ovrl": moderate_ovrl,
                    },
                    "quality_policy_version": policy_version,
                },
                "input_key": input_key,
            },
        }
