"""v2.2 DNSMOS candidate selection and scoring (029 §5).

Scores samples that need DNSMOS for joint classification:
- all stable_empty + energy audible/borderline
- >=2 stable_empty + exactly 1 stable_text + audible/borderline

Reuses existing scores when present. Merge/reuse must use
sample_id + original_audio_sha256. Never forges noisy on failure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from audio_engine.core.operator import ManifestOperator, OperatorConfig
from audio_engine.core.registry import register_operator
from audio_engine.core.sample import Sample
from audio_engine.core.selection_v3.audio_energy import energy_evidence_from_quality
from audio_engine.core.selection_v3.config import SelectionV3Config
from audio_engine.core.selection_v3.dnsmos_decision import (
    derive_dnsmos_decision,
    load_dnsmos_decision_config,
)
from audio_engine.core.selection_v3.family_evidence import collect_route_views
from audio_engine.core.selection_v3.five_class_v2 import build_family_states
from audio_engine.core.selection_v3.input_contract import original_audio_sha256
from audio_engine.core.selection_v3.noise_trigger import ScoreResult
from audio_engine.core.selection_v3.types import (
    DNSMOS_STATUS_FAILED,
    DNSMOS_STATUS_SUCCESS,
    DNSMOS_STATUS_UNSUPPORTED,
    ENERGY_STATE_AUDIBLE,
    ENERGY_STATE_BORDERLINE,
    FAMILY_STATE_STABLE_EMPTY,
    FAMILY_STATE_STABLE_TEXT,
    FAMILY_STATE_UNAVAILABLE,
    FAMILY_STATE_UNSTABLE,
)


TRIGGER_VERSION_V22 = "dnsmos_candidates_v2_2"


def _load_mapping(path: str | None, params: dict[str, Any]) -> dict[str, Any]:
    loaded: dict[str, Any] = {}
    if path:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"config must be a mapping: {path}")
        loaded = dict(raw)
    return {**loaded, **params}


def needs_dnsmos_v2_2(sample: Sample, config: SelectionV3Config) -> dict[str, Any]:
    """Return candidate decision for DNSMOS under v2.2 rules."""
    routes = collect_route_views(sample, config)
    bundles = build_family_states(routes, config)
    duration = float(sample.duration) if sample.duration is not None else None
    energy = energy_evidence_from_quality(
        sample.quality if isinstance(sample.quality, dict) else {},
        config=config,
        duration_sec=duration,
    )
    stable_text = [b for b in bundles if b.state == FAMILY_STATE_STABLE_TEXT]
    stable_empty = [b for b in bundles if b.state == FAMILY_STATE_STABLE_EMPTY]
    unstable = [b for b in bundles if b.state == FAMILY_STATE_UNSTABLE]
    unavailable = [b for b in bundles if b.state == FAMILY_STATE_UNAVAILABLE]
    state = energy.energy_state
    all_empty = (
        bool(stable_empty)
        and not stable_text
        and not unstable
        and not unavailable
        and len(stable_empty) == len(bundles)
    )
    human_shape = (
        len(stable_empty) >= 2
        and len(stable_text) == 1
        and not unstable
        and not unavailable
    )
    energy_ok = state in {ENERGY_STATE_AUDIBLE, ENERGY_STATE_BORDERLINE}
    reasons: list[str] = []
    if all_empty and energy_ok:
        reasons.append("all_stable_empty_audible_or_borderline")
    if human_shape and energy_ok:
        reasons.append("two_empty_one_text_audible_or_borderline")
    return {
        "required": bool(reasons),
        "reasons": reasons,
        "energy_state": state,
        "audio_sha256": original_audio_sha256(sample),
        "trigger_version": TRIGGER_VERSION_V22,
    }


def _has_reusable_scores(quality: dict[str, Any]) -> bool:
    status = str(quality.get("dnsmos_status") or "").strip().lower()
    if status != DNSMOS_STATUS_SUCCESS:
        return False
    return None not in {
        quality.get("dnsmos_sig"),
        quality.get("dnsmos_bak"),
        quality.get("dnsmos_ovrl"),
    }


class _DnsMosSubsetScorer:
    """Lazy DNSMOS session. Constructing does not load ONNX."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.session = None
        self.calls: list[str] = []
        self.model_id = str(cfg.get("model_path") or "dnsmos")
        self.preprocess_version = str(
            cfg.get("preprocess_version") or "p835_official_nonpersonalized_v2"
        )

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
            self.session = DnsmosP835Session(
                model_path, providers=list(providers) if providers else None
            )
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
            status = (
                DNSMOS_STATUS_UNSUPPORTED
                if "unsupported" in message
                else DNSMOS_STATUS_FAILED
            )
            return ScoreResult(
                status=status,
                error=str(exc),
                model_digest=getattr(self.session, "model_digest", ""),
                preprocess_version=self.preprocess_version,
                unsupported_reason=str(exc) if status == DNSMOS_STATUS_UNSUPPORTED else None,
            )
        if scores.status != DNSMOS_STATUS_SUCCESS or None in {
            scores.sig,
            scores.bak,
            scores.ovrl,
        }:
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


def _apply_score_to_quality(sample: Sample, scored: ScoreResult) -> None:
    quality = dict(sample.quality or {})
    quality["dnsmos_status"] = scored.status
    quality["dnsmos_model_digest"] = scored.model_digest
    quality["dnsmos_preprocess_version"] = scored.preprocess_version
    if scored.status == DNSMOS_STATUS_SUCCESS and None not in {
        scored.sig,
        scored.bak,
        scored.ovrl,
    }:
        quality["dnsmos_sig"] = scored.sig
        quality["dnsmos_bak"] = scored.bak
        quality["dnsmos_ovrl"] = scored.ovrl
        quality["dnsmos_error"] = None
    else:
        # Do not forge noisy; leave scores null on failure.
        quality["dnsmos_sig"] = None
        quality["dnsmos_bak"] = None
        quality["dnsmos_ovrl"] = None
        quality["dnsmos_error"] = scored.error
    sample.quality = quality


def merge_dnsmos_sidecar_by_hash(
    samples: list[Sample],
    sidecar_path: str | Path | None,
) -> list[Sample]:
    """Merge DNSMOS quality fields by sample_id + original_audio_sha256."""
    if not sidecar_path:
        return samples
    from audio_engine.core.manifest import Manifest
    from audio_engine.core.selection_v3.input_contract import merge_field_by_join_key

    path = Path(sidecar_path)
    if not path.exists():
        raise ValueError(f"dnsmos sidecar not found: {path}")
    sidecar_samples = Manifest.load(path)
    by_key: dict[tuple[str, str], dict] = {}
    for item in sidecar_samples:
        key = (str(item.id).strip(), original_audio_sha256(item))
        if not key[1]:
            raise ValueError(f"dnsmos sidecar missing original_audio_sha256: {item.id}")
        if key in by_key:
            raise ValueError(f"duplicate dnsmos sidecar key: {key}")
        by_key[key] = dict(item.quality or {})
    base_hashes = {str(s.id).strip(): original_audio_sha256(s) for s in samples}
    for sid, digest in by_key:
        if sid in base_hashes and base_hashes[sid] and base_hashes[sid] != digest:
            raise ValueError(f"dnsmos sidecar audio hash mismatch: {sid}")
    return merge_field_by_join_key(samples, by_key, target="quality")


@register_operator
class DnsmosV22CandidatesOperator(ManifestOperator):
    """Score v2.2 DNSMOS candidates; reuse existing same-hash scores when present."""

    name = "dnsmos_v2_2_candidates"
    version = "1.0.0"
    category = "quality"

    def run(self, samples: list[Sample], config: OperatorConfig) -> list[Sample]:
        params = dict(config.params)
        selection_path = params.get("config_path") or params.get("selection_config")
        if not selection_path:
            raise ValueError("quality.dnsmos_v2_2_candidates requires selection_config")
        selection = SelectionV3Config.from_yaml(selection_path)
        dnsmos_cfg = _load_mapping(params.get("dnsmos_config"), {})
        decision_path = (
            params.get("dnsmos_decision_config")
            or selection.dnsmos_decision_config_path
            or "configs/quality/dnsmos_decision_v2_2.yaml"
        )
        decision = load_dnsmos_decision_config(decision_path)
        calibrated = bool(dnsmos_cfg.get("calibrated", False))
        if calibrated:
            raise ValueError(
                "dnsmos_v2_2_candidates refuses calibrated=true; thresholds are uncalibrated"
            )

        updated = [sample.model_copy(deep=True) for sample in samples]
        sidecar = params.get("quality_sidecar_manifest") or params.get("dnsmos_sidecar")
        if sidecar:
            updated = merge_dnsmos_sidecar_by_hash(updated, sidecar)

        scorer = _DnsMosSubsetScorer(dnsmos_cfg)
        scored = 0
        reused = 0
        required = 0
        for sample in updated:
            decision_need = needs_dnsmos_v2_2(sample, selection)
            labels = dict(sample.labels or {})
            labels["dnsmos_v2_2_candidate"] = decision_need
            sample.labels = labels
            if not decision_need["required"]:
                # Optional gold tagging uses existing scores only; do not force score.
                quality = dict(sample.quality or {})
                derived = derive_dnsmos_decision(quality, decision)
                quality.update(derived.as_dict())
                sample.quality = quality
                continue
            required += 1
            quality = dict(sample.quality or {})
            expected_hash = decision_need["audio_sha256"]
            sample_hash = original_audio_sha256(sample)
            if expected_hash and sample_hash and expected_hash != sample_hash:
                raise ValueError(
                    f"dnsmos candidate audio hash mismatch for {sample.id}: "
                    f"{expected_hash} vs {sample_hash}"
                )
            if _has_reusable_scores(quality):
                reused += 1
            else:
                try:
                    audio_path = str(sample.audio_path("resampled_16k"))
                except Exception:
                    audio_path = str(getattr(sample, "source_path", "") or "")
                result = scorer.score(
                    sample=sample,
                    audio_path=audio_path,
                    audio_sha256=sample_hash or str(sample.id),
                )
                _apply_score_to_quality(sample, result)
                scored += 1
                quality = dict(sample.quality or {})
            derived = derive_dnsmos_decision(quality, decision)
            quality.update(derived.as_dict())
            sample.quality = quality
            sample.add_lineage(
                self.full_name,
                self.version,
                {
                    "trigger_version": TRIGGER_VERSION_V22,
                    "required": True,
                    "reasons": decision_need["reasons"],
                    "dnsmos_decision_policy_version": decision.policy_version,
                },
            )

        for sample in updated:
            if not (sample.labels or {}).get("dnsmos_v2_2_candidate", {}).get("required"):
                sample.add_lineage(
                    self.full_name,
                    self.version,
                    {
                        "trigger_version": TRIGGER_VERSION_V22,
                        "required": False,
                        "dnsmos_decision_policy_version": decision.policy_version,
                    },
                )
        # Attach batch summary on first sample lineage already written; also store in params log via first.
        if updated:
            updated[0].labels["dnsmos_v2_2_batch"] = {
                "required": required,
                "scored": scored,
                "reused": reused,
                "calls": len(scorer.calls),
            }
        return updated

