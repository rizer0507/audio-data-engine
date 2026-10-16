"""037 on-demand Silero VAD / energy / DNSMOS candidate selectors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from audio_engine.core.operator import ManifestOperator, OperatorConfig
from audio_engine.core.registry import register_operator
from audio_engine.core.sample import Sample
from audio_engine.core.selection_v3.config import SelectionV3Config
from audio_engine.core.selection_v3.family_evidence import collect_route_views
from audio_engine.core.selection_v3.five_class_v2 import build_family_states
from audio_engine.core.selection_v3.five_class_v2_4_on_demand_acoustic import (
    AsrRouteDecision,
    QUEUE_NONE,
    needs_dnsmos_v2_4,
    needs_energy_v2_4,
    needs_silero_v2_4,
    route_asr_queue,
)
from audio_engine.core.selection_v3.types import (
    ACOUSTIC_STATUS_NOT_REQUIRED,
    DNSMOS_STATUS_NOT_REQUIRED,
)


def _route_from_sample(sample: Sample, selection: SelectionV3Config) -> AsrRouteDecision:
    labels = sample.labels if isinstance(sample.labels, dict) else {}
    pre = labels.get("asr_route_v2_4")
    if isinstance(pre, dict) and pre.get("queue_id") is not None:
        return AsrRouteDecision(
            queue_id=str(pre.get("queue_id") or QUEUE_NONE),
            early_category=pre.get("early_category"),
            early_reason=pre.get("early_reason"),
            configured_families=list(
                pre.get("configured_families") or selection.ordered_families()
            ),
            family_states=dict(pre.get("family_states") or {}),
            all_families_dual_success=bool(pre.get("all_families_dual_success")),
            voicemail_hit=pre.get("voicemail_hit"),
            risk_tag=pre.get("risk_tag"),
            risk_evidence=pre.get("risk_evidence"),
        )
    routes = collect_route_views(sample, selection)
    bundles = build_family_states(routes, selection)
    return route_asr_queue(routes, bundles, selection)


def _not_required_vad_fields() -> dict[str, Any]:
    return {
        "vad_status": ACOUSTIC_STATUS_NOT_REQUIRED,
        "vad_speech_state": None,
        "speech_ratio": None,
        "speech_duration_ms": None,
        "longest_speech_ms": None,
        "speech_segment_count": None,
        "vad_error": None,
    }


def _not_required_energy_fields() -> dict[str, Any]:
    return {
        "energy_status": ACOUSTIC_STATUS_NOT_REQUIRED,
        "energy_state": None,
        "duration_ms": None,
        "rms_dbfs": None,
        "peak_dbfs": None,
        "non_silent_ratio": None,
        "strict_all_channels_zero": None,
        "any_channel_nonzero": None,
        "channel_count": None,
    }


def _not_required_dnsmos_fields() -> dict[str, Any]:
    return {
        "dnsmos_status": DNSMOS_STATUS_NOT_REQUIRED,
        "dnsmos_sig": None,
        "dnsmos_bak": None,
        "dnsmos_ovrl": None,
        "dnsmos_error": None,
    }


def _fact_status(quality: dict[str, Any], *keys: str) -> str:
    for key in keys:
        status = str(quality.get(key) or "").strip().lower()
        if status:
            return status
    return ""


def _apply_schedule_not_required(
    quality: dict[str, Any],
    *,
    status_keys: tuple[str, ...],
    fields: dict[str, Any],
    schedule_key: str,
) -> None:
    """Record schedule-only not_required without wiping existing SUCCESS facts.

    ``not_required`` is a consumer scheduling decision, not a shared evidence fact.
    Successful VAD/energy/DNSMOS values must remain readable by other modes (e.g. tagging).
    """
    quality[schedule_key] = ACOUSTIC_STATUS_NOT_REQUIRED
    existing = _fact_status(quality, *status_keys)
    if existing == "success":
        return
    quality.update(fields)


@register_operator
class SileroVadCandidatesOperator(ManifestOperator):
    """Mark Silero candidates (queues A—E); non-candidates get not_required."""

    name = "silero_vad_candidates"
    version = "1.0.0"
    category = "quality"

    def run(self, samples: list[Sample], config: OperatorConfig) -> list[Sample]:
        params = dict(config.params)
        selection_path = params.get("config_path") or params.get("selection_config")
        if not selection_path:
            raise ValueError("quality.silero_vad_candidates requires selection_config")
        selection = SelectionV3Config.from_yaml(selection_path)
        updated = [sample.model_copy(deep=True) for sample in samples]
        required = 0
        for sample in updated:
            route = _route_from_sample(sample, selection)
            decision = needs_silero_v2_4(route)
            labels = dict(sample.labels or {})
            labels["silero_vad_v2_4_candidate"] = decision
            sample.labels = labels
            quality = dict(sample.quality or {})
            if decision["required"]:
                required += 1
                quality["vad_schedule"] = "required"
                quality.setdefault("vad_status", "pending")
            else:
                _apply_schedule_not_required(
                    quality,
                    status_keys=("vad_status",),
                    fields=_not_required_vad_fields(),
                    schedule_key="vad_schedule",
                )
            sample.quality = quality
        return updated


@register_operator
class AudioEnergyCandidatesOperator(ManifestOperator):
    """Mark energy candidates (B ∩ VAD absent); others not_required."""

    name = "audio_energy_candidates"
    version = "1.0.0"
    category = "quality"

    def run(self, samples: list[Sample], config: OperatorConfig) -> list[Sample]:
        params = dict(config.params)
        selection_path = params.get("config_path") or params.get("selection_config")
        if not selection_path:
            raise ValueError("quality.audio_energy_candidates requires selection_config")
        selection = SelectionV3Config.from_yaml(selection_path)
        updated = [sample.model_copy(deep=True) for sample in samples]
        required = 0
        for sample in updated:
            route = _route_from_sample(sample, selection)
            quality = dict(sample.quality or {})
            decision = needs_energy_v2_4(
                route, vad_speech_state=quality.get("vad_speech_state")
            )
            labels = dict(sample.labels or {})
            labels["audio_energy_v2_4_candidate"] = decision
            sample.labels = labels
            if decision["required"]:
                required += 1
                quality["energy_schedule"] = "required"
                quality.setdefault("energy_status", "pending")
            else:
                _apply_schedule_not_required(
                    quality,
                    status_keys=("energy_status", "energy_compute_status"),
                    fields=_not_required_energy_fields(),
                    schedule_key="energy_schedule",
                )
            sample.quality = quality
        return updated


@register_operator
class DnsmosV24CandidatesOperator(ManifestOperator):
    """On-demand DNSMOS for (A∪B) ∩ speech; SIG used by classify, BAK/OVRL stored."""

    name = "dnsmos_v2_4_candidates"
    version = "1.0.0"
    category = "quality"

    def run(self, samples: list[Sample], config: OperatorConfig) -> list[Sample]:
        # Reuse v2.2 scoring machinery; candidate gate is v2.4-specific.
        from audio_engine.operators.quality.dnsmos_v2_2_candidates import (
            _DnsMosSubsetScorer,
            _apply_score_to_quality,
            _has_reusable_scores,
            _load_mapping,
            merge_dnsmos_sidecar_by_hash,
        )
        from audio_engine.core.selection_v3.input_contract import original_audio_sha256
        from audio_engine.core.selection_v3.types import DNSMOS_STATUS_SUCCESS

        params = dict(config.params)
        selection_path = params.get("config_path") or params.get("selection_config")
        if not selection_path:
            raise ValueError("quality.dnsmos_v2_4_candidates requires selection_config")
        selection = SelectionV3Config.from_yaml(selection_path)
        dnsmos_path = (
            params.get("dnsmos_config")
            or getattr(selection, "dnsmos_config_path", None)
            or "configs/quality/dnsmos_p835.yaml"
        )
        dnsmos_cfg = _load_mapping(str(dnsmos_path), {})
        samples = merge_dnsmos_sidecar_by_hash(
            samples, params.get("quality_sidecar_manifest")
        )
        updated = [sample.model_copy(deep=True) for sample in samples]
        scorer: _DnsMosSubsetScorer | None = None
        required = 0
        scored = 0
        for sample in updated:
            route = _route_from_sample(sample, selection)
            quality = dict(sample.quality or {})
            decision = needs_dnsmos_v2_4(
                route, vad_speech_state=quality.get("vad_speech_state")
            )
            labels = dict(sample.labels or {})
            labels["dnsmos_v2_4_candidate"] = decision
            sample.labels = labels
            if not decision["required"]:
                _apply_schedule_not_required(
                    quality,
                    status_keys=("dnsmos_status",),
                    fields=_not_required_dnsmos_fields(),
                    schedule_key="dnsmos_schedule",
                )
                sample.quality = quality
                continue
            required += 1
            quality["dnsmos_schedule"] = "required"
            if _has_reusable_scores(quality):
                sample.quality = quality
                continue
            if scorer is None:
                scorer = _DnsMosSubsetScorer(dnsmos_cfg)
            input_key = str(dnsmos_cfg.get("input_audio_key") or "resampled_16k")
            try:
                audio_path = sample.audio_path(input_key)
            except Exception:  # noqa: BLE001
                audio_path = (sample.audio or {}).get(input_key) or sample.source_path
            result = scorer.score(
                sample=sample,
                audio_path=str(audio_path or ""),
                audio_sha256=original_audio_sha256(sample),
            )
            _apply_score_to_quality(sample, result)
            if result.status == DNSMOS_STATUS_SUCCESS:
                scored += 1
        return updated
