"""EfficientAT candidate selection (034 §8.2). Does not run inference."""

from __future__ import annotations

from typing import Any

from audio_engine.core.operator import ManifestOperator, OperatorConfig
from audio_engine.core.registry import register_operator
from audio_engine.core.sample import Sample
from audio_engine.core.selection_v3.audio_energy import energy_evidence_from_quality
from audio_engine.core.selection_v3.config import SelectionV3Config
from audio_engine.core.selection_v3.family_evidence import collect_route_views
from audio_engine.core.selection_v3.five_class_v2 import (
    _is_critical_short_response,
    build_family_states,
)
from audio_engine.core.selection_v3.five_class_v2_3_min_acoustic import (
    TEXT_ALL_EMPTY,
    TEXT_CONSENSUS,
    TEXT_DIVERGENT,
    TEXT_SINGLE,
    TEXT_STRICT,
    derive_text_evidence_state,
)
from audio_engine.core.selection_v3.input_contract import original_audio_sha256
from audio_engine.core.selection_v3.types import (
    ENERGY_STATE_INAUDIBLE,
    FAMILY_STATE_STABLE_TEXT,
)

TRIGGER_VERSION_V23 = "efficientat_candidates_v2_3"


def needs_efficientat_v2_3(sample: Sample, config: SelectionV3Config) -> dict[str, Any]:
    """Return candidate decision for EfficientAT under v2.3 rules."""
    routes = collect_route_views(sample, config)
    bundles = build_family_states(routes, config)
    duration = float(sample.duration) if sample.duration is not None else None
    energy = energy_evidence_from_quality(
        sample.quality if isinstance(sample.quality, dict) else {},
        config=config,
        duration_sec=duration,
    )
    quality = sample.quality if isinstance(sample.quality, dict) else {}
    text_state = derive_text_evidence_state(bundles, config)
    vad = str(quality.get("vad_speech_state") or "").strip().lower()
    energy_state = energy.energy_state
    reasons: list[str] = []

    if text_state == TEXT_ALL_EMPTY and energy_state != ENERGY_STATE_INAUDIBLE:
        reasons.append("all_empty_not_inaudible")
    if text_state == TEXT_SINGLE:
        reasons.append("single_text")
    if text_state == TEXT_CONSENSUS and vad != "present":
        reasons.append("consensus_text_vad_not_present")
    if text_state == TEXT_DIVERGENT:
        reasons.append("divergent_text")
    if text_state == TEXT_STRICT and vad == "absent":
        reasons.append("strict_conflict_vad_absent")

    critical = False
    stable_text = [b for b in bundles if b.state == FAMILY_STATE_STABLE_TEXT]
    if len(stable_text) == 1:
        critical = _is_critical_short_response(
            stable_text[0].representative_text or "", config
        )
    if critical and vad != "present":
        reasons.append("critical_short_vad_not_present")

    return {
        "required": bool(reasons),
        "reasons": reasons,
        "text_evidence_state": text_state,
        "vad_speech_state": vad or None,
        "energy_state": energy_state,
        "audio_sha256": original_audio_sha256(sample),
        "trigger_version": TRIGGER_VERSION_V23,
    }


@register_operator
class EfficientAtCandidatesOperator(ManifestOperator):
    """Mark EfficientAT candidates; write skipped placeholders for non-candidates."""

    name = "efficientat_candidates"
    version = "1.0.0"
    category = "quality"

    def run(self, samples: list[Sample], config: OperatorConfig) -> list[Sample]:
        from audio_engine.core.quality.efficientat import (
            load_efficientat_config,
            skipped_efficientat_fields,
        )

        params = dict(config.params)
        selection_path = params.get("config_path") or params.get("selection_config")
        if not selection_path:
            raise ValueError("quality.efficientat_candidates requires selection_config")
        selection = SelectionV3Config.from_yaml(selection_path)
        eat_path = (
            params.get("efficientat_config")
            or selection.efficientat_config_path
            or "configs/quality/efficientat_v1.yaml"
        )
        eat_cfg = load_efficientat_config(eat_path)

        updated = [sample.model_copy(deep=True) for sample in samples]
        required = 0
        for sample in updated:
            decision = needs_efficientat_v2_3(sample, selection)
            labels = dict(sample.labels or {})
            labels["efficientat_v2_3_candidate"] = decision
            sample.labels = labels
            quality = dict(sample.quality or {})
            if decision["required"]:
                required += 1
                # Leave scoring to quality.efficientat; do not forge probs.
                quality.setdefault("efficientat_status", "pending")
            else:
                quality.update(skipped_efficientat_fields(eat_cfg))
            sample.quality = quality
            sample.add_lineage(
                self.full_name,
                self.version,
                {
                    "trigger_version": TRIGGER_VERSION_V23,
                    "required": bool(decision["required"]),
                    "reasons": decision["reasons"],
                },
            )
        if updated:
            updated[0].labels["efficientat_v2_3_batch"] = {
                "required": required,
                "total": len(updated),
            }
        return updated
