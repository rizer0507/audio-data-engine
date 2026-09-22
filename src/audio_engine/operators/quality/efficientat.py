"""quality.efficientat — score EfficientAT candidates only."""

from __future__ import annotations

from typing import Any

from audio_engine.core.operator import ManifestOperator, OperatorConfig
from audio_engine.core.quality.acoustic_evidence import (
    aggregate_acoustic_state,
    load_acoustic_evidence_config,
    resolve_audio_event_state,
)
from audio_engine.core.quality.efficientat import (
    EfficientAtSession,
    load_efficientat_config,
    skipped_efficientat_fields,
)
from audio_engine.core.registry import register_operator
from audio_engine.core.sample import Sample
from audio_engine.core.selection_v3.config import SelectionV3Config
from audio_engine.core.selection_v3.input_contract import original_audio_sha256
from audio_engine.operators.quality.efficientat_candidates import (
    TRIGGER_VERSION_V23,
    needs_efficientat_v2_3,
)


def _has_reusable_scores(quality: dict[str, Any]) -> bool:
    status = str(quality.get("efficientat_status") or "").strip().lower()
    if status != "success":
        return False
    return quality.get("speech_prob") is not None


@register_operator
class EfficientAtOperator(ManifestOperator):
    """Run EfficientAT only when candidate.required; never forge on failure."""

    name = "efficientat"
    version = "1.0.0"
    category = "quality"

    def run(self, samples: list[Sample], config: OperatorConfig) -> list[Sample]:
        params = dict(config.params)
        selection_path = params.get("config_path") or params.get("selection_config")
        if not selection_path:
            raise ValueError("quality.efficientat requires selection_config")
        selection = SelectionV3Config.from_yaml(selection_path)
        eat_path = (
            params.get("efficientat_config")
            or selection.efficientat_config_path
            or "configs/quality/efficientat_v1.yaml"
        )
        eat_cfg = load_efficientat_config(eat_path)
        # Allow pipeline params to override device / allow_download / checkpoint.
        for key in ("device", "allow_download", "checkpoint_path", "model_name"):
            if key in params:
                setattr(eat_cfg, key, params[key])

        ae_path = (
            params.get("acoustic_evidence_config")
            or selection.acoustic_evidence_config_path
            or "configs/quality/acoustic_evidence_v1.yaml"
        )
        ae_cfg = load_acoustic_evidence_config(ae_path)

        updated = [sample.model_copy(deep=True) for sample in samples]
        session: EfficientAtSession | None = None
        scored = 0
        reused = 0
        required = 0
        failed = 0

        for sample in updated:
            decision = needs_efficientat_v2_3(sample, selection)
            labels = dict(sample.labels or {})
            labels.setdefault("efficientat_v2_3_candidate", decision)
            sample.labels = labels
            quality = dict(sample.quality or {})

            if not decision["required"]:
                if str(quality.get("efficientat_status") or "") not in {
                    "skipped",
                    "success",
                    "failed",
                }:
                    quality.update(skipped_efficientat_fields(eat_cfg))
                event = resolve_audio_event_state(quality, cfg=ae_cfg)
                quality["audio_event_state"] = event
                quality["acoustic_state"] = aggregate_acoustic_state(quality, cfg=ae_cfg)
                sample.quality = quality
                continue

            required += 1
            if _has_reusable_scores(quality):
                reused += 1
            else:
                try:
                    audio_path = str(
                        sample.audio_path(str(eat_cfg.input_audio_key or "resampled_16k"))
                    )
                except Exception:  # noqa: BLE001
                    audio_path = str(getattr(sample, "source_path", "") or "")
                if session is None:
                    session = EfficientAtSession(eat_cfg)
                fields = session.analyze(audio_path)
                quality.update(fields)
                scored += 1
                if fields.get("efficientat_status") == "failed":
                    failed += 1

            event = resolve_audio_event_state(quality, cfg=ae_cfg)
            quality["audio_event_state"] = event
            quality["acoustic_state"] = aggregate_acoustic_state(quality, cfg=ae_cfg)
            sample.quality = quality
            sample.add_lineage(
                self.full_name,
                self.version,
                {
                    "trigger_version": TRIGGER_VERSION_V23,
                    "required": True,
                    "reasons": decision["reasons"],
                    "efficientat_status": quality.get("efficientat_status"),
                    "model_digest": quality.get("efficientat_model_digest"),
                    "audio_sha256": original_audio_sha256(sample),
                },
            )

        if updated:
            updated[0].labels["efficientat_v2_3_batch"] = {
                "required": required,
                "scored": scored,
                "reused": reused,
                "failed": failed,
                "total": len(updated),
            }
        return updated
