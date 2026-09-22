"""Aggregate energy + VAD + EfficientAT into acoustic_state / decision helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from audio_engine.core.quality.efficientat import derive_audio_event_state


@dataclass
class AcousticEvidenceConfig:
    policy_version: str = "acoustic_evidence_v1"
    skipped_event_policy: str = "vad_energy_fallback"
    strong_veto_states: list[str] = field(
        default_factory=lambda: ["silence", "non_speech_noise"]
    )
    dnsmos_quality_tags: dict[str, dict[str, float]] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> AcousticEvidenceConfig:
        data = dict(raw or {})
        tags = data.get("dnsmos_quality_tags") or {}
        veto = data.get("strong_veto_states") or ["silence", "non_speech_noise"]
        return cls(
            policy_version=str(data.get("policy_version") or "acoustic_evidence_v1"),
            skipped_event_policy=str(
                data.get("skipped_event_policy") or "vad_energy_fallback"
            ),
            strong_veto_states=[str(x) for x in veto],
            dnsmos_quality_tags={
                str(k): {str(a): float(b) for a, b in dict(v or {}).items()}
                for k, v in dict(tags).items()
            },
        )


def load_acoustic_evidence_config(path: str | Path | None) -> AcousticEvidenceConfig:
    if not path:
        return AcousticEvidenceConfig()
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"acoustic_evidence config must be a mapping: {path}")
    return AcousticEvidenceConfig.from_mapping(raw)


def resolve_audio_event_state(
    quality: dict[str, Any],
    *,
    cfg: AcousticEvidenceConfig | None = None,
) -> str:
    """Prefer EfficientAT-derived state; fall back when skipped."""
    cfg = cfg or AcousticEvidenceConfig()
    status = str(quality.get("efficientat_status") or "").strip().lower()
    explicit = str(quality.get("audio_event_state") or "").strip()
    if status == "success" and explicit:
        return explicit
    if status == "failed":
        return "failed"
    if status == "skipped" or not status:
        if cfg.skipped_event_policy == "unknown":
            return "unknown"
        vad = str(quality.get("vad_speech_state") or "").strip().lower()
        energy = str(quality.get("energy_state") or "").strip().lower()
        if vad == "present":
            return "speech_like"
        if vad == "absent" and energy in {"audible", "borderline"}:
            return "non_speech_environment"
        if vad == "absent" and energy == "inaudible":
            return "unknown"
        if vad == "failed":
            return "failed"
        return "unknown"
    if explicit:
        return explicit
    return derive_audio_event_state(
        {
            "speech_prob": quality.get("speech_prob"),
            "conversation_prob": quality.get("conversation_prob"),
            "speech_babble_prob": quality.get("speech_babble_prob"),
            "environment_noise_prob": quality.get("environment_noise_prob"),
            "music_prob": quality.get("music_prob"),
            "vehicle_prob": quality.get("vehicle_prob"),
            "mechanical_prob": quality.get("mechanical_prob"),
        },
        status=status or "unknown",
    )


def aggregate_acoustic_state(
    quality: dict[str, Any],
    *,
    cfg: AcousticEvidenceConfig | None = None,
) -> str:
    """Map energy + VAD + audio_event_state → acoustic_state."""
    cfg = cfg or AcousticEvidenceConfig()
    energy = str(quality.get("energy_state") or "").strip().lower()
    vad = str(quality.get("vad_speech_state") or "").strip().lower()
    event = resolve_audio_event_state(quality, cfg=cfg)

    if energy == "failed" or vad == "failed":
        return "failed"
    if event == "failed" and str(quality.get("efficientat_status") or "") == "failed":
        return "uncertain"

    if energy == "inaudible" and vad == "absent":
        return "silence"

    if vad == "absent" and event == "non_speech_environment":
        return "non_speech_noise"

    if vad == "present" and event == "human_background":
        return "human_noise_suspected"

    if vad == "present" and event != "human_background":
        return "speech_supported"

    if vad == "uncertain" or event in {"unknown", "mixed"}:
        return "uncertain"
    if vad == "present" and event == "non_speech_environment":
        return "uncertain"
    if vad == "absent" and event in {"speech_like", "human_background", "mixed"}:
        return "uncertain"
    return "uncertain"


def optional_dnsmos_quality_tags(
    quality: dict[str, Any],
    *,
    cfg: AcousticEvidenceConfig | None = None,
) -> list[str]:
    """DNSMOS may only emit quality tags — never category evidence."""
    cfg = cfg or AcousticEvidenceConfig()
    status = str(quality.get("dnsmos_status") or "").strip().lower()
    if status != "success":
        return []
    tags: list[str] = []
    bak = quality.get("dnsmos_bak")
    ovrl = quality.get("dnsmos_ovrl")
    sig = quality.get("dnsmos_sig")
    noisy = cfg.dnsmos_quality_tags.get("perceptual_noisy") or {}
    poor = cfg.dnsmos_quality_tags.get("poor_speech_quality") or {}
    if bak is not None and ovrl is not None and noisy:
        if float(bak) <= float(noisy.get("bak_max", 3.0)) or float(ovrl) <= float(
            noisy.get("ovrl_max", 2.8)
        ):
            tags.append("perceptual_noisy")
    if sig is not None and poor:
        if float(sig) <= float(poor.get("sig_max", 2.8)):
            tags.append("poor_speech_quality")
    return tags
