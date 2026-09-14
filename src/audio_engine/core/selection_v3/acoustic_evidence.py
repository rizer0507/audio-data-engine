"""Consume existing acoustic sidecar fields. Missing detectors stay unknown.

Low DNSMOS, a lone ``crosstalk_suspected`` boolean, or all-empty ASR text
cannot confirm environment noise or crosstalk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AcousticEvidence:
    state: str
    no_target_speech: bool | None = None
    overlap_confirmed: bool = False
    human_crosstalk_confirmed: bool = False
    background_only: bool = False
    vad_calibrated: bool = False
    vad_speech_present: bool | None = None
    sources: list[str] = field(default_factory=list)
    versions: dict[str, str] = field(default_factory=dict)
    gaps: list[str] = field(default_factory=list)
    scores: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "no_target_speech": self.no_target_speech,
            "overlap_confirmed": self.overlap_confirmed,
            "human_crosstalk_confirmed": self.human_crosstalk_confirmed,
            "background_only": self.background_only,
            "vad_calibrated": self.vad_calibrated,
            "vad_speech_present": self.vad_speech_present,
            "sources": list(self.sources),
            "versions": dict(self.versions),
            "gaps": list(self.gaps),
            "scores": dict(self.scores),
        }


def _opt_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def collect_acoustic_evidence(quality: dict[str, Any] | None, labels: dict[str, Any] | None) -> AcousticEvidence:
    q = quality if isinstance(quality, dict) else {}
    labels = labels if isinstance(labels, dict) else {}
    gaps: list[str] = []
    sources: list[str] = []
    versions: dict[str, str] = {}

    no_target = _opt_bool(q.get("no_target_speech"))
    if no_target is None:
        no_target = _opt_bool(labels.get("no_target_speech"))
    trusted_absent = _opt_bool(q.get("no_target_speech_trusted"))
    if trusted_absent is None:
        trusted_absent = _opt_bool(labels.get("no_target_speech_trusted"))
    if no_target is True and trusted_absent is True:
        sources.append("no_target_speech")
        if q.get("no_target_speech_version") or labels.get("no_target_speech_version"):
            versions["no_target_speech"] = str(
                q.get("no_target_speech_version") or labels.get("no_target_speech_version")
            )
    elif no_target is True and trusted_absent is not True:
        gaps.append("no_target_speech_untrusted")
        no_target = None
    else:
        gaps.append("no_target_speech_missing")

    overlap = _opt_bool(q.get("overlap_detected"))
    if overlap is None:
        overlap = _opt_bool(labels.get("overlap_detected"))
    overlap_trusted = _opt_bool(q.get("overlap_detected_trusted"))
    if overlap_trusted is None:
        overlap_trusted = _opt_bool(labels.get("overlap_detected_trusted"))
    overlap_confirmed = overlap is True and overlap_trusted is True
    if overlap_confirmed:
        sources.append("overlap_detected")
        if q.get("overlap_detector_version"):
            versions["overlap"] = str(q.get("overlap_detector_version"))
    elif overlap is True:
        gaps.append("overlap_untrusted")
    else:
        gaps.append("overlap_detector_missing")

    human = _opt_bool(q.get("human_crosstalk_confirmed"))
    if human is None:
        human = _opt_bool(labels.get("human_crosstalk_confirmed"))
    if human is True:
        sources.append("human_crosstalk_confirmed")

    # A lone boolean cannot confirm.
    suspected = _opt_bool(q.get("crosstalk_suspected"))
    if suspected is True and not overlap_confirmed and human is not True:
        gaps.append("crosstalk_suspected_not_sufficient")

    background = _opt_bool(q.get("background_only"))
    if background is None:
        background = _opt_bool(labels.get("background_only"))
    background_label = str(q.get("background_label") or labels.get("background_label") or "")
    background_only = background is True or background_label in {"music", "noise", "background"}

    vad_calibrated = _opt_bool(q.get("vad_calibrated")) is True or _opt_bool(
        labels.get("vad_calibrated")
    ) is True
    vad_present = _opt_bool(q.get("vad_speech_present"))
    if vad_present is None:
        vad_present = _opt_bool(labels.get("vad_speech_present"))
    if not vad_calibrated:
        gaps.append("vad_uncalibrated")
        vad_present = None

    calibrated = _opt_bool(q.get("calibrated"))
    diagnosis_status = str(q.get("noise_diagnosis_status") or q.get("dnsmos_status") or "")
    not_required = diagnosis_status == "not_required"
    scores = {
        "dnsmos_ovrl": None if not_required else q.get("dnsmos_ovrl"),
        "noise_band": None if not_required else q.get("noise_band"),
        "noise_risk": None if not_required else q.get("noise_risk"),
        "dnsmos_status": "not_required" if not_required else q.get("dnsmos_status"),
        "calibrated": False if not_required else calibrated,
    }
    if not_required:
        gaps.append("noise_diagnosis_not_required")
    elif calibrated is not True:
        gaps.append("dnsmos_uncalibrated")

    if no_target is True and trusted_absent is True:
        state = "environment_confirmed"
    elif overlap_confirmed or human is True:
        state = "crosstalk_confirmed"
    elif background_only:
        state = "background_only"
    else:
        state = "unknown"

    return AcousticEvidence(
        state=state,
        no_target_speech=no_target if trusted_absent is True else None,
        overlap_confirmed=overlap_confirmed,
        human_crosstalk_confirmed=human is True,
        background_only=background_only,
        vad_calibrated=vad_calibrated,
        vad_speech_present=vad_present,
        sources=sources,
        versions=versions,
        gaps=gaps,
        scores=scores,
    )
