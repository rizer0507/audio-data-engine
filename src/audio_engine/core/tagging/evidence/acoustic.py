"""Acoustic evidence views (groups D/E) — read quality without forging values."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from audio_engine.core.selection_v3.config import SelectionV3Config
from audio_engine.core.selection_v3.five_class_v2_4_on_demand_acoustic import (
    _sig_is_low,
    _strict_zero_flag,
    _vad_state,
)
from audio_engine.core.selection_v3.types import ACOUSTIC_STATUS_NOT_REQUIRED
from audio_engine.core.tagging.types import EvidenceStatus


@dataclass
class AcousticEvidenceView:
    vad_status: str | None = None
    vad_speech_state: str | None = None
    energy_status: str | None = None
    strict_all_channels_zero: bool | None = None
    energy_state: str | None = None
    dnsmos_status: str | None = None
    dnsmos_sig: float | None = None
    dnsmos_bak: float | None = None
    dnsmos_ovrl: float | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def vad_compute_status(self) -> EvidenceStatus:
        s = str(self.vad_status or "").strip().lower()
        if not s or s in {"pending", "not_computed"}:
            return EvidenceStatus.NOT_COMPUTED
        if s == ACOUSTIC_STATUS_NOT_REQUIRED or s == "not_required":
            return EvidenceStatus.NOT_REQUIRED
        if s == "failed":
            return EvidenceStatus.FAILED
        if s == "missing":
            return EvidenceStatus.MISSING
        if s == "success":
            return EvidenceStatus.SUCCESS
        # Present state fields without explicit status still usable.
        if self.vad_speech_state:
            return EvidenceStatus.SUCCESS
        return EvidenceStatus.NOT_COMPUTED

    def energy_compute_status(self) -> EvidenceStatus:
        s = str(self.energy_status or "").strip().lower()
        if not s or s in {"pending", "not_computed"}:
            if self.strict_all_channels_zero is not None:
                return EvidenceStatus.SUCCESS
            return EvidenceStatus.NOT_COMPUTED
        if s == ACOUSTIC_STATUS_NOT_REQUIRED or s == "not_required":
            return EvidenceStatus.NOT_REQUIRED
        if s == "failed":
            return EvidenceStatus.FAILED
        if s == "missing":
            return EvidenceStatus.MISSING
        if s == "success":
            return EvidenceStatus.SUCCESS
        return EvidenceStatus.NOT_COMPUTED

    def dnsmos_compute_status(self) -> EvidenceStatus:
        s = str(self.dnsmos_status or "").strip().lower()
        if not s or s in {"pending", "not_computed"}:
            return EvidenceStatus.NOT_COMPUTED
        if s == ACOUSTIC_STATUS_NOT_REQUIRED or s == "not_required":
            return EvidenceStatus.NOT_REQUIRED
        if s == "failed":
            return EvidenceStatus.FAILED
        if s == "missing":
            return EvidenceStatus.MISSING
        if s == "success":
            return EvidenceStatus.SUCCESS
        return EvidenceStatus.NOT_COMPUTED


def acoustic_from_quality(quality: dict[str, Any] | None) -> AcousticEvidenceView:
    q = quality if isinstance(quality, dict) else {}
    sig = q.get("dnsmos_sig")
    try:
        sig_f = float(sig) if sig is not None else None
    except (TypeError, ValueError):
        sig_f = None
    bak = q.get("dnsmos_bak")
    ovrl = q.get("dnsmos_ovrl")
    try:
        bak_f = float(bak) if bak is not None else None
    except (TypeError, ValueError):
        bak_f = None
    try:
        ovrl_f = float(ovrl) if ovrl is not None else None
    except (TypeError, ValueError):
        ovrl_f = None
    return AcousticEvidenceView(
        vad_status=q.get("vad_status"),
        vad_speech_state=_vad_state(q) if q else None,
        energy_status=q.get("energy_status") or q.get("energy_compute_status"),
        strict_all_channels_zero=(
            bool(q["strict_all_channels_zero"])
            if q.get("strict_all_channels_zero") is not None
            else None
        ),
        energy_state=q.get("energy_state"),
        dnsmos_status=q.get("dnsmos_status"),
        dnsmos_sig=sig_f,
        dnsmos_bak=bak_f,
        dnsmos_ovrl=ovrl_f,
        details={"raw_vad_speech_state": q.get("vad_speech_state")},
    )


def sig_is_low(
    view: AcousticEvidenceView,
    selection: SelectionV3Config,
) -> tuple[bool | None, str]:
    quality = {
        "dnsmos_status": view.dnsmos_status,
        "dnsmos_sig": view.dnsmos_sig,
    }
    return _sig_is_low(quality, selection)


def strict_zero(
    view: AcousticEvidenceView,
) -> tuple[bool | None, str]:
    quality = {
        "energy_status": view.energy_status,
        "strict_all_channels_zero": view.strict_all_channels_zero,
        "energy_state": view.energy_state,
    }
    return _strict_zero_flag(quality)
