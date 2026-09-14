"""Quality four-state and batch availability helpers for selection_v3 (020)."""

from __future__ import annotations

from typing import Any

from audio_engine.core.sample import Sample
from audio_engine.core.selection_v3.types import (
    DNSMOS_STATUS_FAILED,
    DNSMOS_STATUS_NOT_REQUIRED,
    DNSMOS_STATUS_SUCCESS,
    DNSMOS_STATUS_UNSUPPORTED,
    NOISE_BAND_CLEAN,
    NOISE_BAND_MODERATE,
    NOISE_BAND_NOISY,
    NOISE_BAND_UNKNOWN,
    QUALITY_STATE_FAILED,
    QUALITY_STATE_NOT_REQUIRED,
    QUALITY_STATE_SCORED_NOISY,
    QUALITY_STATE_SCORED_OK,
    QUALITY_STATE_UNCALIBRATED,
    QUALITY_STATE_UNSUPPORTED,
    RESERVATION_GOVERNANCE_HOLD,
)


def derive_quality_state(
    *,
    noise_band: str | None,
    noise_risk: bool | None,
    dnsmos_status: str | None,
    quality_calibrated: bool,
) -> str:
    """Map DNSMOS fields to an explicit four-state (plus scored_ok).

    Uncalibrated success scores are **not** treated as scored_noisy or scored_ok.
    """
    status = str(dnsmos_status or "").strip().lower() or None
    band = str(noise_band or "").strip().lower() or None

    # 023: not scored because the sample was not an ASR anomaly. Not uncalibrated.
    if status == DNSMOS_STATUS_NOT_REQUIRED:
        return QUALITY_STATE_NOT_REQUIRED
    if status == DNSMOS_STATUS_FAILED:
        return QUALITY_STATE_FAILED
    if status == DNSMOS_STATUS_UNSUPPORTED:
        return QUALITY_STATE_UNSUPPORTED
    if not quality_calibrated:
        # Scores may exist, but bands/risks are not release-grade judgments.
        return QUALITY_STATE_UNCALIBRATED
    if band == NOISE_BAND_NOISY or noise_risk is True:
        return QUALITY_STATE_SCORED_NOISY
    if (
        status == DNSMOS_STATUS_SUCCESS
        and band in {NOISE_BAND_CLEAN, NOISE_BAND_MODERATE}
        and noise_risk is False
    ):
        return QUALITY_STATE_SCORED_OK
    if band in {NOISE_BAND_UNKNOWN, None} or noise_risk is None or status is None:
        return QUALITY_STATE_UNCALIBRATED
    return QUALITY_STATE_UNCALIBRATED


def sample_quality_fields(sample: Sample) -> dict[str, Any]:
    q = sample.quality if isinstance(sample.quality, dict) else {}
    band = q.get("noise_band")
    risk = q.get("noise_risk")
    status = q.get("dnsmos_status")
    return {
        "noise_band": str(band) if band is not None else None,
        "noise_risk": risk if risk is None or isinstance(risk, bool) else None,
        "dnsmos_status": str(status) if status is not None else None,
    }


def sample_reservation_role(sample: Sample) -> str:
    return str(
        sample.labels.get("reservation_role")
        or sample.labels.get("dataset_role")
        or ""
    )


def is_governance_hold(sample: Sample) -> bool:
    role = sample_reservation_role(sample)
    if role == RESERVATION_GOVERNANCE_HOLD:
        return True
    flags = sample.labels.get("governance_flags") or []
    if isinstance(flags, str):
        flags = [x.strip() for x in flags.split(",") if x.strip()]
    return "missing_group_metadata" in {str(x) for x in flags}
