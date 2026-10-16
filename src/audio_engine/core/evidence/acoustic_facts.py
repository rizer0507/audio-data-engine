"""Acoustic / voicemail fact helpers with explicit parameters (044).

Extracted from ``selection_v3.five_class_v2_4_on_demand_acoustic``. Nothing here takes a
selection config: thresholds (for example the SIG weak threshold) are passed in
explicitly so each business mode owns its own number.

``not_required`` in stored quality fields is a *consumer scheduling marker*. These
helpers surface it as "no valid evidence" (``None`` / explicit reason); they never
interpret it as a computed value.
"""

from __future__ import annotations

import re
from typing import Any

from audio_engine.core.evidence.family_routes import RouteView
from audio_engine.core.evidence.types import (
    ACOUSTIC_STATUS_NOT_REQUIRED,
    RUN_STATUS_FAILED,
    RUN_STATUS_MISSING,
    RUN_STATUS_SUCCESS_EMPTY,
    RUN_STATUS_SUCCESS_TEXT,
)


def family_dual_run_success(routes: list[RouteView], family: str, expected: int) -> bool:
    """Both (expected) runs succeeded (text or empty); no failed/missing."""
    members = [r for r in routes if r.family == family]
    if len(members) < expected:
        return False
    if any(r.status in {RUN_STATUS_FAILED, RUN_STATUS_MISSING} for r in members):
        return False
    success = [
        r
        for r in members
        if r.status in {RUN_STATUS_SUCCESS_TEXT, RUN_STATUS_SUCCESS_EMPTY}
    ]
    return len(success) >= expected


def voicemail_hit_raw(
    routes: list[RouteView],
    pattern: re.Pattern[str] | None,
) -> dict[str, Any] | None:
    """Match voicemail patterns against raw ASR text (pre-clean)."""
    if pattern is None:
        return None
    for route in routes:
        if route.status in {RUN_STATUS_FAILED, RUN_STATUS_MISSING}:
            continue
        text = str(route.raw_text or "")
        if not text.strip():
            continue
        matched = pattern.search(text)
        if matched:
            return {
                "run_id": route.run_id,
                "family": route.family,
                "field": "raw_text",
                "snippet": text[:120],
                "match": matched.group(0),
            }
    return None


def sig_is_low(
    quality: dict[str, Any],
    weak_sig: float | None,
) -> tuple[bool | None, str]:
    """Return (is_low, reason). ``None`` means unavailable; must not guess.

    ``weak_sig`` is the caller's business threshold; model scoring never owns it.
    """
    status = str(quality.get("dnsmos_status") or "").strip().lower()
    if status == ACOUSTIC_STATUS_NOT_REQUIRED or status == "not_required":
        return None, "dnsmos_not_required"
    if status != "success":
        return None, f"dnsmos_status_{status or 'missing'}"
    sig = quality.get("dnsmos_sig")
    if sig is None:
        return None, "sig_missing"
    try:
        sig_f = float(sig)
    except (TypeError, ValueError):
        return None, "sig_invalid"
    if weak_sig is None:
        return None, "weak_sig_unconfigured"
    try:
        weak_f = float(weak_sig)
    except (TypeError, ValueError):
        return None, "weak_sig_invalid"
    return sig_f <= weak_f, "ok"


def vad_state(quality: dict[str, Any]) -> str:
    if str(quality.get("vad_status") or "").strip().lower() == "failed":
        return "failed"
    if str(quality.get("vad_status") or "").strip().lower() == ACOUSTIC_STATUS_NOT_REQUIRED:
        return ACOUSTIC_STATUS_NOT_REQUIRED
    return str(quality.get("vad_speech_state") or "").strip().lower()


def strict_zero_flag(quality: dict[str, Any]) -> tuple[bool | None, str]:
    """None = missing/failed check; True = all channels strict zero."""
    status = str(
        quality.get("energy_status") or quality.get("energy_compute_status") or ""
    ).strip().lower()
    if status == ACOUSTIC_STATUS_NOT_REQUIRED:
        return None, "energy_not_required"
    flag = quality.get("strict_all_channels_zero")
    if flag is None:
        energy_state = str(quality.get("energy_state") or "").strip().lower()
        if energy_state == "failed" or status == "failed":
            return None, "energy_failed"
        return None, "strict_zero_missing"
    return bool(flag), "ok"
