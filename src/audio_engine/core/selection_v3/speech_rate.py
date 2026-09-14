"""Implausible speech-rate guard for selection_v3 (018).

Flags ASR transcripts whose comparison_text length vs audio duration cannot
be produced by normal human speech. Applied after ASR attach, in classify —
not in audio cleaning (no transcripts there).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from audio_engine.core.selection_v3.family_evidence import RouteView
from audio_engine.core.selection_v3.types import RUN_STATUS_SUCCESS_TEXT


@dataclass
class SpeechRateViolation:
    run_id: str
    family: str
    text_chars: int
    chars_per_sec: float


@dataclass
class SpeechRateAssessment:
    triggered: bool
    max_chars_per_sec_observed: float | None = None
    violations: list[SpeechRateViolation] = field(default_factory=list)

    @property
    def implausible_routes(self) -> list[str]:
        return [v.run_id for v in self.violations]


def assess_speech_rate(
    routes: list[RouteView],
    *,
    duration_sec: float | None,
    max_chars_per_sec: float,
    min_text_chars: int,
) -> SpeechRateAssessment:
    """Return whether any success_text route exceeds the configured rate.

    Empty / missing duration skips the check. Short texts below
    ``min_text_chars`` are ignored to avoid false positives on sub-second clips.
    """
    if duration_sec is None or duration_sec <= 0:
        return SpeechRateAssessment(triggered=False)
    if max_chars_per_sec <= 0:
        return SpeechRateAssessment(triggered=False)

    violations: list[SpeechRateViolation] = []
    observed_max: float | None = None
    for route in routes:
        if route.status != RUN_STATUS_SUCCESS_TEXT:
            continue
        text = route.comparison_text or ""
        n = len(text)
        if n < min_text_chars:
            continue
        cps = n / float(duration_sec)
        observed_max = cps if observed_max is None else max(observed_max, cps)
        if cps > max_chars_per_sec:
            violations.append(
                SpeechRateViolation(
                    run_id=route.run_id,
                    family=route.family,
                    text_chars=n,
                    chars_per_sec=round(cps, 3),
                )
            )
    return SpeechRateAssessment(
        triggered=bool(violations),
        max_chars_per_sec_observed=round(observed_max, 3) if observed_max is not None else None,
        violations=violations,
    )
