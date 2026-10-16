"""Neutral evidence layer (044): text, routes, family states, acoustic facts.

Must not import tagging or selection business classifiers / queues.
"""

from audio_engine.core.evidence.acoustic_facts import (
    family_dual_run_success,
    sig_is_low,
    strict_zero_flag,
    vad_state,
    voicemail_hit_raw,
)
from audio_engine.core.evidence.family_routes import RouteView, collect_route_views
from audio_engine.core.evidence.family_states import FamilyStateBundle, build_family_states
from audio_engine.core.evidence.text import comparison_text, raw_transcript_text, transcript_text

__all__ = [
    "RouteView",
    "collect_route_views",
    "FamilyStateBundle",
    "build_family_states",
    "comparison_text",
    "raw_transcript_text",
    "transcript_text",
    "family_dual_run_success",
    "voicemail_hit_raw",
    "sig_is_low",
    "vad_state",
    "strict_zero_flag",
]
