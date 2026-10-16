"""Text helpers for selection_v3: raw vs comparison_text, similarity.

044: truth lives in ``audio_engine.core.evidence.text``; this module re-exports
for compatibility.
"""

from __future__ import annotations

from audio_engine.core.evidence.text import (  # noqa: F401
    _levenshtein,
    comparison_text,
    pairwise_min_similarity,
    raw_transcript_text,
    text_similarity,
    transcript_text,
)

__all__ = [
    "comparison_text",
    "pairwise_min_similarity",
    "raw_transcript_text",
    "text_similarity",
    "transcript_text",
]
