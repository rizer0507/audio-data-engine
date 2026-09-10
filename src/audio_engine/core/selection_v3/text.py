"""Text helpers for selection_v3: raw vs comparison_text, similarity."""

from __future__ import annotations

import unicodedata
from typing import Any

from audio_engine.core.transcript_reconcile import clean_control_tags


def raw_transcript_text(entry: Any) -> str:
    """Permanent raw text from a transcript entry (never mutated for voting)."""
    if entry is None:
        return ""
    if isinstance(entry, dict):
        extra = entry.get("extra") if isinstance(entry.get("extra"), dict) else {}
        raw = extra.get("raw_text")
        if raw is not None:
            return str(raw)
        text = entry.get("text")
        return "" if text is None else str(text)
    return str(entry)


def comparison_text(
    value: Any,
    *,
    punctuation_to_strip: str | list[str] | None = None,
) -> str:
    """Normalize only for comparison — keep negation / fillers / numbers / profanity.

    Allowed transforms: Unicode NFKC, full/half-width via NFKC, whitespace collapse,
    and configured punctuation stripping. Does not invent or clear critical content.
    """
    if value is None:
        return ""
    text = clean_control_tags(value) if not isinstance(value, str) else clean_control_tags(value)
    if not isinstance(text, str):
        text = str(text or "")
    text = unicodedata.normalize("NFKC", text)
    punct = punctuation_to_strip
    if punct is None:
        punct_chars = "，。！？、；：\"\"''（）【】《》…—·,.!?;:'\"()[]{}"
    elif isinstance(punct, list):
        punct_chars = "".join(str(x) for x in punct)
    else:
        punct_chars = str(punct)
    if punct_chars:
        table = str.maketrans({ch: "" for ch in punct_chars})
        text = text.translate(table)
    text = "".join(text.split())
    return text.strip()


def text_similarity(left: str, right: str) -> float:
    """``1 - levenshtein(a,b) / max(len(a), len(b))``; both empty → 1.0."""
    a = str(left or "")
    b = str(right or "")
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    dist = _levenshtein(a, b)
    return round(1.0 - dist / max(len(a), len(b)), 6)


def pairwise_min_similarity(texts: list[str]) -> float | None:
    nonempty = [str(t) for t in texts]
    if len(nonempty) < 2:
        return 1.0 if nonempty else None
    best = 1.0
    for i, left in enumerate(nonempty):
        for right in nonempty[i + 1 :]:
            best = min(best, text_similarity(left, right))
    return best


def _levenshtein(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    prev = list(range(len(right) + 1))
    for i, ch_l in enumerate(left, start=1):
        curr = [i]
        for j, ch_r in enumerate(right, start=1):
            ins = curr[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (0 if ch_l == ch_r else 1)
            curr.append(min(ins, delete, sub))
        prev = curr
    return prev[-1]
