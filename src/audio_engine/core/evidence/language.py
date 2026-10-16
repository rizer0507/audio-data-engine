"""Neutral speech-language check of a transcript body (zh_speech_v1).

Extracted from ``selection_v3.noise_trigger`` so the shared classify-text pre-layer
does not depend on a selection business module. ``noise_trigger`` re-imports these.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from audio_engine.core.evidence.text import transcript_text

LANGUAGE_POLICY_VERSION = "zh_speech_v1"

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_LATIN_RE = re.compile(r"[A-Za-z]+")
_PUNCT_RE = re.compile(r"[\s\W_]+", re.UNICODE)
_ABBREV = frozenset(
    {
        "AI",
        "APP",
        "ATM",
        "CPU",
        "GPS",
        "ID",
        "LCD",
        "LED",
        "NFC",
        "PDF",
        "POS",
        "QR",
        "SIM",
        "SMS",
        "TV",
        "URL",
        "USB",
        "VIP",
        "WIFI",
    }
)


def usable_body(value: Any) -> str:
    """Transcript body after control-tag strip. Punctuation-only is empty.

    Short fillers such as 嗯/啊/好 remain. They are real speech, not empty.
    """
    body = transcript_text(value)
    if not body:
        return ""
    compact = _PUNCT_RE.sub("", body)
    if not compact:
        return ""
    return body.strip()


def _is_abbreviation(token: str, *, has_cjk: bool) -> bool:
    if len(token) <= 1:
        return True
    if not has_cjk:
        return False
    upper = token.upper()
    if upper in _ABBREV:
        return True
    if token.isupper() and 2 <= len(token) <= 6:
        return True
    if 2 <= len(token) <= 8 and any(ch.isupper() for ch in token) and any(ch.islower() for ch in token):
        return True
    return False


def assess_speech_language(value: Any) -> dict[str, Any]:
    """Versioned speech-language check of the actual body, not the model tag.

    Returns ``zh`` / ``non_zh`` / ``mixed`` / ``unknown`` / ``empty``.
    Unknown does not add a scoring trigger. Traditional Chinese is Chinese.
    """
    body = usable_body(value)
    if not body:
        return {
            "label": "empty",
            "evidence": "empty_after_tag_strip",
            "spans": [],
            "policy": LANGUAGE_POLICY_VERSION,
        }
    normalized = unicodedata.normalize("NFKC", body)
    cjk = _CJK_RE.findall(normalized)
    tokens = _LATIN_RE.findall(normalized)
    has_cjk = bool(cjk)
    foreign = [tok for tok in tokens if not _is_abbreviation(tok, has_cjk=has_cjk)]
    # A lone 1-2 letter token without Chinese is too short to call foreign speech.
    if not has_cjk:
        foreign = [tok for tok in foreign if len(tok) >= 3]
    if has_cjk and not foreign:
        label = "zh"
        evidence = "cjk_without_foreign_speech"
    elif has_cjk and foreign:
        label = "mixed"
        evidence = "foreign_fragment_in_chinese"
    elif foreign:
        label = "non_zh"
        evidence = "foreign_speech"
    else:
        label = "unknown"
        evidence = "no_reliable_speech_script"
    return {
        "label": label,
        "evidence": evidence,
        "spans": foreign[:8],
        "policy": LANGUAGE_POLICY_VERSION,
        "snippet": body[:80],
    }
