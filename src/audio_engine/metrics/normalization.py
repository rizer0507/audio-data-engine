from __future__ import annotations

import re
import unicodedata
from typing import Any

_CONTROL_TAG = re.compile(r"<\|.*?\|>")
# Human annotation markers: drop 「【…】」 including inner text (before punctuation strip).
_ANNOTATION_BRACKET_RE = re.compile(r"【[^】]*】")


def normalize_text(value: Any, config: dict[str, Any] | None = None) -> str:
    """Normalize text according to a versioned profile (fillers are preserved)."""
    config = config or {}
    text = "" if value is None else str(value)
    text = _CONTROL_TAG.sub("", text)
    unicode_cfg = config.get("unicode", {})
    if unicode_cfg.get("normalize", True):
        text = unicodedata.normalize(unicode_cfg.get("form", "NFKC"), text)
    if config.get("annotation_brackets", {}).get("remove", False):
        text = _ANNOTATION_BRACKET_RE.sub("", text)
    if config.get("english", {}).get("lowercase", True):
        text = text.casefold()
    if config.get("punctuation", {}).get("remove", True):
        text = "".join(ch for ch in text if not unicodedata.category(ch).startswith("P"))
    if config.get("whitespace", {}).get("remove", True):
        text = "".join(text.split())
    fillers = config.get("filler", {})
    if fillers.get("remove", False):
        for filler in fillers.get("values", []):
            text = text.replace(str(filler), "")
    return text
