"""Strong voicemail templates. Weak words never assign automatic V."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from audio_engine.core.selection_v3.text_tolerance import transcript_text

DEFAULT_NEGATIONS = (
    "不想留言",
    "没有语音信箱",
    "沒有語音信箱",
    "不是语音信箱",
    "不是語音信箱",
    "别留言",
    "別留言",
)
DEFAULT_WEAK = (
    "我是机主",
    "总机",
    "人工服务",
    "你找他什么事",
    "您找他什么事",
    "正在转接",
)


@dataclass(frozen=True)
class StrongTemplate:
    pattern_id: str
    scene: str
    subtype: str
    pattern: re.Pattern[str]
    anchors: tuple[str, ...]
    canonical: str


@dataclass
class VoicemailHit:
    pattern_id: str
    scene: str
    subtype: str
    matched: str
    method: str
    similarity: float


@dataclass
class VoicemailLibrary:
    version: str
    strong: list[StrongTemplate] = field(default_factory=list)
    weak: tuple[str, ...] = DEFAULT_WEAK
    negations: tuple[str, ...] = DEFAULT_NEGATIONS
    min_similarity: float = 0.90


def _compile(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern)


def load_voicemail_library(path: str | Path | None) -> VoicemailLibrary:
    if not path:
        return VoicemailLibrary(version="empty")
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"voicemail library must be a mapping: {path}")
    strong: list[StrongTemplate] = []
    for item in raw.get("strong") or []:
        if not isinstance(item, dict):
            continue
        pattern = str(item.get("pattern") or "").strip()
        if not pattern:
            continue
        anchors = tuple(str(x) for x in (item.get("anchors") or []) if str(x))
        strong.append(
            StrongTemplate(
                pattern_id=str(item.get("id") or item.get("pattern_id") or "unnamed"),
                scene=str(item.get("scene") or item.get("id") or "unknown"),
                subtype=str(item.get("subtype") or "mailbox"),
                pattern=_compile(pattern),
                anchors=anchors,
                canonical=str(item.get("canonical") or ""),
            )
        )
    weak = tuple(str(x) for x in (raw.get("weak") or DEFAULT_WEAK) if str(x))
    negations = tuple(str(x) for x in (raw.get("negations") or DEFAULT_NEGATIONS) if str(x))
    return VoicemailLibrary(
        version=str(raw.get("version") or "voicemail_strong_v1"),
        strong=strong,
        weak=weak,
        negations=negations,
        min_similarity=float(raw.get("min_template_similarity") or 0.90),
    )


def _normalize_template_text(text: str) -> str:
    body = transcript_text(text)
    body = body.replace("您", "你").replace("嘀", "滴").replace("助手", "助理")
    return "".join(body.split())


def has_human_negation(text: str, library: VoicemailLibrary) -> bool:
    body = transcript_text(text)
    return any(token and token in body for token in library.negations)


def weak_only(text: str, library: VoicemailLibrary) -> bool:
    body = transcript_text(text)
    return any(token and token in body for token in library.weak)


def match_strong(text: str, library: VoicemailLibrary) -> VoicemailHit | None:
    if not text or not library.strong:
        return None
    if has_human_negation(text, library):
        return None
    normalized = _normalize_template_text(text)
    for template in library.strong:
        found = template.pattern.search(normalized) or template.pattern.search(text)
        if found:
            return VoicemailHit(
                pattern_id=template.pattern_id,
                scene=template.scene,
                subtype=template.subtype,
                matched=found.group(0),
                method="regex",
                similarity=1.0,
            )
        fuzzy = _fuzzy_anchor(normalized, template, library.min_similarity)
        if fuzzy is not None:
            return fuzzy
    return None


def _fuzzy_anchor(
    normalized: str,
    template: StrongTemplate,
    min_similarity: float,
) -> VoicemailHit | None:
    """At most one non-anchor character off, anchors still present, sim>=threshold."""
    if not template.canonical or not template.anchors:
        return None
    if not all(anchor in normalized for anchor in template.anchors):
        return None
    canonical = _normalize_template_text(template.canonical)
    from audio_engine.core.selection_v3.text_tolerance import symmetric_distance

    dist = symmetric_distance(normalized, canonical)
    if dist is None:
        return None
    similarity = 1.0 - dist
    if similarity < min_similarity:
        return None
    # One-character slack only outside the scene anchors.
    if _non_anchor_edits(normalized, canonical, template.anchors) > 1:
        return None
    return VoicemailHit(
        pattern_id=template.pattern_id,
        scene=template.scene,
        subtype=template.subtype,
        matched=normalized,
        method="fuzzy_anchor",
        similarity=round(similarity, 6),
    )


def _non_anchor_edits(left: str, right: str, anchors: tuple[str, ...]) -> int:
    from audio_engine.core.selection_v3.text import _levenshtein

    masked_left = left
    masked_right = right
    for anchor in anchors:
        masked_left = masked_left.replace(anchor, "")
        masked_right = masked_right.replace(anchor, "")
    return _levenshtein(masked_left, masked_right)


def agreed_scene(
    texts_by_family: dict[str, str],
    library: VoicemailLibrary,
) -> tuple[str, str, list[str], list[dict[str, Any]]] | None:
    """Same automatic-answer scene on at least two independent families."""
    grouped: dict[str, list[tuple[str, VoicemailHit]]] = {}
    for family in sorted(texts_by_family):
        hit = match_strong(texts_by_family[family], library)
        if hit is None:
            continue
        grouped.setdefault(hit.scene, []).append((family, hit))
    for scene in sorted(grouped):
        families = [family for family, _hit in grouped[scene]]
        unique = list(dict.fromkeys(families))
        if len(unique) < 2:
            continue
        subtype = grouped[scene][0][1].subtype
        evidence = [
            {
                "family": family,
                "pattern_id": hit.pattern_id,
                "scene": hit.scene,
                "subtype": hit.subtype,
                "matched": hit.matched,
                "method": hit.method,
            }
            for family, hit in grouped[scene]
        ]
        return scene, subtype, unique, evidence
    return None
