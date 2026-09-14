"""Versioned text layers and safe tolerance for selection_v3 semantic-tolerant rule.

Three strings stay distinct:

- ``raw_text``: model original, audit only.
- ``transcript_text``: control tags removed, wording and punctuation kept.
- ``comparison_text`` / tolerant key: alignment only. Never written as the candidate.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from audio_engine.core.selection_v3.text import transcript_text
from audio_engine.core.selection_v3.types import TOLERANCE_VERSION

# One-to-one traditional → simplified. Ambiguous forms (乾/幹, 發) are omitted.
_T2S: dict[str, str] = {
    "現": "现",
    "麼": "么",
    "嗎": "吗",
    "為": "为",
    "這": "这",
    "個": "个",
    "後": "后",
    "對": "对",
    "說": "说",
    "話": "话",
    "請": "请",
    "問": "问",
    "時": "时",
    "間": "间",
    "電": "电",
    "無": "无",
    "關": "关",
    "機": "机",
    "國": "国",
    "學": "学",
    "開": "开",
    "門": "门",
    "長": "长",
    "聲": "声",
    "語": "语",
    "車": "车",
    "東": "东",
    "來": "来",
    "過": "过",
    "還": "还",
    "會": "会",
    "從": "从",
    "經": "经",
    "聽": "听",
    "見": "见",
    "點": "点",
    "線": "线",
    "歲": "岁",
    "號": "号",
    "裡": "里",
    "鐘": "钟",
    "戶": "户",
    "幫": "帮",
    "應": "应",
    "該": "该",
    "頭": "头",
    "務": "务",
    "聯": "联",
    "繫": "系",
    "撥": "拨",
    "暫": "暂",
    "員": "员",
    "與": "与",
    "於": "于",
    "臺": "台",
    "業": "业",
    "歡": "欢",
    "迎": "迎",
    "進": "进",
    "選": "选",
    "擇": "择",
    "確": "确",
    "認": "认",
    "實": "实",
    "際": "际",
    "價": "价",
    "錢": "钱",
    "萬": "万",
    "億": "亿",
    "親": "亲",
    "愛": "爱",
    "謝": "谢",
}

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_CONTROL_LEFTOVER = re.compile(r"<\|.*?\|>")

# Sentence-edge particles that do not change an answer. Never drop 吗/吧/好/行
# or a standalone 嗯/啊.
DEFAULT_EDGE_PARTICLES = ("啊", "呀", "呢")
FORBIDDEN_GLOBAL_DELETES = frozenset({"不", "没", "沒", "别", "別", "吗", "吧", "好", "行", "嗯", "啊"})


def to_simplified(text: str) -> tuple[str, bool]:
    """Safe one-to-one traditional conversion. Returns (converted, changed)."""
    changed = False
    chars: list[str] = []
    for ch in text:
        mapped = _T2S.get(ch)
        if mapped and mapped != ch:
            changed = True
            chars.append(mapped)
        else:
            chars.append(ch)
    return "".join(chars), changed


def detect_script(text: str) -> str:
    if any(ch in _T2S for ch in text):
        return "zh-Hant"
    if _CJK_RE.search(text):
        return "zh-Hans"
    return ""


_LEXICAL_RE = re.compile(
    r"[0-9A-Za-z\u00C0-\u024F\u0400-\u04FF\u0600-\u06FF\u3040-\u30FF\u3400-\u9FFF\uAC00-\uD7AF]"
)


def lexical_content(text: str) -> str:
    """Letters and digits only. Punctuation, spaces, and control tags are not content.

    Digit strings stay. This is not a candidate transcript and must not be written
    back as the selected body.
    """
    value = transcript_text(text)
    value = unicodedata.normalize("NFKC", value)
    return "".join(ch for ch in value if ch.isalnum() or _LEXICAL_RE.match(ch))


def has_lexical_content(text: str) -> bool:
    return bool(lexical_content(text))


def content_language(text: str) -> str:
    """Language of lexical content. Punctuation-only and tag-only output is ``empty``.

    Unlike :func:`detect_language`, a lone ``.`` is not ``unknown``. Digit strings
    and other writing systems stay content (usually ``unknown``), not empty.
    """
    lexical = lexical_content(text)
    if not lexical:
        return "empty"
    return detect_language(lexical)


def detect_language(text: str) -> str:
    """Language of the actual output, not the family name.

    Returns ``zh``, ``en``, ``mixed``, ``empty``, or ``unknown``.
    Mixed utterances are not whole-sentence Chinese.
    """
    value = transcript_text(text)
    if not value or not value.strip():
        return "empty"
    cjk = len(_CJK_RE.findall(value))
    latin = len(_LATIN_RE.findall(value))
    letters = cjk + latin
    if letters == 0:
        return "unknown"
    if cjk and latin and cjk / letters >= 0.2 and latin / letters >= 0.2:
        return "mixed"
    if cjk / letters >= 0.8:
        return "zh"
    if latin / letters >= 0.8:
        return "en"
    if cjk and not latin:
        return "zh"
    if latin and not cjk:
        return "en"
    return "unknown"


def languages_char_comparable(left: str, right: str) -> bool:
    """Whole-utterance character metrics only when both sides are Chinese."""
    if left == "zh" and right == "zh":
        return True
    return False


@dataclass(frozen=True)
class TextLayers:
    raw_text: str
    transcript_text: str
    comparison_text: str
    tolerant_key: str
    language: str
    script: str
    script_converted: bool
    char_comparable: bool
    tolerance_version: str = TOLERANCE_VERSION
    classify_text: str = ""
    empty_reason_codes: tuple[str, ...] = ()
    pre_filter_language: str = ""
    classify_text_policy: str = "legacy"


def _punct_chars(punctuation_to_strip: str | list[str] | None) -> str:
    if punctuation_to_strip is None:
        return "，。！？、；：\"\"''（）【】《》…—·,.!?;:'\"()[]{}—－-…"
    if isinstance(punctuation_to_strip, list):
        return "".join(str(x) for x in punctuation_to_strip)
    return str(punctuation_to_strip)


def comparison_form(
    value: str,
    *,
    punctuation_to_strip: str | list[str] | None = None,
) -> tuple[str, bool]:
    """NFKC, tag strip, punctuation/space collapse, safe t2s. No semantic rewrite."""
    text = transcript_text(value)
    text = unicodedata.normalize("NFKC", text)
    punct = _punct_chars(punctuation_to_strip)
    if punct:
        text = text.translate(str.maketrans({ch: "" for ch in punct}))
    text = "".join(text.split())
    converted, changed = to_simplified(text)
    return converted.strip(), changed


def apply_tolerance_key(
    comparison: str,
    *,
    edge_particles: tuple[str, ...] = DEFAULT_EDGE_PARTICLES,
    homophone_pairs: tuple[tuple[str, str], ...] = (),
) -> str:
    """Safe tolerance key. Does not delete negation or answer words."""
    text = comparison
    text = text.replace("您", "你")
    for src, dst in homophone_pairs:
        if src and dst:
            text = text.replace(src, dst)
    # Drop only a trailing non-answer particle, never the whole utterance.
    for particle in edge_particles:
        if particle in FORBIDDEN_GLOBAL_DELETES and particle != "啊":
            continue
        if particle in {"吗", "吧", "好", "行", "嗯"}:
            continue
        if text.endswith(particle) and len(text) > len(particle):
            text = text[: -len(particle)]
            break
    return text


def build_layers(
    raw: str,
    *,
    punctuation_to_strip: str | list[str] | None = None,
    edge_particles: tuple[str, ...] = DEFAULT_EDGE_PARTICLES,
    homophone_pairs: tuple[tuple[str, str], ...] = (),
    classify_text_policy: str = "legacy",
    echo=None,
    keep_digits: bool = True,
) -> TextLayers:
    from audio_engine.core.selection_v3.classify_text import (
        POLICY_LEGACY,
        prepare_classify_text,
        uses_chinese_only_text,
    )

    body = transcript_text(raw)
    if uses_chinese_only_text(classify_text_policy):
        prepared = prepare_classify_text(
            raw,
            echo=echo,
            keep_digits=keep_digits,
            policy=classify_text_policy,
        )
        body = prepared.transcript_text
        if prepared.is_empty:
            return TextLayers(
                raw_text=raw or "",
                transcript_text=body,
                comparison_text="",
                tolerant_key="",
                language="empty",
                script="",
                script_converted=False,
                char_comparable=False,
                classify_text="",
                empty_reason_codes=prepared.empty_reason_codes,
                pre_filter_language=prepared.pre_filter_language,
                classify_text_policy=classify_text_policy,
            )
        comparison, converted = to_simplified(prepared.classify_text)
        key = apply_tolerance_key(
            comparison,
            edge_particles=edge_particles,
            homophone_pairs=homophone_pairs,
        )
        script = detect_script(comparison)
        return TextLayers(
            raw_text=raw or "",
            transcript_text=body,
            comparison_text=comparison,
            tolerant_key=key,
            language="zh",
            script=script,
            script_converted=converted,
            char_comparable=True,
            classify_text=prepared.classify_text,
            empty_reason_codes=(),
            pre_filter_language=prepared.pre_filter_language,
            classify_text_policy=classify_text_policy,
        )
    comparison, converted = comparison_form(body, punctuation_to_strip=punctuation_to_strip)
    key = apply_tolerance_key(
        comparison,
        edge_particles=edge_particles,
        homophone_pairs=homophone_pairs,
    )
    language = detect_language(body)
    script = detect_script(body) if language == "zh" else ""
    return TextLayers(
        raw_text=raw or "",
        transcript_text=body,
        comparison_text=comparison,
        tolerant_key=key,
        language=language,
        script=script,
        script_converted=converted,
        char_comparable=language == "zh",
        classify_text=comparison,
        pre_filter_language=language,
        classify_text_policy=POLICY_LEGACY,
    )


def symmetric_distance(left: str, right: str) -> float | None:
    """``Levenshtein / max(len)``. Both empty is undefined for gold (None)."""
    from audio_engine.core.selection_v3.text import _levenshtein

    a = str(left or "")
    b = str(right or "")
    if not a and not b:
        return None
    if not a or not b:
        return 1.0
    dist = _levenshtein(a, b)
    return round(dist / max(len(a), len(b)), 6)


def tolerant_distance(left_key: str, right_key: str) -> float | None:
    return symmetric_distance(left_key, right_key)


def longer_side_chars(left_comparison: str, right_comparison: str) -> int:
    return max(len(left_comparison or ""), len(right_comparison or ""))


def is_short_pair(left_comparison: str, right_comparison: str, *, short_max_chars: int) -> bool:
    return longer_side_chars(left_comparison, right_comparison) <= short_max_chars


def contains_control_tag(text: str | None) -> bool:
    return bool(text) and bool(_CONTROL_LEFTOVER.search(text))
