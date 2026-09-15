"""Strict semantic-risk predicates for selection_five_class_v1 (027 §7.3).

Old risk tags (false_affirmation / filler / rejection_sanitization / broad
negation_flip) never assign the main class. Priority: hallucinated_assertion
> short_polarity_ambiguity > semantic_reversal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from audio_engine.core.selection_v3.classify_text import is_cjk_ideograph
from audio_engine.core.selection_v3.semantic_risk import LexiconPatterns, polarity_of_text
from audio_engine.core.selection_v3.text_tolerance import apply_tolerance_key, to_simplified
from audio_engine.core.selection_v3.types import (
    POLARITY_NEGATIVE,
    POLARITY_POSITIVE,
    SUBTYPE_HALLUCINATED_ASSERTION,
    SUBTYPE_SEMANTIC_REVERSAL,
    SUBTYPE_SHORT_POLARITY_AMBIGUITY,
)

# Complete polarity response templates (027 §7.3 ③).
_RESPONSE_TEMPLATES = (
    "不需要",
    "需要",
    "不要",
    "要",
    "不用",
    "用",
    "不可以",
    "可以",
    "不是",
    "是",
    "没有",
    "有",
    "好的",
    "好",
)

_NEGATION_MARKERS = ("不", "没", "無", "无", "别", "別", "未")
_UNKNOWN_FILLERS = frozenset({"不知道", "不清楚", "嗯", "啊", "哦", "呃", "额", "唔", "喂", "谁啊"})
_QUESTION_MARKERS = ("吗", "嗎", "么", "麼", "呢", "？", "?")
_HEDGE_MARKERS = ("如果", "假如", "要是", "可能", "好像", "据说", "他说", "她说")

_OBJECT_SPLIT = re.compile(
    r"(贷款|保险|服务|信用卡|理财|产品|业务|套餐|优惠|活动)"
)


@dataclass
class FamilyPolarityView:
    family: str
    text: str
    polarity: str
    stable: bool
    is_empty: bool = False
    run_ids: list[str] = field(default_factory=list)


@dataclass
class StrictRiskHit:
    subtype: str
    evidence: dict[str, Any] = field(default_factory=dict)
    conflict_spans: list[dict[str, Any]] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)

    @property
    def formal(self) -> bool:
        return not self.missing_evidence


@dataclass
class StrictRiskAnalysis:
    hit: StrictRiskHit | None = None
    candidate_questions: list[str] = field(default_factory=list)
    wide_triggers: list[str] = field(default_factory=list)


def han_char_count(text: str) -> int:
    return sum(1 for ch in text if is_cjk_ideograph(ch))


def hits_response_template(text: str) -> bool:
    value = str(text or "").strip()
    if not value:
        return False
    simplified, _ = to_simplified(value)
    templates = sorted(_RESPONSE_TEMPLATES, key=len, reverse=True)
    for template in templates:
        if simplified == template:
            return True
    # Short utterances that end with a complete template (e.g. 我不需要).
    if han_char_count(simplified) <= 6:
        for template in templates:
            if simplified.endswith(template) and len(template) >= 1:
                return True
    return False


def _normalize_proposition(text: str) -> str:
    simplified, _ = to_simplified(str(text or ""))
    key = apply_tolerance_key(simplified)
    for marker in _NEGATION_MARKERS:
        key = key.replace(marker, "")
    return key


def _objects(text: str) -> frozenset[str]:
    return frozenset(_OBJECT_SPLIT.findall(str(text or "")))


def _is_hedged_or_question(text: str) -> bool:
    value = str(text or "")
    if any(m in value for m in _QUESTION_MARKERS):
        return True
    if any(m in value for m in _HEDGE_MARKERS):
        return True
    if "不是不" in value or "不能不" in value:
        return True
    return False


def _is_unknown_or_filler(text: str) -> bool:
    simplified, _ = to_simplified(str(text or "").strip())
    return simplified in _UNKNOWN_FILLERS or not simplified


def align_polarity_pair(
    left: FamilyPolarityView,
    right: FamilyPolarityView,
) -> dict[str, Any] | None:
    """Return alignment evidence when same proposition, opposite polarity."""
    if left.family == right.family:
        return None
    if not left.stable or not right.stable:
        return None
    if left.is_empty or right.is_empty:
        return None
    if left.polarity not in {POLARITY_POSITIVE, POLARITY_NEGATIVE}:
        return None
    if right.polarity not in {POLARITY_POSITIVE, POLARITY_NEGATIVE}:
        return None
    if left.polarity == right.polarity:
        return None
    if _is_hedged_or_question(left.text) or _is_hedged_or_question(right.text):
        return None
    if _is_unknown_or_filler(left.text) or _is_unknown_or_filler(right.text):
        return None
    left_obj = _objects(left.text)
    right_obj = _objects(right.text)
    if left_obj and right_obj and left_obj.isdisjoint(right_obj):
        return None
    left_prop = _normalize_proposition(left.text)
    right_prop = _normalize_proposition(right.text)
    if not left_prop or not right_prop:
        return None
    if left_prop != right_prop:
        # Allow minor particle differences already stripped by tolerance key.
        if abs(len(left_prop) - len(right_prop)) > 1:
            return None
        # Require shared core length for short responses.
        shared = sum(1 for a, b in zip(left_prop, right_prop) if a == b)
        if shared < max(1, min(len(left_prop), len(right_prop)) - 1):
            return None
        if left_prop != right_prop and not (
            left_prop in right_prop or right_prop in left_prop
        ):
            return None
    return {
        "left_family": left.family,
        "right_family": right.family,
        "left_text": left.text,
        "right_text": right.text,
        "left_polarity": left.polarity,
        "right_polarity": right.polarity,
        "proposition": left_prop,
    }


def evaluate_semantic_reversal(
    families: Iterable[FamilyPolarityView],
) -> StrictRiskHit | None:
    views = [v for v in families if v.stable and not v.is_empty]
    for i, left in enumerate(views):
        for right in views[i + 1 :]:
            aligned = align_polarity_pair(left, right)
            if aligned:
                return StrictRiskHit(
                    subtype=SUBTYPE_SEMANTIC_REVERSAL,
                    evidence=aligned,
                    conflict_spans=[
                        {
                            "family": aligned["left_family"],
                            "text": aligned["left_text"],
                            "polarity": aligned["left_polarity"],
                        },
                        {
                            "family": aligned["right_family"],
                            "text": aligned["right_text"],
                            "polarity": aligned["right_polarity"],
                        },
                    ],
                )
    return None


def evaluate_hallucinated_assertion(
    families: Iterable[FamilyPolarityView],
    *,
    acoustic: dict[str, Any] | None = None,
    human_verified_no_target: bool = False,
) -> StrictRiskHit | None:
    """② empty/non-speech family vs polarity family, with audio confirmation."""
    acoustic = acoustic or {}
    # Real short response missed by an empty family is never ②.
    if acoustic.get("short_response_missed") is True:
        return None
    if acoustic.get("target_speech_present") is True and not human_verified_no_target:
        return None
    empty_stable = [
        v for v in families if v.stable and v.is_empty
    ]
    polarity_stable = [
        v
        for v in families
        if v.stable
        and not v.is_empty
        and v.polarity in {POLARITY_POSITIVE, POLARITY_NEGATIVE}
        and not _is_unknown_or_filler(v.text)
        and not _is_hedged_or_question(v.text)
    ]
    if not empty_stable or not polarity_stable:
        return None
    # Cross-family only.
    if all(e.family == p.family for e in empty_stable for p in polarity_stable):
        return None
    confirmed = bool(
        human_verified_no_target
        or acoustic.get("state") in {"environment_confirmed", "crosstalk_confirmed"}
        or (
            acoustic.get("no_target_speech") is True
            and acoustic.get("state") != "unknown"
            and not acoustic.get("gaps")
        )
    )
    evidence = {
        "empty_families": [v.family for v in empty_stable],
        "polarity_families": [
            {"family": v.family, "text": v.text, "polarity": v.polarity}
            for v in polarity_stable
        ],
        "acoustic_state": acoustic.get("state"),
    }
    if not confirmed:
        return StrictRiskHit(
            subtype=SUBTYPE_HALLUCINATED_ASSERTION,
            evidence=evidence,
            missing_evidence=["audio_or_human_no_target_speech"],
        )
    return StrictRiskHit(
        subtype=SUBTYPE_HALLUCINATED_ASSERTION,
        evidence=evidence,
        conflict_spans=[
            {"family": v.family, "text": v.text, "role": "polarity"}
            for v in polarity_stable
        ]
        + [{"family": v.family, "text": "", "role": "empty"} for v in empty_stable],
    )


def evaluate_short_polarity_ambiguity(
    families: Iterable[FamilyPolarityView],
    *,
    max_han_chars: int = 4,
    acoustic: dict[str, Any] | None = None,
    human_unintelligible_polarity: bool = False,
) -> StrictRiskHit | None:
    """③ max representative Han length ≤4, polarity conflict, audio unintelligible."""
    reps = [v for v in families if v.stable and not v.is_empty]
    if not reps:
        return None
    max_len = max(han_char_count(v.text) for v in reps)
    if max_len > max_han_chars:
        return None
    if not any(hits_response_template(v.text) for v in reps):
        return None
    polarities = {v.polarity for v in reps if v.polarity in {POLARITY_POSITIVE, POLARITY_NEGATIVE}}
    empty_stable = [v for v in families if v.stable and v.is_empty]
    conflict = len(polarities) >= 2 or (
        bool(polarities) and bool(empty_stable) and any(v.family != e.family for v in reps for e in empty_stable)
    )
    if not conflict:
        return None
    # All-family identical short answer is not ③.
    texts = {v.text for v in reps}
    if len(texts) == 1 and not empty_stable:
        return None
    acoustic = acoustic or {}
    confirmed = bool(
        human_unintelligible_polarity
        or acoustic.get("polarity_syllable_unintelligible") is True
    )
    evidence = {
        "max_han_chars": max_len,
        "representatives": [{"family": v.family, "text": v.text, "polarity": v.polarity} for v in reps],
        "empty_families": [v.family for v in empty_stable],
    }
    if not confirmed:
        return StrictRiskHit(
            subtype=SUBTYPE_SHORT_POLARITY_AMBIGUITY,
            evidence=evidence,
            missing_evidence=["polarity_syllable_unintelligible"],
        )
    if acoustic.get("generally_unintelligible") is True and not human_unintelligible_polarity:
        # Ordinary "听不清" without polarity-syllable evidence is not enough.
        if acoustic.get("polarity_syllable_unintelligible") is not True:
            return None
    return StrictRiskHit(
        subtype=SUBTYPE_SHORT_POLARITY_AMBIGUITY,
        evidence=evidence,
        conflict_spans=[{"family": v.family, "text": v.text} for v in reps],
    )


def analyze_strict_risks(
    families: Iterable[FamilyPolarityView],
    *,
    patterns: LexiconPatterns,
    acoustic: dict[str, Any] | None = None,
    max_han_chars: int = 4,
    human_verified_no_target: bool = False,
    human_unintelligible_polarity: bool = False,
) -> StrictRiskAnalysis:
    views = list(families)
    # Ensure polarity filled.
    for view in views:
        if not view.polarity or view.polarity == "unknown":
            view.polarity = (
                "unknown"
                if view.is_empty
                else polarity_of_text(view.text, patterns)
            )

    hallu = evaluate_hallucinated_assertion(
        views,
        acoustic=acoustic,
        human_verified_no_target=human_verified_no_target,
    )
    short = evaluate_short_polarity_ambiguity(
        views,
        max_han_chars=max_han_chars,
        acoustic=acoustic,
        human_unintelligible_polarity=human_unintelligible_polarity,
    )
    reversal = evaluate_semantic_reversal(views)

    questions: list[str] = []
    wide: list[str] = []
    # Wide triggers only create annotation questions, never main class.
    nonempty = [v for v in views if not v.is_empty and v.text]
    if any(v.polarity == POLARITY_POSITIVE for v in nonempty) and any(
        v.polarity == POLARITY_NEGATIVE for v in nonempty
    ):
        wide.append("polarity_divergence_candidate")
    if hallu and hallu.missing_evidence:
        questions.append("是否存在可支持该肯否内容的目标语音？空转写是否漏识别？")
    if short and short.missing_evidence:
        questions.append("短应答中的关键否定音节是否可辨？请标出不可辨位置。")
    if reversal is None and wide:
        questions.append("跨族是否同一命题的正反冲突？请写出正确转写。")

    for candidate in (hallu, short, reversal):
        if candidate is None:
            continue
        if candidate.formal:
            return StrictRiskAnalysis(hit=candidate, candidate_questions=questions, wide_triggers=wide)
        # Informal (missing evidence) → keep questions, do not assign class yet.
        if candidate.missing_evidence:
            for gap in candidate.missing_evidence:
                questions.append(f"缺少证据: {gap}")

    return StrictRiskAnalysis(hit=None, candidate_questions=list(dict.fromkeys(questions)), wide_triggers=wide)
