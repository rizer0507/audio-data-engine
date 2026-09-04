"""Transcript-consensus selection engine (selection_v1.1).

Separates ``type`` (what the sample is), ``decision`` (how to handle it),
and ``label`` (final gold text). Uses model-family awareness and medoid
selection — never random picking.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from audio_engine.core.sample import Sample
from audio_engine.core.transcript_reconcile import (
    character_similarity,
    plain_transcript_text,
)

RULE_VERSION = "selection_v1.1"

DECISION_AUTO_ACCEPT = "auto_accept"
DECISION_AUTO_EMPTY = "auto_empty"
DECISION_MODEL_REVIEW = "model_review"
DECISION_MANUAL_REVIEW = "manual_review"

TYPE_NOISE = "noise"
TYPE_VOICEMAIL = "voicemail"
TYPE_SEMANTIC_INVERSION = "semantic_inversion"
TYPE_HALLUCINATION = "hallucination"
TYPE_QWEN_MISSING = "qwen_missing"
TYPE_AUTO_GOLD = "auto_gold"
TYPE_CONSENSUS_GOLD = "consensus_gold"
TYPE_HARDCASE = "hardcase"

SEMANTIC_POSITIVE = "positive"
SEMANTIC_NEGATIVE = "negative"
SEMANTIC_NONE = "none"
SEMANTIC_CONFLICT = "conflict"

DEFAULT_STRICT_THRESHOLD = 0.95
DEFAULT_CONSENSUS_THRESHOLD = 0.90
DEFAULT_DOMINANT_RATIO = 0.75
DEFAULT_EMPTY_RATIO = 0.75

DEFAULT_MODEL_FAMILIES: dict[str, list[str]] = {
    "qwen": ["qwen", "qwen1", "qwen2"],
    "sensevoice": ["sensevoice", "sensevoice1", "sensevoice2"],
    "doubao": ["doubao", "doubao1", "doubao2"],
    "kimi": ["kimi", "kimi1", "kimi2"],
}

DEFAULT_NEGATIVE_PHRASES = [
    "暂时不需要",
    "不考虑",
    "没需要",
    "不需要",
    "不愿意",
    "不可以",
    "不用",
    "不要",
]

DEFAULT_POSITIVE_PHRASES = [
    "有需要",
    "可以",
    "愿意",
    "考虑",
    "需要",
    "要",
]


@dataclass
class TranscriptView:
    model: str
    family: str
    text: str


@dataclass
class ClassificationResult:
    type: str
    decision: str
    label: str | None
    reason: str
    selected_model: str | None = None
    support_models: list[str] = field(default_factory=list)
    support_count: int = 0
    support_family_count: int = 0
    consensus_score: float | None = None
    min_similarity: float | None = None
    qwen_family_text: str | None = None
    sensevoice_family_text: str | None = None
    semantic_qwen: str | None = None
    semantic_sensevoice: str | None = None
    review_reason: str | None = None
    rule_version: str = RULE_VERSION

    def to_labels(self, policy_version: str) -> dict[str, Any]:
        labels: dict[str, Any] = {
            "classification_bucket": self.type,
            "type": self.type,
            "decision": self.decision,
            "classification_reason_codes": [self.reason],
            "selection_policy_version": policy_version,
            "rule_version": self.rule_version,
            "selected_model": self.selected_model or "",
            "support_models": list(self.support_models),
            "support_count": self.support_count,
            "support_family_count": self.support_family_count,
            "consensus_score": self.consensus_score,
            "min_similarity": self.min_similarity,
            "qwen_family_text": self.qwen_family_text or "",
            "sensevoice_family_text": self.sensevoice_family_text or "",
            "semantic_qwen": self.semantic_qwen or "",
            "semantic_sensevoice": self.semantic_sensevoice or "",
            "review_reason": self.review_reason or "",
            "annotation_revision": policy_version,
        }
        if self.decision == DECISION_AUTO_ACCEPT:
            text = str(self.label or "")
            labels.update(
                {
                    "annotation_state": "auto_accepted",
                    "gold_text": text,
                    "label": text,
                    "gold_source": self.selected_model or "",
                }
            )
        elif self.decision == DECISION_AUTO_EMPTY:
            labels.update(
                {
                    "annotation_state": "auto_accepted",
                    "gold_text": "",
                    "label": "",
                    "gold_source": self.selected_model or "",
                }
            )
        else:
            labels.setdefault("gold_text", "")
            labels.setdefault("label", self.label if self.label is not None else "")
        return labels


@dataclass
class SelectionConfig:
    strict_threshold: float = DEFAULT_STRICT_THRESHOLD
    consensus_threshold: float = DEFAULT_CONSENSUS_THRESHOLD
    dominant_ratio: float = DEFAULT_DOMINANT_RATIO
    empty_ratio_for_hallucination: float = DEFAULT_EMPTY_RATIO
    model_families: dict[str, list[str]] = field(
        default_factory=lambda: dict(DEFAULT_MODEL_FAMILIES)
    )
    negative_phrases: list[str] = field(default_factory=lambda: list(DEFAULT_NEGATIVE_PHRASES))
    positive_phrases: list[str] = field(default_factory=lambda: list(DEFAULT_POSITIVE_PHRASES))
    primary_family: str = "qwen"
    secondary_family: str = "sensevoice"
    rule_version: str = RULE_VERSION

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> SelectionConfig:
        families = params.get("model_families") or DEFAULT_MODEL_FAMILIES
        semantic = params.get("semantic") or {}
        return cls(
            strict_threshold=float(
                params.get("strict_threshold", DEFAULT_STRICT_THRESHOLD)
            ),
            consensus_threshold=float(
                params.get("consensus_threshold", DEFAULT_CONSENSUS_THRESHOLD)
            ),
            dominant_ratio=float(params.get("dominant_ratio", DEFAULT_DOMINANT_RATIO)),
            empty_ratio_for_hallucination=float(
                params.get("empty_ratio_for_hallucination", DEFAULT_EMPTY_RATIO)
            ),
            model_families={str(k): [str(x) for x in (v or [])] for k, v in families.items()},
            negative_phrases=[
                str(x)
                for x in (semantic.get("negative") or DEFAULT_NEGATIVE_PHRASES)
                if str(x).strip()
            ],
            positive_phrases=[
                str(x)
                for x in (semantic.get("positive") or DEFAULT_POSITIVE_PHRASES)
                if str(x).strip()
            ],
            primary_family=str(params.get("primary_family") or "qwen"),
            secondary_family=str(params.get("secondary_family") or "sensevoice"),
            rule_version=str(params.get("rule_version") or RULE_VERSION),
        )


def resolve_family(model: str, families: dict[str, list[str]]) -> str:
    key = str(model or "").strip().lower()
    if not key:
        return "unknown"
    alias_map: dict[str, str] = {}
    for family, aliases in families.items():
        alias_map[str(family).lower()] = str(family)
        for alias in aliases:
            alias_map[str(alias).lower()] = str(family)
    if key in alias_map:
        return alias_map[key]
    for alias, family in sorted(alias_map.items(), key=lambda item: len(item[0]), reverse=True):
        if key.startswith(alias):
            return family
    return key


def _compile_phrase_pattern(phrases: Iterable[str]) -> re.Pattern[str] | None:
    items = sorted({str(p).strip() for p in phrases if str(p).strip()}, key=len, reverse=True)
    if not items:
        return None
    return re.compile("|".join(f"(?:{re.escape(p)})" for p in items))


def semantic_class(
    text: str,
    *,
    negative_pattern: re.Pattern[str] | None,
    positive_pattern: re.Pattern[str] | None,
) -> str:
    value = str(text or "").strip()
    if not value:
        return SEMANTIC_NONE
    # Negative must win over positive ("不需要" contains "需要").
    if negative_pattern and negative_pattern.search(value):
        return SEMANTIC_NEGATIVE
    if positive_pattern and positive_pattern.search(value):
        return SEMANTIC_POSITIVE
    return SEMANTIC_NONE


def pairwise_min_similarity(texts: list[str]) -> float | None:
    if len(texts) < 2:
        return 1.0 if texts else None
    best = 1.0
    for i, left in enumerate(texts):
        for right in texts[i + 1 :]:
            best = min(best, character_similarity(left, right))
    return best


def medoid_index(texts: list[str]) -> int:
    """Return index of the transcript closest to all others (deterministic ties)."""
    if not texts:
        raise ValueError("medoid_index requires non-empty texts")
    if len(texts) == 1:
        return 0
    scores: list[tuple[float, int]] = []
    for i, left in enumerate(texts):
        others = [character_similarity(left, right) for j, right in enumerate(texts) if j != i]
        scores.append((sum(others) / len(others), -i))
    # Highest mean similarity; ties → lowest index (stable).
    scores.sort(reverse=True)
    return -scores[0][1]


def pick_medoid(views: list[TranscriptView]) -> TranscriptView:
    if not views:
        raise ValueError("pick_medoid requires non-empty views")
    ordered = sorted(views, key=lambda item: item.model)
    index = medoid_index([item.text for item in ordered])
    return ordered[index]


def _cluster_members(
    views: list[TranscriptView],
    *,
    threshold: float,
) -> list[TranscriptView]:
    """Greedy largest subset where every pairwise similarity >= threshold."""
    if not views:
        return []
    ordered = sorted(views, key=lambda item: item.model)
    best: list[TranscriptView] = []
    for seed in ordered:
        cluster = [seed]
        for candidate in ordered:
            if candidate.model == seed.model:
                continue
            if all(
                character_similarity(candidate.text, member.text) >= threshold
                for member in cluster
            ):
                cluster.append(candidate)
        if _cluster_better(cluster, best):
            best = cluster
    return best


def _cluster_better(candidate: list[TranscriptView], incumbent: list[TranscriptView]) -> bool:
    if len(candidate) != len(incumbent):
        return len(candidate) > len(incumbent)
    cand_families = len({item.family for item in candidate})
    inc_families = len({item.family for item in incumbent})
    if cand_families != inc_families:
        return cand_families > inc_families
    cand_key = tuple(sorted(item.model for item in candidate))
    inc_key = tuple(sorted(item.model for item in incumbent))
    return cand_key < inc_key


def family_semantic(
    views: list[TranscriptView],
    family: str,
    *,
    negative_pattern: re.Pattern[str] | None,
    positive_pattern: re.Pattern[str] | None,
) -> str:
    classes = {
        semantic_class(
            item.text,
            negative_pattern=negative_pattern,
            positive_pattern=positive_pattern,
        )
        for item in views
        if item.family == family and item.text
    }
    classes.discard(SEMANTIC_NONE)
    if not classes:
        return SEMANTIC_NONE
    if classes == {SEMANTIC_POSITIVE}:
        return SEMANTIC_POSITIVE
    if classes == {SEMANTIC_NEGATIVE}:
        return SEMANTIC_NEGATIVE
    return SEMANTIC_CONFLICT


def family_representative_text(views: list[TranscriptView], family: str) -> str | None:
    members = [item for item in views if item.family == family and item.text]
    if not members:
        return None
    return pick_medoid(members).text


def collect_transcripts(sample: Sample, families: dict[str, list[str]]) -> list[TranscriptView]:
    views: list[TranscriptView] = []
    for model in sorted(sample.transcripts):
        text = plain_transcript_text(sample.get_transcript_text(model))
        views.append(
            TranscriptView(
                model=str(model),
                family=resolve_family(str(model), families),
                text=text,
            )
        )
    return views


def _result(
    *,
    type_: str,
    decision: str,
    reason: str,
    label: str | None = None,
    selected: TranscriptView | None = None,
    support: list[TranscriptView] | None = None,
    consensus_score: float | None = None,
    min_similarity: float | None = None,
    review_reason: str | None = None,
    diagnostics: dict[str, Any] | None = None,
    rule_version: str = RULE_VERSION,
) -> ClassificationResult:
    support = support or ([] if selected is None else [selected])
    diagnostics = diagnostics or {}
    return ClassificationResult(
        type=type_,
        decision=decision,
        label=label,
        reason=reason,
        selected_model=selected.model if selected else None,
        support_models=[item.model for item in support],
        support_count=len(support),
        support_family_count=len({item.family for item in support}),
        consensus_score=consensus_score,
        min_similarity=min_similarity,
        qwen_family_text=diagnostics.get("qwen_family_text"),
        sensevoice_family_text=diagnostics.get("sensevoice_family_text"),
        semantic_qwen=diagnostics.get("semantic_qwen"),
        semantic_sensevoice=diagnostics.get("semantic_sensevoice"),
        review_reason=review_reason,
        rule_version=rule_version,
    )


def _voicemail_hit_families(
    sample: Sample,
    views: list[TranscriptView],
    pattern: re.Pattern[str] | None,
) -> set[str]:
    if pattern is None:
        return set()
    hit_families: set[str] = set()
    for view in views:
        texts = [view.text] if view.text else []
        entry = sample.transcripts.get(view.model)
        if isinstance(entry, dict):
            extra = entry.get("extra") if isinstance(entry.get("extra"), dict) else {}
            raw = str(extra.get("raw_text") or "").strip()
            if raw and raw not in texts:
                texts.append(raw)
        if any(pattern.search(text) for text in texts if text):
            hit_families.add(view.family)
    return hit_families


def classify_sample(
    sample: Sample,
    config: SelectionConfig,
    *,
    voicemail_pattern: re.Pattern[str] | None = None,
) -> ClassificationResult:
    views = collect_transcripts(sample, config.model_families)
    nonempty = [item for item in views if item.text]
    empty = [item for item in views if not item.text]
    diagnostics = {
        "qwen_family_text": family_representative_text(nonempty, config.primary_family),
        "sensevoice_family_text": family_representative_text(
            nonempty, config.secondary_family
        ),
        "semantic_qwen": None,
        "semantic_sensevoice": None,
    }
    negative_pattern = _compile_phrase_pattern(config.negative_phrases)
    positive_pattern = _compile_phrase_pattern(config.positive_phrases)
    diagnostics["semantic_qwen"] = family_semantic(
        nonempty,
        config.primary_family,
        negative_pattern=negative_pattern,
        positive_pattern=positive_pattern,
    )
    diagnostics["semantic_sensevoice"] = family_semantic(
        nonempty,
        config.secondary_family,
        negative_pattern=negative_pattern,
        positive_pattern=positive_pattern,
    )

    broken = bool(
        sample.labels.get("label_broken")
        or sample.labels.get("broken")
        or sample.quality.get("broken")
    )
    duration = sample.duration
    if broken or (duration is not None and duration <= 0):
        return _result(
            type_=TYPE_NOISE,
            decision=DECISION_AUTO_EMPTY,
            reason="broken_or_invalid_audio",
            label="",
            diagnostics=diagnostics,
            rule_version=config.rule_version,
        )

    if views and not nonempty:
        return _result(
            type_=TYPE_NOISE,
            decision=DECISION_AUTO_EMPTY,
            reason="empty_asr_transcripts",
            label="",
            support=[],
            diagnostics=diagnostics,
            rule_version=config.rule_version,
        )

    voicemail_families = _voicemail_hit_families(sample, views, voicemail_pattern)
    if len(voicemail_families) >= 2:
        support = [
            item
            for item in nonempty
            if item.family in voicemail_families
            and voicemail_pattern is not None
            and (
                (item.text and voicemail_pattern.search(item.text))
                or _raw_hit(sample, item.model, voicemail_pattern)
            )
        ]
        if not support:
            support = [item for item in nonempty if item.family in voicemail_families]
        return _decide_voicemail(support, config, diagnostics)

    primary_views = [item for item in views if item.family == config.primary_family]
    primary_nonempty = [item for item in primary_views if item.text]

    sem_primary = diagnostics["semantic_qwen"]
    sem_secondary = diagnostics["semantic_sensevoice"]
    if (
        sem_primary in {SEMANTIC_POSITIVE, SEMANTIC_NEGATIVE}
        and sem_secondary in {SEMANTIC_POSITIVE, SEMANTIC_NEGATIVE}
        and sem_primary != sem_secondary
    ):
        return _decide_semantic_inversion(
            nonempty,
            config,
            diagnostics,
            primary_class=sem_primary,
            secondary_class=sem_secondary,
        )

    primary_all_empty = bool(primary_views) and not primary_nonempty
    other_nonempty = [item for item in nonempty if item.family != config.primary_family]
    if primary_all_empty and other_nonempty:
        return _result(
            type_=TYPE_QWEN_MISSING,
            decision=DECISION_MODEL_REVIEW,
            reason="primary_family_empty_conflict",
            label=None,
            support=other_nonempty,
            review_reason="qwen_family_empty_other_nonempty",
            diagnostics=diagnostics,
            rule_version=config.rule_version,
        )

    total_models = len(views)
    empty_ratio = (len(empty) / total_models) if total_models else 0.0
    nonempty_families = {item.family for item in nonempty}
    observed_families = {item.family for item in views}
    if (
        total_models > 0
        and empty_ratio >= config.empty_ratio_for_hallucination
        and len(nonempty_families) == 1
        and nonempty
    ):
        # Doc: isolated family + empty majority → hallucination candidate.
        # If ≥3 families and every non-speaking family is empty → auto_empty.
        if len(observed_families) >= 3:
            return _result(
                type_=TYPE_HALLUCINATION,
                decision=DECISION_AUTO_EMPTY,
                reason="hallucination_other_families_empty",
                label="",
                support=nonempty,
                min_similarity=pairwise_min_similarity([item.text for item in nonempty]),
                review_reason="isolated_family_others_empty",
                diagnostics=diagnostics,
                rule_version=config.rule_version,
            )
        return _result(
            type_=TYPE_HALLUCINATION,
            decision=DECISION_MODEL_REVIEW,
            reason="hallucination_candidate_isolated_family",
            label=None,
            support=nonempty,
            min_similarity=pairwise_min_similarity([item.text for item in nonempty]),
            review_reason="empty_majority_single_family_text",
            diagnostics=diagnostics,
            rule_version=config.rule_version,
        )

    # Only a single model empty (or sparse empties) → continue consensus on nonempty.
    if len(nonempty) < 2 or len({item.family for item in nonempty}) < 2:
        return _result(
            type_=TYPE_HARDCASE,
            decision=DECISION_MODEL_REVIEW,
            reason="insufficient_independent_families",
            label=None,
            support=nonempty,
            review_reason="need_at_least_two_families",
            diagnostics=diagnostics,
            rule_version=config.rule_version,
        )

    if len(nonempty) == len(views):
        min_sim = pairwise_min_similarity([item.text for item in nonempty])
        semantic_classes = {
            semantic_class(
                item.text,
                negative_pattern=negative_pattern,
                positive_pattern=positive_pattern,
            )
            for item in nonempty
        }
        semantic_classes.discard(SEMANTIC_NONE)
        semantic_ok = len(semantic_classes) <= 1
        if (
            min_sim is not None
            and min_sim >= config.strict_threshold
            and semantic_ok
        ):
            chosen = pick_medoid(nonempty)
            return _result(
                type_=TYPE_AUTO_GOLD,
                decision=DECISION_AUTO_ACCEPT,
                reason="all_models_strict_consensus",
                label=chosen.text,
                selected=chosen,
                support=nonempty,
                consensus_score=min_sim,
                min_similarity=min_sim,
                diagnostics=diagnostics,
                rule_version=config.rule_version,
            )

    cluster = _cluster_members(nonempty, threshold=config.consensus_threshold)
    if cluster:
        ratio = len(cluster) / max(len(views), 1)
        family_count = len({item.family for item in cluster})
        semantic_classes = {
            semantic_class(
                item.text,
                negative_pattern=negative_pattern,
                positive_pattern=positive_pattern,
            )
            for item in cluster
        }
        semantic_classes.discard(SEMANTIC_NONE)
        semantic_ok = len(semantic_classes) <= 1
        min_sim = pairwise_min_similarity([item.text for item in cluster])
        if (
            ratio >= config.dominant_ratio
            and family_count >= 2
            and semantic_ok
            and min_sim is not None
            and min_sim >= config.consensus_threshold
        ):
            chosen = pick_medoid(cluster)
            return _result(
                type_=TYPE_CONSENSUS_GOLD,
                decision=DECISION_AUTO_ACCEPT,
                reason="dominant_cross_family_cluster",
                label=chosen.text,
                selected=chosen,
                support=cluster,
                consensus_score=ratio,
                min_similarity=min_sim,
                diagnostics=diagnostics,
                rule_version=config.rule_version,
            )

    min_sim = pairwise_min_similarity([item.text for item in nonempty])
    return _result(
        type_=TYPE_HARDCASE,
        decision=DECISION_MODEL_REVIEW,
        reason="no_reliable_consensus",
        label=None,
        support=nonempty,
        min_similarity=min_sim,
        review_reason="no_dominant_cluster",
        diagnostics=diagnostics,
        rule_version=config.rule_version,
    )


def _raw_hit(sample: Sample, model: str, pattern: re.Pattern[str]) -> bool:
    entry = sample.transcripts.get(model)
    if not isinstance(entry, dict):
        return False
    extra = entry.get("extra") if isinstance(entry.get("extra"), dict) else {}
    raw = str(extra.get("raw_text") or "").strip()
    return bool(raw and pattern.search(raw))


def _decide_voicemail(
    support: list[TranscriptView],
    config: SelectionConfig,
    diagnostics: dict[str, Any],
) -> ClassificationResult:
    if not support:
        return _result(
            type_=TYPE_VOICEMAIL,
            decision=DECISION_MANUAL_REVIEW,
            reason="voicemail_multi_family_no_text",
            label=None,
            review_reason="voicemail_hit_without_text",
            diagnostics=diagnostics,
            rule_version=config.rule_version,
        )
    min_sim = pairwise_min_similarity([item.text for item in support])
    if min_sim is not None and min_sim >= config.strict_threshold:
        chosen = pick_medoid(support)
        return _result(
            type_=TYPE_VOICEMAIL,
            decision=DECISION_AUTO_ACCEPT,
            reason="voicemail_strict_consensus",
            label=chosen.text,
            selected=chosen,
            support=support,
            consensus_score=min_sim,
            min_similarity=min_sim,
            diagnostics=diagnostics,
            rule_version=config.rule_version,
        )
    cluster = _cluster_members(support, threshold=config.consensus_threshold)
    if cluster and len(cluster) / max(len(support), 1) >= config.dominant_ratio:
        min_cluster = pairwise_min_similarity([item.text for item in cluster])
        if min_cluster is not None and min_cluster >= config.consensus_threshold:
            chosen = pick_medoid(cluster)
            return _result(
                type_=TYPE_VOICEMAIL,
                decision=DECISION_AUTO_ACCEPT,
                reason="voicemail_dominant_cluster",
                label=chosen.text,
                selected=chosen,
                support=cluster,
                consensus_score=len(cluster) / max(len(support), 1),
                min_similarity=min_cluster,
                diagnostics=diagnostics,
                rule_version=config.rule_version,
            )
    return _result(
        type_=TYPE_VOICEMAIL,
        decision=DECISION_MANUAL_REVIEW,
        reason="voicemail_low_agreement",
        label=None,
        support=support,
        min_similarity=min_sim,
        review_reason="voicemail_texts_disagree",
        diagnostics=diagnostics,
        rule_version=config.rule_version,
    )


def _decide_semantic_inversion(
    nonempty: list[TranscriptView],
    config: SelectionConfig,
    diagnostics: dict[str, Any],
    *,
    primary_class: str,
    secondary_class: str,
) -> ClassificationResult:
    """Family-level polarity conflict; third family may adjudicate."""
    by_family: dict[str, list[TranscriptView]] = {}
    for item in nonempty:
        by_family.setdefault(item.family, []).append(item)

    negative_pattern = _compile_phrase_pattern(config.negative_phrases)
    positive_pattern = _compile_phrase_pattern(config.positive_phrases)
    family_classes: dict[str, str] = {}
    for family, members in by_family.items():
        family_classes[family] = family_semantic(
            members,
            family,
            negative_pattern=negative_pattern,
            positive_pattern=positive_pattern,
        )

    votes: dict[str, set[str]] = {
        SEMANTIC_POSITIVE: set(),
        SEMANTIC_NEGATIVE: set(),
    }
    for family, cls in family_classes.items():
        if cls in votes:
            votes[cls].add(family)

    pos_n = len(votes[SEMANTIC_POSITIVE])
    neg_n = len(votes[SEMANTIC_NEGATIVE])
    majority: str | None = None
    if pos_n >= 2 and pos_n > neg_n:
        majority = SEMANTIC_POSITIVE
    elif neg_n >= 2 and neg_n > pos_n:
        majority = SEMANTIC_NEGATIVE

    if majority is not None:
        support = [
            item for item in nonempty if family_classes.get(item.family) == majority
        ]
        chosen = pick_medoid(support)
        return _result(
            type_=TYPE_SEMANTIC_INVERSION,
            decision=DECISION_AUTO_ACCEPT,
            reason="semantic_inversion_third_family_majority",
            label=chosen.text,
            selected=chosen,
            support=support,
            review_reason=None,
            diagnostics=diagnostics,
            rule_version=config.rule_version,
        )

    return _result(
        type_=TYPE_SEMANTIC_INVERSION,
        decision=DECISION_MODEL_REVIEW,
        reason="semantic_polarity_family_conflict",
        label=None,
        support=nonempty,
        review_reason=f"{config.primary_family}={primary_class},{config.secondary_family}={secondary_class}",
        diagnostics=diagnostics,
        rule_version=config.rule_version,
    )


def result_as_dict(result: ClassificationResult) -> dict[str, Any]:
    return asdict(result)
