"""Review queue selection and dual-review requirement for annotation_v3."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Any, Iterable

from audio_engine.core.annotation_v3.config import AnnotationConfig
from audio_engine.core.sample import Sample
from audio_engine.core.selection_v3.types import (
    PRIORITY_P0,
    PRIORITY_P1,
    PRIORITY_P2,
)


def _risk_tags(sample: Sample) -> set[str]:
    raw = sample.labels.get("risk_tags") or []
    if isinstance(raw, str):
        return {x.strip() for x in raw.split(",") if x.strip()}
    return {str(x) for x in raw}


def requires_dual_review(sample: Sample, config: AnnotationConfig) -> bool:
    """Eval formal / P0 / empty-gold candidates / pseudo audit samples need dual review."""
    dual = config.dual_review
    labels = sample.labels
    role = str(labels.get("reservation_role") or labels.get("dataset_role") or "")
    if dual.eval_formal and role in dual.reservation_roles:
        return True
    priority = str(labels.get("review_priority") or "")
    if dual.priority_p0 and priority == PRIORITY_P0:
        return True
    queue = str(labels.get("review_queue") or "")
    if dual.pseudo_audit_samples and queue == "pseudo_audit":
        return True
    if dual.empty_gold_candidates:
        # Proposed empty / all-empty / presence conflict → empty gold candidates
        typ = str(labels.get("type") or "")
        if typ in {"all_empty_unverified", "speech_presence_disagreement"}:
            return True
        tags = _risk_tags(sample)
        if "presence_conflict" in tags:
            return True
    return False


def _p1_priority_score(sample: Sample, config: AnnotationConfig) -> int:
    tags = _risk_tags(sample)
    score = 0
    for i, tag in enumerate(config.budget.p1_priority_tags):
        if tag in tags or tag == str(sample.labels.get("type") or ""):
            score = max(score, len(config.budget.p1_priority_tags) - i)
        if tag == "qwen_correction_candidate" and sample.labels.get("qwen_correction_candidate"):
            score = max(score, len(config.budget.p1_priority_tags) - i)
        if tag == "all_empty_unverified" and sample.labels.get("type") == "all_empty_unverified":
            score = max(score, len(config.budget.p1_priority_tags) - i)
    return score


def _stable_rank(seed: str, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}\0{sample_id}".encode()).hexdigest()


def select_review_batch(
    samples: Iterable[Sample],
    config: AnnotationConfig,
    *,
    priorities: list[str] | None = None,
    queues: list[str] | None = None,
    limit: int | None = None,
    seed: str = "review_batch_v3",
    view: str | None = None,
) -> list[Sample]:
    """Select a review batch: P0 all, then P1 by budget priority, then stratified P2.

    Unselected samples remain in the pool — backlog must not relax auto-admit.
    """
    wanted_prio = set(priorities or [PRIORITY_P0, PRIORITY_P1, PRIORITY_P2])
    wanted_queues = set(queues) if queues else None
    pool: list[Sample] = []
    for sample in samples:
        queue = str(sample.labels.get("review_queue") or "")
        priority = str(sample.labels.get("review_priority") or "")
        # Also allow legacy classification_bucket review paths when queue empty.
        if not queue and not priority:
            bucket = str(sample.labels.get("classification_bucket") or "")
            if bucket in {
                "review_queue",
                "hardcase",
                "hallucination",
                "semantic_inversion",
                "critical_token_conflict",
            }:
                priority = PRIORITY_P1
                queue = "manual_review"
            else:
                continue
        if priority and priority not in wanted_prio:
            # pseudo_audit has priority None — include when queues ask for it
            if not (wanted_queues and queue in wanted_queues):
                continue
        if wanted_queues is not None and queue not in wanted_queues:
            continue
        state = str(sample.labels.get("annotation_state") or "")
        if state in {"second_review", "adjudicated", "rejected", "human_accepted"}:
            continue
        if view in {"second_review", "candidate_check"} and not sample.labels.get("annotator_id"):
            continue
        if view == "adjudication" and state != "conflict":
            continue
        if view == "spot_check" and (state != "annotated" or requires_dual_review(sample, config)):
            continue
        pool.append(sample)
    if view == "spot_check":
        layers = defaultdict(list)
        for sample in pool:
            layers[spot_check_stratum(sample)].append(sample)
        selected = []
        for layer, members in sorted(layers.items()):
            ranked = sorted(members, key=lambda s: (_stable_rank(seed + layer, s.id), s.id))
            expanded = any(s.labels.get("spot_check_layer_blocked") for s in members)
            count = len(ranked) if expanded else math.ceil(len(ranked) * config.spot_check.single_review_min_rate)
            selected.extend(ranked[:count])
        return selected  # Mandatory layer quota is not silently truncated by --limit.

    p0 = [s for s in pool if str(s.labels.get("review_priority") or "") == PRIORITY_P0]
    p1 = [s for s in pool if str(s.labels.get("review_priority") or "") == PRIORITY_P1]
    p2 = [s for s in pool if str(s.labels.get("review_priority") or "") == PRIORITY_P2]
    other = [
        s
        for s in pool
        if str(s.labels.get("review_priority") or "") not in {PRIORITY_P0, PRIORITY_P1, PRIORITY_P2}
    ]

    p1.sort(
        key=lambda s: (-_p1_priority_score(s, config), _stable_rank(seed, s.id), s.id)
    )
    p0.sort(key=lambda s: (_stable_rank(seed + "|p0", s.id), s.id))
    # P2: stratified by source / duration band / noise / family — use stable hash, not longest text.
    p2.sort(key=lambda s: (_stable_rank(seed + "|p2", s.id), s.id))
    other.sort(key=lambda s: (_stable_rank(seed + "|o", s.id), s.id))

    ordered = p0 + p1 + p2 + other
    if limit is None:
        return ordered
    return ordered[: max(0, int(limit))]


def spot_check_stratum(sample: Sample) -> str:
    labels = sample.labels
    duration_band = "short" if sample.duration is not None and sample.duration <= 2 else "long"
    return "|".join(str(x or "unknown") for x in (labels.get("source_snapshot_id"),
                    labels.get("type"), labels.get("human_noise"), duration_band))


def queue_id_v3(
    dataset_path: str,
    *,
    revision: str,
    view: str,
    priorities: list[str],
    queues: list[str] | None,
) -> str:
    parts = [
        str(dataset_path),
        revision,
        view,
        ",".join(sorted(set(priorities))),
        ",".join(sorted(set(queues or []))),
    ]
    digest = hashlib.sha256("\0".join(parts).encode()).hexdigest()[:16]
    return f"review_v3_{digest}"
