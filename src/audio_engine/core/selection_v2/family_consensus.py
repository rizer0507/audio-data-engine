"""Family-level consensus helpers for selection_v2.0."""

from __future__ import annotations

from dataclasses import dataclass

from audio_engine.core.selection_engine import (
    TranscriptView,
    _cluster_members,
    pairwise_min_similarity,
    pick_medoid,
)
from audio_engine.core.selection_v2.config import SelectionV2Config
from audio_engine.core.transcript_reconcile import character_similarity


@dataclass
class FamilyState:
    family: str
    views: list[TranscriptView]
    representative: TranscriptView | None
    internal_conflict: bool
    min_similarity: float | None


def analyze_families(
    views: list[TranscriptView],
    config: SelectionV2Config,
) -> dict[str, FamilyState]:
    by_family: dict[str, list[TranscriptView]] = {}
    for item in views:
        by_family.setdefault(item.family, []).append(item)

    states: dict[str, FamilyState] = {}
    for family, members in by_family.items():
        nonempty = [item for item in members if item.text]
        if not nonempty:
            states[family] = FamilyState(
                family=family,
                views=members,
                representative=None,
                internal_conflict=False,
                min_similarity=None,
            )
            continue
        min_sim = pairwise_min_similarity([item.text for item in nonempty])
        # Severe intra-family conflict: below consensus threshold across members.
        internal = (
            len(nonempty) >= 2
            and min_sim is not None
            and min_sim < config.consensus_threshold
        )
        states[family] = FamilyState(
            family=family,
            views=members,
            representative=pick_medoid(nonempty),
            internal_conflict=internal,
            min_similarity=min_sim,
        )
    return states


def voting_views(
    states: dict[str, FamilyState],
    *,
    exclude_conflict: bool = True,
) -> list[TranscriptView]:
    """One representative per family for cross-family consensus."""
    result: list[TranscriptView] = []
    for state in states.values():
        if state.representative is None:
            continue
        if exclude_conflict and state.internal_conflict:
            continue
        result.append(state.representative)
    return result


def find_dominant_cluster(
    views: list[TranscriptView],
    *,
    threshold: float,
    dominant_ratio: float,
    total_models: int,
    min_family_count: int,
) -> tuple[list[TranscriptView], float | None, float | None] | None:
    if len(views) < min_family_count:
        return None
    cluster = _cluster_members(views, threshold=threshold)
    if not cluster:
        return None
    ratio = len(cluster) / max(total_models, 1)
    family_count = len({item.family for item in cluster})
    min_sim = pairwise_min_similarity([item.text for item in cluster])
    if (
        ratio >= dominant_ratio
        and family_count >= min_family_count
        and min_sim is not None
        and min_sim >= threshold
    ):
        return cluster, ratio, min_sim
    return None


def strict_cross_family_consensus(
    views: list[TranscriptView],
    *,
    threshold: float,
    min_family_count: int,
) -> tuple[list[TranscriptView], float] | None:
    nonempty = [item for item in views if item.text]
    if len({item.family for item in nonempty}) < min_family_count:
        return None
    min_sim = pairwise_min_similarity([item.text for item in nonempty])
    if min_sim is not None and min_sim >= threshold:
        return nonempty, min_sim
    # Also accept a full-family medoid cluster that is pairwise strict.
    reps = list(views)
    if len({item.family for item in reps}) < min_family_count:
        return None
    min_rep = pairwise_min_similarity([item.text for item in reps])
    if min_rep is not None and min_rep >= threshold:
        return reps, min_rep
    return None


def texts_fully_agree(views: list[TranscriptView]) -> bool:
    texts = [item.text for item in views if item.text]
    return len(texts) >= 2 and len(set(texts)) == 1


def mean_pairwise(views: list[TranscriptView]) -> float | None:
    texts = [item.text for item in views if item.text]
    if len(texts) < 2:
        return 1.0 if texts else None
    total = 0.0
    count = 0
    for i, left in enumerate(texts):
        for right in texts[i + 1 :]:
            total += character_similarity(left, right)
            count += 1
    return total / count if count else None
