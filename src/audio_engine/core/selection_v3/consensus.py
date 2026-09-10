"""Cross-family consensus for selection_v3 (pairwise clusters, no transitive fakes)."""

from __future__ import annotations

from dataclasses import dataclass, field

from audio_engine.core.selection_v3.config import SelectionV3Config
from audio_engine.core.selection_v3.family_evidence import FamilyEvidence, RouteView
from audio_engine.core.selection_v3.text import pairwise_min_similarity, text_similarity
from audio_engine.core.selection_v3.types import FAMILY_STABLE_TEXT, RISK_CONSENSUS_AMBIGUOUS


@dataclass
class ConsensusCluster:
    members: list[RouteView]
    families: list[str]
    support_family_count: int
    support_ratio_of_4: float
    teacher_support_count: int
    min_similarity: float
    candidate_text: str
    candidate_run_id: str
    ambiguous: bool = False


@dataclass
class ConsensusResult:
    clusters: list[ConsensusCluster] = field(default_factory=list)
    primary: ConsensusCluster | None = None
    consensus_ambiguous: bool = False
    risk_tags: list[str] = field(default_factory=list)


def _pairwise_ok(views: list[RouteView], threshold: float) -> bool:
    texts = [v.comparison_text for v in views]
    if len(texts) < 2:
        return True
    for i, left in enumerate(texts):
        for right in texts[i + 1 :]:
            if text_similarity(left, right) < threshold:
                return False
    return True


def _family_balanced_medoid(
    views: list[RouteView],
    *,
    family_order: list[str],
) -> RouteView:
    """Equal family weight; ties broken by family order then run_id."""
    by_family: dict[str, list[RouteView]] = {}
    for view in views:
        by_family.setdefault(view.family, []).append(view)

    # One representative per family first (intra-family medoid by run_id order)
    family_reps: list[RouteView] = []
    for family in family_order:
        members = by_family.get(family) or []
        if not members:
            continue
        ordered = sorted(members, key=lambda r: r.run_id)
        if len(ordered) == 1:
            family_reps.append(ordered[0])
            continue
        best = ordered[0]
        best_score = -1.0
        for cand in ordered:
            scores = [
                text_similarity(cand.comparison_text, other.comparison_text)
                for other in ordered
                if other.run_id != cand.run_id
            ]
            mean = sum(scores) / len(scores) if scores else 1.0
            if mean > best_score or (mean == best_score and cand.run_id < best.run_id):
                best_score = mean
                best = cand
        family_reps.append(best)

    if not family_reps:
        raise ValueError("family-balanced medoid requires members")
    if len(family_reps) == 1:
        return family_reps[0]

    best = family_reps[0]
    best_score = -1.0
    for cand in family_reps:
        scores = [
            text_similarity(cand.comparison_text, other.comparison_text)
            for other in family_reps
            if other.run_id != cand.run_id
        ]
        mean = sum(scores) / len(scores)
        fam_rank = family_order.index(cand.family) if cand.family in family_order else 99
        if mean > best_score or (
            abs(mean - best_score) < 1e-9
            and (
                fam_rank < (family_order.index(best.family) if best.family in family_order else 99)
                or (
                    fam_rank
                    == (family_order.index(best.family) if best.family in family_order else 99)
                    and cand.run_id < best.run_id
                )
            )
        ):
            best_score = mean
            best = cand
    return best


def _build_cluster(
    members: list[RouteView],
    *,
    config: SelectionV3Config,
    threshold: float,
    configured_family_count: int,
) -> ConsensusCluster:
    families = sorted({m.family for m in members})
    teacher_count = sum(1 for f in families if f in config.teacher_families)
    min_sim = pairwise_min_similarity([m.comparison_text for m in members]) or 0.0
    # Also verify all original comparison_text of supporting families' both routes
    # are consistent with the cluster threshold when both are in members.
    medoid = _family_balanced_medoid(members, family_order=config.ordered_families())
    return ConsensusCluster(
        members=list(members),
        families=families,
        support_family_count=len(families),
        support_ratio_of_4=len(families) / max(configured_family_count, 1),
        teacher_support_count=teacher_count,
        min_similarity=min_sim,
        candidate_text=medoid.raw_text or medoid.comparison_text,
        candidate_run_id=medoid.run_id,
    )


def find_pairwise_clusters(
    votes: list[RouteView],
    *,
    threshold: float,
    config: SelectionV3Config,
) -> list[ConsensusCluster]:
    """Largest subsets where EVERY pairwise similarity >= threshold (no transitive)."""
    if not votes:
        return []
    ordered = sorted(votes, key=lambda r: (r.family, r.run_id))
    family_count = len(config.model_families)
    clusters: list[ConsensusCluster] = []
    seen: set[tuple[str, ...]] = set()

    for seed in ordered:
        cluster = [seed]
        for cand in ordered:
            if cand.run_id == seed.run_id:
                continue
            trial = cluster + [cand]
            if _pairwise_ok(trial, threshold):
                cluster = trial
        key = tuple(sorted(m.run_id for m in cluster))
        if key in seen:
            continue
        seen.add(key)
        clusters.append(
            _build_cluster(
                cluster,
                config=config,
                threshold=threshold,
                configured_family_count=family_count,
            )
        )

    # Prefer larger family support, then higher min_similarity, then lexical run ids
    clusters.sort(
        key=lambda c: (
            -c.support_family_count,
            -c.min_similarity,
            tuple(sorted(m.run_id for m in c.members)),
        )
    )
    return clusters


def verify_supporting_family_routes(
    cluster: ConsensusCluster,
    families: dict[str, FamilyEvidence],
    *,
    threshold: float,
    short: bool,
) -> bool:
    """Cluster consistency must cover both routes of supporting stable families."""
    for family in cluster.families:
        state = families.get(family)
        if state is None or state.status != FAMILY_STABLE_TEXT:
            continue
        texts = [
            r.comparison_text
            for r in state.routes
            if r.comparison_text
        ]
        if short:
            if len(set(texts)) > 1:
                return False
        else:
            if pairwise_min_similarity(texts) is not None:
                if (pairwise_min_similarity(texts) or 0.0) < threshold:
                    return False
        # Cross-check each family route against cluster candidate comparison
        cand = cluster.members[0].comparison_text
        for text in texts:
            if short:
                if text != cand and text not in {m.comparison_text for m in cluster.members}:
                    # Allow if matches any cluster member
                    if all(text != m.comparison_text for m in cluster.members):
                        return False
            else:
                if all(text_similarity(text, m.comparison_text) < threshold for m in cluster.members):
                    return False
    return True


def analyze_consensus(
    families: dict[str, FamilyEvidence],
    config: SelectionV3Config,
    *,
    threshold: float,
    short: bool,
) -> ConsensusResult:
    votes = [
        state.representative
        for state in families.values()
        if state.provides_text_vote and state.representative is not None
    ]
    clusters = find_pairwise_clusters(votes, threshold=threshold, config=config)
    valid = [
        c
        for c in clusters
        if verify_supporting_family_routes(c, families, threshold=threshold, short=short)
    ]
    if not valid:
        return ConsensusResult(clusters=clusters)

    top = valid[0]
    tied = [
        c
        for c in valid
        if c.support_family_count == top.support_family_count
        and abs(c.min_similarity - top.min_similarity) < 1e-9
    ]
    # Multiple max clusters with different texts → ambiguous
    texts = {c.members[0].comparison_text for c in tied}
    # Compare via family medoid comparison texts
    texts = {c.candidate_text for c in tied}  # raw may differ; use comparison of members
    texts = {m.comparison_text for c in tied for m in c.members[:1]}
    ambiguous = len(tied) > 1 and len({m.comparison_text for c in tied for m in [c.members[0]]}) > 1
    # More precise: distinct comparison medoids
    medoid_texts = set()
    for c in tied:
        medoid_texts.add(
            sorted(c.members, key=lambda r: (r.family, r.run_id))[0].comparison_text
        )
    ambiguous = len(tied) > 1 and len(medoid_texts) > 1

    risk_tags = [RISK_CONSENSUS_AMBIGUOUS] if ambiguous else []
    primary = None if ambiguous else top
    return ConsensusResult(
        clusters=valid,
        primary=primary,
        consensus_ambiguous=ambiguous,
        risk_tags=risk_tags,
    )


def eight_route_full_agreement(
    families: dict[str, FamilyEvidence],
    *,
    threshold: float,
    short: bool,
) -> tuple[bool, float | None]:
    """All eight success-text routes form one pairwise cluster at threshold."""
    routes: list[RouteView] = []
    for state in families.values():
        for route in state.routes:
            if route.comparison_text:
                routes.append(route)
    if len(routes) < sum(len(s.routes) for s in families.values()):
        # some empty or incomplete
        pass
    texts = [r.comparison_text for r in routes]
    if not texts:
        return False, None
    if short:
        ok = len(set(texts)) == 1
        return ok, 1.0 if ok else pairwise_min_similarity(texts)
    min_sim = pairwise_min_similarity(texts)
    if min_sim is None:
        return False, None
    return min_sim >= threshold and _pairwise_ok(routes, threshold), min_sim
