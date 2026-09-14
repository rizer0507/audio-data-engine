"""Deterministic gold-text selection. Eligibility is decided before this runs.

``D`` uses the versioned tolerant distance. Exact-transcript support does not
merge 你/您 or other tolerance mappings. Dual-run never adds a second vote.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from audio_engine.core.selection_v3.text_tolerance import tolerant_distance


@dataclass(frozen=True)
class FamilyRep:
    family: str
    run_id: str
    transcript_text: str
    raw_text: str
    tolerant_key: str
    comparison_text: str


@dataclass
class SelectionResult:
    transcript_text: str
    raw_text: str
    family: str
    run_id: str
    distance: float
    transcript_support_count: int
    tie_break_reason: str
    representatives: list[dict[str, str]] = field(default_factory=list)
    family_order: list[str] = field(default_factory=list)
    run_order: list[str] = field(default_factory=list)
    tolerance_version: str = ""


def _round6(value: float) -> float:
    return round(float(value), 6)


def mean_tolerant_distance(index: int, reps: list[FamilyRep]) -> float:
    if len(reps) < 2:
        raise ValueError("gold selection requires k>=2 representatives")
    total = 0.0
    for j, other in enumerate(reps):
        if j == index:
            continue
        dist = tolerant_distance(reps[index].tolerant_key, other.tolerant_key)
        total += 1.0 if dist is None else dist
    return _round6(total / (len(reps) - 1))


def transcript_support_count(rep: FamilyRep, reps: list[FamilyRep]) -> int:
    """Exact body match among qualified family representatives. One family, one count."""
    return sum(1 for item in reps if item.transcript_text == rep.transcript_text)


def select_representative_text(
    reps: list[FamilyRep],
    *,
    family_order: list[str],
    run_order: list[str] | None = None,
    tolerance_version: str = "",
) -> SelectionResult | None:
    """Pick the body with the smallest mean tolerant distance.

    Tie order: more exact-transcript supporters, then configured family order.
    Input order and dict order must not matter.
    """
    if len(reps) < 2:
        return None
    ordered = sorted(
        reps,
        key=lambda item: (
            family_order.index(item.family) if item.family in family_order else 10_000,
            run_order.index(item.run_id) if run_order and item.run_id in run_order else item.run_id,
        ),
    )
    scored: list[tuple[FamilyRep, float, int, int]] = []
    for item in ordered:
        idx = ordered.index(item)
        distance = mean_tolerant_distance(index=idx, reps=ordered)
        support = transcript_support_count(item, ordered)
        rank = family_order.index(item.family) if item.family in family_order else 10_000
        scored.append((item, distance, support, rank))

    best_d = min(item[1] for item in scored)
    tied = [item for item in scored if item[1] == best_d]
    reason = "min_mean_tolerant_distance"
    if len(tied) > 1:
        best_support = max(item[2] for item in tied)
        supported = [item for item in tied if item[2] == best_support]
        if len(supported) < len(tied):
            reason = "exact_transcript_support"
            tied = supported
        else:
            reason = "configured_family_order"
        tied = sorted(tied, key=lambda item: (item[3], item[0].run_id))
    chosen, distance, support, _rank = tied[0]
    if len(scored) == 1:
        reason = "min_mean_tolerant_distance"
    return SelectionResult(
        transcript_text=chosen.transcript_text,
        raw_text=chosen.raw_text,
        family=chosen.family,
        run_id=chosen.run_id,
        distance=distance,
        transcript_support_count=support,
        tie_break_reason=reason,
        representatives=[
            {
                "family": item.family,
                "run_id": item.run_id,
                "transcript_text": item.transcript_text,
                "tolerant_key": item.tolerant_key,
            }
            for item in ordered
        ],
        family_order=list(family_order),
        run_order=list(run_order or []),
        tolerance_version=tolerance_version,
    )


def exclude_family(reps: list[FamilyRep], family: str) -> list[FamilyRep]:
    return [item for item in reps if item.family != family]
