"""Stratified spot-audit assignment for automatic gold / voicemail candidates.

Quota is ``min(n, max(floor_n, ceil(n * rate)))``. Default floor 100 cannot
prove a 1% error bound; this only marks a review sample, not a passed audit.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict

from audio_engine.core.sample import Sample


def spot_audit_quota(n: int, *, rate: float = 0.01, floor_n: int = 100) -> int:
    if n <= 0:
        return 0
    return min(n, max(floor_n, math.ceil(n * rate)))


def _stratum(sample: Sample) -> str:
    labels = sample.labels
    flags = []
    langs = labels.get("language_by_run") or {}
    if isinstance(langs, dict) and any(v == "en" for v in langs.values()):
        flags.append("glm_en")
    if isinstance(langs, dict) and any(v == "mixed" for v in langs.values()):
        flags.append("mixed")
    codes = labels.get("reason_codes") or []
    if isinstance(codes, str):
        codes = [codes]
    if any("tolerance" in str(c) or "pronoun" in str(c) for c in codes):
        flags.append("tolerance")
    category = str(labels.get("category") or "")
    return "|".join([category, *flags]) or category or "other"


def apply_spot_audit_flags(
    samples: list[Sample],
    *,
    rate: float = 0.01,
    floor_n: int = 100,
    seed: str = "spot_audit_v3_20260911",
) -> dict[str, int]:
    """Mark ``spot_audit_selected`` without pulling those rows into P0 packs."""
    eligible = [
        sample
        for sample in samples
        if (
            sample.labels.get("category") in {"gold", "voicemail"}
            and sample.labels.get("status") == "candidate"
        )
        or (
            sample.labels.get("category") in {"business_consistent", "voicemail", "non_speech"}
            and sample.labels.get("status") == "auto_classified"
        )
    ]
    by_cat: dict[str, list[Sample]] = defaultdict(list)
    for sample in eligible:
        by_cat[str(sample.labels.get("category"))].append(sample)

    selected_ids: set[str] = set()
    summary: dict[str, int] = {}
    for category, rows in by_cat.items():
        quota = spot_audit_quota(len(rows), rate=rate, floor_n=floor_n)
        strata: dict[str, list[Sample]] = defaultdict(list)
        for sample in rows:
            strata[_stratum(sample)].append(sample)
        picked: list[Sample] = []
        # Cover each stratum first, then fill by stable hash.
        for key in sorted(strata):
            if len(picked) >= quota:
                break
            ordered = sorted(
                strata[key],
                key=lambda s: hashlib.sha256(f"{seed}\0{category}\0{s.id}".encode()).hexdigest(),
            )
            picked.append(ordered[0])
        rest = [s for s in rows if s not in picked]
        rest.sort(
            key=lambda s: hashlib.sha256(f"{seed}\0{category}\0fill\0{s.id}".encode()).hexdigest()
        )
        for sample in rest:
            if len(picked) >= quota:
                break
            picked.append(sample)
        for sample in picked:
            selected_ids.add(sample.id)
        summary[category] = len(picked)

    for sample in samples:
        category = str(sample.labels.get("category") or "")
        if category not in {"gold", "voicemail", "business_consistent", "non_speech"}:
            continue
        auto = sample.labels.get("status") in {"candidate", "auto_classified"}
        sample.labels["spot_audit_eligible"] = bool(auto)
        sample.labels["spot_audit_selected"] = sample.id in selected_ids
        sample.labels["spot_audit_quota"] = summary.get(category, 0)
    return summary
