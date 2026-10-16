"""Shared acoustic evidence store (044).

Acoustic *facts* are shared across consumers (selection_v3, tagging, ...). A fact is
only ever one of ``success / failed / missing / not_computed``. ``not_required`` is a
*scheduling* decision made by a consumer ("I do not need this dep"), so it is never
stored as a fact and it must never overwrite an existing ``success``.

This module is neutral: it does not know about tag rules, queues or categories.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from audio_engine.core.evidence.types import (
    ACOUSTIC_DEPS,
    FACT_FAILED,
    FACT_MISSING,
    FACT_NOT_COMPUTED,
    FACT_SUCCESS,
    normalize_fact_status,
)

# quality dict keys written per dependency (the persisted wire format is unchanged).
DEP_STATUS_KEYS: dict[str, str] = {
    "vad": "vad_status",
    "energy": "energy_status",
    "dnsmos": "dnsmos_status",
}

DEP_VALUE_KEYS: dict[str, tuple[str, ...]] = {
    "vad": ("vad_speech_state", "vad_error"),
    "energy": (
        "energy_state",
        "strict_all_channels_zero",
        "duration_ms",
        "rms_dbfs",
        "peak_dbfs",
        "non_silent_ratio",
        "energy_error",
    ),
    "dnsmos": (
        "dnsmos_sig",
        "dnsmos_bak",
        "dnsmos_ovrl",
        "dnsmos_model_digest",
        "dnsmos_preprocess_version",
        "dnsmos_short_audio_strategy",
        "dnsmos_error",
        "noise_band",
        "noise_risk",
        "quality_policy_version",
    ),
}

# Preference when two facts for the same dep collide (higher wins).
_FACT_RANK = {
    FACT_SUCCESS: 3,
    FACT_FAILED: 2,
    FACT_MISSING: 1,
    FACT_NOT_COMPUTED: 0,
}


@dataclass
class AcousticFact:
    dep: str
    status: str = FACT_NOT_COMPUTED
    values: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    source: str = ""  # computed | cached | quality

    def is_success(self) -> bool:
        return self.status == FACT_SUCCESS

    def to_quality_fields(self) -> dict[str, Any]:
        status_key = DEP_STATUS_KEYS[self.dep]
        out: dict[str, Any] = dict(self.values)
        if self.status in {FACT_SUCCESS, FACT_FAILED}:
            out[status_key] = self.status
        if self.error and self.status == FACT_FAILED:
            out.setdefault(f"{self.dep}_error", self.error)
        return out


def fact_from_quality(dep: str, quality: Mapping[str, Any] | None) -> AcousticFact:
    """Read one dependency fact out of a quality mapping (``not_required`` -> not computed)."""
    if dep not in DEP_STATUS_KEYS:
        raise KeyError(f"unknown acoustic dep: {dep!r}")
    quality = quality or {}
    raw = quality.get(DEP_STATUS_KEYS[dep])
    status = normalize_fact_status(raw)
    values = {k: quality[k] for k in DEP_VALUE_KEYS[dep] if k in quality and quality[k] is not None}
    error = quality.get(f"{dep}_error")
    return AcousticFact(
        dep=dep,
        status=status,
        values=values,
        error=str(error) if error else None,
        source="quality",
    )


def merge_facts(existing: AcousticFact | None, incoming: AcousticFact) -> AcousticFact:
    """Merge two facts for the same dep; never downgrade a SUCCESS."""
    if existing is None:
        return incoming
    if existing.dep != incoming.dep:
        raise ValueError(f"cannot merge facts of different deps: {existing.dep} vs {incoming.dep}")
    if _FACT_RANK[incoming.status] > _FACT_RANK[existing.status]:
        return incoming
    if _FACT_RANK[incoming.status] == _FACT_RANK[existing.status] and incoming.status == FACT_SUCCESS:
        # Same-rank success: keep the existing (already trusted) values, fill gaps only.
        merged = dict(incoming.values)
        merged.update(existing.values)
        return AcousticFact(dep=existing.dep, status=FACT_SUCCESS, values=merged, source=existing.source)
    return existing


def merge_quality_fields(
    existing: Mapping[str, Any] | None,
    incoming: Mapping[str, Any] | None,
    *,
    deps: Iterable[str] = ACOUSTIC_DEPS,
) -> dict[str, Any]:
    """Merge acoustic fields from ``incoming`` into ``existing`` without losing SUCCESS.

    Non-acoustic keys from ``incoming`` overwrite normally. For each acoustic dep the
    stored status/values follow :func:`merge_facts`. ``not_required`` / ``pending`` /
    empty statuses coming from a consumer schedule never replace a computed fact.
    """
    base = dict(existing or {})
    inc = dict(incoming or {})
    dep_keys: set[str] = set()
    for dep in deps:
        if dep not in DEP_STATUS_KEYS:
            continue
        dep_keys.add(DEP_STATUS_KEYS[dep])
        dep_keys.update(DEP_VALUE_KEYS[dep])
    for key, value in inc.items():
        if key not in dep_keys:
            base[key] = value
    for dep in deps:
        if dep not in DEP_STATUS_KEYS:
            continue
        old = fact_from_quality(dep, base)
        new = fact_from_quality(dep, inc)
        # Only consider incoming facts that actually carry something.
        has_incoming = DEP_STATUS_KEYS[dep] in inc or any(k in inc for k in DEP_VALUE_KEYS[dep])
        if not has_incoming:
            continue
        old_present = DEP_STATUS_KEYS[dep] in base or any(k in base for k in DEP_VALUE_KEYS[dep])
        merged = merge_facts(old if old_present else None, new)
        if merged is new and merged.status == FACT_NOT_COMPUTED and not old_present:
            # Preserve the explicit schedule label (e.g. not_required) when nothing exists yet.
            for k in (DEP_STATUS_KEYS[dep], *DEP_VALUE_KEYS[dep]):
                if k in inc:
                    base[k] = inc[k]
            continue
        if merged is old:
            continue
        for k in (DEP_STATUS_KEYS[dep], *DEP_VALUE_KEYS[dep]):
            base.pop(k, None)
        base.update(merged.to_quality_fields())
    return base


class SharedAcousticEvidenceStore:
    """In-memory (sample_id -> dep -> fact) store with SUCCESS-preserving merge."""

    def __init__(self) -> None:
        self._facts: dict[str, dict[str, AcousticFact]] = {}

    def get(self, sample_id: str, dep: str) -> AcousticFact | None:
        return self._facts.get(str(sample_id), {}).get(dep)

    def has_success(self, sample_id: str, dep: str) -> bool:
        fact = self.get(sample_id, dep)
        return bool(fact and fact.is_success())

    def put(self, sample_id: str, fact: AcousticFact) -> AcousticFact:
        slot = self._facts.setdefault(str(sample_id), {})
        slot[fact.dep] = merge_facts(slot.get(fact.dep), fact)
        return slot[fact.dep]

    def put_from_quality(self, sample_id: str, quality: Mapping[str, Any] | None) -> None:
        for dep in DEP_STATUS_KEYS:
            fact = fact_from_quality(dep, quality)
            if fact.status != FACT_NOT_COMPUTED or fact.values:
                self.put(sample_id, fact)

    def missing_deps(self, sample_id: str, deps: Iterable[str]) -> set[str]:
        """Deps with no SUCCESS fact yet (candidates for real compute)."""
        return {d for d in deps if not self.has_success(sample_id, d)}

    def quality_fields(self, sample_id: str, deps: Iterable[str] | None = None) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for dep in deps or DEP_STATUS_KEYS:
            fact = self.get(sample_id, dep)
            if fact is not None:
                out.update(fact.to_quality_fields())
        return out

    def __len__(self) -> int:
        return len(self._facts)
