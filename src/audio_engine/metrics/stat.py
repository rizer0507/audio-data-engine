"""Corpus-level metric statistic (business_metrics_v1)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class MetricStat:
    """Unified metric output: zero denominator → value=null (never 0%)."""

    name: str
    numerator: float | int
    denominator: int
    value: float | None
    eligible_count: int
    excluded_count: int
    extras: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_counts(
        name: str,
        *,
        numerator: float | int,
        denominator: int,
        eligible_count: int | None = None,
        excluded_count: int = 0,
        extras: dict[str, Any] | None = None,
    ) -> MetricStat:
        den = int(denominator)
        value = None if den <= 0 else float(numerator) / float(den)
        eligible = int(eligible_count) if eligible_count is not None else den
        return MetricStat(
            name=name,
            numerator=numerator,
            denominator=den,
            value=None if value is None else round(value, 6),
            eligible_count=eligible,
            excluded_count=int(excluded_count),
            extras=dict(extras or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload
