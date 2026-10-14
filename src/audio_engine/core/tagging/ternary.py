"""Three-valued logic for rule conditions (040 §5.1)."""

from __future__ import annotations

from collections.abc import Iterable

from audio_engine.core.tagging.types import Ternary


def ternary_and(values: Iterable[Ternary]) -> Ternary:
    items = list(values)
    if not items:
        return Ternary.MATCHED
    if any(v == Ternary.NOT_MATCHED for v in items):
        return Ternary.NOT_MATCHED
    if all(v == Ternary.MATCHED for v in items):
        return Ternary.MATCHED
    return Ternary.UNKNOWN


def ternary_or(values: Iterable[Ternary]) -> Ternary:
    items = list(values)
    if not items:
        return Ternary.NOT_MATCHED
    if any(v == Ternary.MATCHED for v in items):
        return Ternary.MATCHED
    if all(v == Ternary.NOT_MATCHED for v in items):
        return Ternary.NOT_MATCHED
    return Ternary.UNKNOWN


def ternary_not(value: Ternary) -> Ternary:
    if value == Ternary.MATCHED:
        return Ternary.NOT_MATCHED
    if value == Ternary.NOT_MATCHED:
        return Ternary.MATCHED
    return Ternary.UNKNOWN


def at_least_n(values: Iterable[Ternary], n: int) -> Ternary:
    """Family-set quantifier: at least n MATCHED; unknowns don't count as matched."""
    items = list(values)
    matched = sum(1 for v in items if v == Ternary.MATCHED)
    unknown = sum(1 for v in items if v == Ternary.UNKNOWN)
    if matched >= n:
        return Ternary.MATCHED
    # Even if all unknowns become matched, still cannot reach n.
    if matched + unknown < n:
        return Ternary.NOT_MATCHED
    return Ternary.UNKNOWN
