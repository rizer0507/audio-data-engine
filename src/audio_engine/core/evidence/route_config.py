"""Route / family mapping interface for neutral evidence builders (044).

``RouteConfig`` is a structural Protocol. Both the historical ``SelectionV3Config``
and the independent ``TagEvidenceConfig`` satisfy it, so route views and family
states can be built without importing either business config type.

``RouteConfigData`` is a plain, standalone implementation (useful for tests and as a
base for ``TagEvidenceConfig``). It deliberately carries **no** selection thresholds
such as family similarity, queue gating, or category policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from audio_engine.core.evidence.classify_text import (
    echo_for_family,
    uses_chinese_only_text,
)

DEFAULT_PUNCTUATION_TO_STRIP = list("，。！？、；：\"\"''（）【】《》…—·,.!?;:'\"()[]{}")


@runtime_checkable
class RouteConfig(Protocol):
    """Minimal contract used by collect_route_views / build_family_states / run status."""

    model_families: dict[str, list[str]]
    expected_runs_per_family: int
    punctuation_to_strip: list[str]
    classify_text_policy: str
    classify_text_keep_digits: bool
    critical_short_responses: list[str]
    short_polarity_max_han_chars: int
    negative_phrases: list[str]
    positive_phrases: list[str]
    critical_tokens: list[str]
    profanity_or_reject: list[str]
    filler_phrases: list[str]
    affirmation_phrases: list[str]
    dnsmos_weak_sig: float | None

    def ordered_families(self) -> list[str]: ...

    def uses_chinese_only_text(self) -> bool: ...

    def echo_table_for(self, family: str | None) -> Any: ...

    def family_of_key(self, transcript_key: str) -> str | None: ...


@dataclass
class RouteConfigData:
    """Standalone RouteConfig implementation (no selection business fields)."""

    model_families: dict[str, list[str]] = field(default_factory=dict)
    family_order: list[str] = field(default_factory=list)
    expected_runs_per_family: int = 2
    run_aliases: dict[str, str] = field(default_factory=dict)
    punctuation_to_strip: list[str] = field(default_factory=lambda: list(DEFAULT_PUNCTUATION_TO_STRIP))
    classify_text_policy: str = "five_class_v1"
    classify_text_keep_digits: bool = True
    classify_text_echo_missing: str = "fail"
    classify_text_echo: dict[str, Any] = field(default_factory=dict)
    classify_text_echo_fingerprint: str = ""
    critical_short_responses: list[str] = field(default_factory=list)
    short_polarity_max_han_chars: int = 4
    negative_phrases: list[str] = field(default_factory=list)
    positive_phrases: list[str] = field(default_factory=list)
    critical_tokens: list[str] = field(default_factory=list)
    profanity_or_reject: list[str] = field(default_factory=list)
    filler_phrases: list[str] = field(default_factory=list)
    affirmation_phrases: list[str] = field(default_factory=list)
    dnsmos_weak_sig: float | None = None

    def ordered_families(self) -> list[str]:
        ordered: list[str] = []
        for family in self.family_order:
            if family in self.model_families and family not in ordered:
                ordered.append(family)
        for family in self.model_families:
            if family not in ordered:
                ordered.append(family)
        return ordered

    def uses_chinese_only_text(self) -> bool:
        return uses_chinese_only_text(self.classify_text_policy)

    def echo_table_for(self, family: str | None):
        return echo_for_family(self.classify_text_echo, family)

    def family_of_key(self, transcript_key: str) -> str | None:
        key = str(transcript_key).strip()
        if key in self.run_aliases:
            return self.run_aliases[key]
        for family, keys in self.model_families.items():
            if key in keys:
                return family
        return None
