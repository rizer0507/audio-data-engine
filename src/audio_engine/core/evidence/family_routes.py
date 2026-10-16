"""Per-route views over ASR transcripts (extracted from selection_v3.family_evidence, 044).

``collect_route_views`` accepts any ``RouteConfig`` (family/route mapping + text policy).
It never reads selection thresholds, queues, or categories.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from audio_engine.core.evidence.asr_status import classify_run_status
from audio_engine.core.evidence.route_config import RouteConfig
from audio_engine.core.evidence.text import (
    comparison_text,
    raw_transcript_text,
    transcript_text,
)
from audio_engine.core.evidence.types import (
    ROUTE_FAILED,
    ROUTE_MISSING,
    RUN_STATUS_FAILED,
    RUN_STATUS_MISSING,
    RUN_STATUS_SUCCESS_EMPTY,
    RUN_STATUS_SUCCESS_TEXT,
)
from audio_engine.core.sample import Sample


@dataclass
class RouteView:
    run_id: str
    family: str
    status: str
    raw_text: str
    comparison_text: str
    transcript_text: str = ""
    classify_text: str = ""
    empty_reason_codes: tuple[str, ...] = ()
    pre_filter_language: str = ""
    body_text: str = ""
    route_disposition: str = "eligible"
    exclusion_reasons: tuple[str, ...] = ()


def _entry(sample: Sample, key: str) -> Any:
    return sample.transcripts.get(key)


def collect_route_views(
    sample: Sample,
    config: RouteConfig,
    *,
    quarantine_run_ids: set[str] | frozenset[str] | None = None,
) -> list[RouteView]:
    views: list[RouteView] = []
    punct = config.punctuation_to_strip
    blocked = set(quarantine_run_ids or ())
    chinese_only = config.uses_chinese_only_text()
    for family in config.ordered_families():
        for key in config.model_families.get(family, []):
            status = classify_run_status(sample, key, config)
            entry = _entry(sample, key)
            raw = raw_transcript_text(entry) if entry is not None else ""
            # Quarantined routes keep raw text for evidence but do not vote.
            if str(key) in blocked:
                views.append(
                    RouteView(
                        run_id=str(key),
                        family=family,
                        status=RUN_STATUS_FAILED,
                        raw_text=raw,
                        comparison_text="",
                    )
                )
                continue
            transcript = ""
            classify = ""
            reasons: tuple[str, ...] = ()
            pre_lang = ""
            body = ""
            disposition = "eligible"
            exclusions: tuple[str, ...] = ()
            if chinese_only:
                from audio_engine.core.evidence.classify_text import prepare_classify_text

                prepared = prepare_classify_text(
                    raw or (entry.get("text") if isinstance(entry, dict) else raw),
                    echo=config.echo_table_for(family),
                    keep_digits=config.classify_text_keep_digits,
                    policy=config.classify_text_policy,
                )
                transcript = prepared.transcript_text
                body = prepared.body_text or prepared.transcript_text
                classify = prepared.classify_text
                reasons = prepared.empty_reason_codes
                pre_lang = prepared.pre_filter_language
                disposition = prepared.route_disposition
                exclusions = prepared.exclusion_reasons
                if status in {RUN_STATUS_FAILED, RUN_STATUS_MISSING}:
                    cmp = ""
                    disposition = ROUTE_FAILED if status == RUN_STATUS_FAILED else ROUTE_MISSING
                elif prepared.is_excluded:
                    # Excluded routes keep audit text but never vote or count as empty.
                    status = RUN_STATUS_SUCCESS_EMPTY
                    cmp = ""
                    classify = ""
                elif prepared.status == RUN_STATUS_SUCCESS_EMPTY:
                    status = RUN_STATUS_SUCCESS_EMPTY
                    cmp = ""
                else:
                    status = RUN_STATUS_SUCCESS_TEXT
                    cmp = classify
            elif status == RUN_STATUS_SUCCESS_TEXT:
                cmp = comparison_text(
                    raw or (entry.get("text") if isinstance(entry, dict) else raw),
                    punctuation_to_strip=punct,
                )
                transcript = transcript_text(raw)
                body = transcript
            elif status == RUN_STATUS_SUCCESS_EMPTY:
                cmp = ""
                transcript = transcript_text(raw)
                body = transcript
            else:
                cmp = ""
                disposition = ROUTE_FAILED if status == RUN_STATUS_FAILED else (
                    ROUTE_MISSING if status == RUN_STATUS_MISSING else disposition
                )
            views.append(
                RouteView(
                    run_id=str(key),
                    family=family,
                    status=status,
                    raw_text=raw,
                    comparison_text=cmp,
                    transcript_text=transcript,
                    classify_text=classify,
                    empty_reason_codes=reasons,
                    pre_filter_language=pre_lang,
                    body_text=body,
                    route_disposition=disposition,
                    exclusion_reasons=exclusions,
                )
            )
    return views
