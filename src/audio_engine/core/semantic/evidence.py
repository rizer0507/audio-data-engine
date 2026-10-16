"""Extract family/run evidence for semantic templates (neutral, no business rules)."""

from __future__ import annotations

from typing import Any

from audio_engine.core.evidence.asr_status import classify_run_status, transcript_entry
from audio_engine.core.evidence.text import raw_transcript_text
from audio_engine.core.sample import Sample
from audio_engine.core.semantic.config import EvidenceConfig
from audio_engine.core.semantic.types import (
    RUN_FAILED,
    RUN_MISSING,
    RUN_SUCCESS_EMPTY,
    RUN_SUCCESS_TEXT,
)


def build_sample_context(sample: Sample, fields: list[str]) -> dict[str, Any]:
    """Whitelist sample metadata for templates."""
    allowed = {
        "id": sample.id,
        "duration": sample.duration,
        "source_path": sample.source_path,
        "sha256": sample.sha256,
        "sample_rate": sample.sample_rate,
        "channels": sample.channels,
    }
    out: dict[str, Any] = {}
    for name in fields:
        if name in allowed:
            out[name] = allowed[name]
            continue
        if name.startswith("labels.") and isinstance(sample.labels, dict):
            key = name.split(".", 1)[1]
            out[name] = sample.labels.get(key)
            continue
        if name.startswith("quality.") and isinstance(sample.quality, dict):
            key = name.split(".", 1)[1]
            out[name] = sample.quality.get(key)
            continue
        raise ValueError(
            f"sample_context_fields entry {name!r} is not allowed; "
            "use id/duration/source_path/sha256/sample_rate/channels "
            "or labels.<key> / quality.<key>"
        )
    return out


def _run_view(sample: Sample, family: str, run_id: str) -> dict[str, Any]:
    entry = transcript_entry(sample, run_id)
    status = classify_run_status(sample, run_id, config=None)
    text: str | None = None
    raw_text: str | None = None
    model: str | None = None
    version: str | None = None
    source = "missing"
    if entry is not None:
        source = "transcripts"
        model = entry.get("model")
        version = entry.get("version")
        if status == RUN_FAILED:
            # Explicit failure: do not treat residual text as evidence.
            text = None
            raw = raw_transcript_text(entry)
            raw_text = raw if raw else None
        elif status == RUN_MISSING:
            text = None
            raw_text = None
        else:
            body = entry.get("text")
            text = "" if body is None else str(body)
            extra = entry.get("extra") if isinstance(entry.get("extra"), dict) else {}
            if "raw_text" in extra and extra.get("raw_text") is not None:
                raw_text = str(extra.get("raw_text"))
    return {
        "run_id": run_id,
        "family": family,
        "status": status,
        "text": text,
        "raw_text": raw_text,
        "model": model,
        "version": version,
        "source": source,
    }


def build_families_context(
    sample: Sample, evidence: EvidenceConfig
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (families_for_template, evidence_summary)."""
    families: dict[str, Any] = {}
    summary_runs: list[dict[str, Any]] = []
    success_text_count = 0
    for family, run_ids in evidence.model_families.items():
        runs = [_run_view(sample, family, run_id) for run_id in run_ids]
        families[family] = {"runs": runs}
        for run in runs:
            summary_runs.append(
                {
                    "family": family,
                    "run_id": run["run_id"],
                    "status": run["status"],
                    "source": run["source"],
                }
            )
            if run["status"] == RUN_SUCCESS_TEXT:
                success_text_count += 1

    missing_required = [
        family
        for family in evidence.required_families
        if not any(
            r["status"] in {RUN_SUCCESS_TEXT, RUN_SUCCESS_EMPTY}
            for r in families.get(family, {}).get("runs", [])
        )
    ]
    insufficient = bool(missing_required) or (
        success_text_count < int(evidence.min_successful_runs)
    )
    summary = {
        "families": list(evidence.model_families.keys()),
        "required_families": list(evidence.required_families),
        "min_successful_runs": evidence.min_successful_runs,
        "success_text_runs": success_text_count,
        "insufficient_evidence": insufficient,
        "missing_required_families": missing_required,
        "runs": summary_runs,
    }
    return families, summary


def evidence_fingerprint_payload(
    sample: Sample, evidence: EvidenceConfig
) -> dict[str, Any]:
    """Stable payload for cache identity (texts + statuses + metadata)."""
    families, summary = build_families_context(sample, evidence)
    # Avoid carrying mutable summary flags that depend only on thresholds already
    # in task config — keep texts/statuses/models.
    runs_payload = []
    for family, body in families.items():
        for run in body["runs"]:
            runs_payload.append(
                {
                    "family": family,
                    "run_id": run["run_id"],
                    "status": run["status"],
                    "text": run["text"],
                    "raw_text": run["raw_text"],
                    "model": run["model"],
                    "version": run["version"],
                    "source": run["source"],
                }
            )
    return {
        "sample": build_sample_context(sample, evidence.sample_context_fields),
        "runs": runs_payload,
        "mapping": evidence.model_families,
        "required_families": evidence.required_families,
        "min_successful_runs": evidence.min_successful_runs,
        "on_insufficient_evidence": evidence.on_insufficient_evidence,
        "insufficient_evidence": summary["insufficient_evidence"],
    }
