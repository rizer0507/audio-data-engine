"""Shared evaluation-set readiness inspection (CLI check / register)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from audio_engine.core.manifest import Manifest
from audio_engine.core.sample import Sample


def gold_text_of(sample: Sample, gold_field: str = "gold_text") -> str:
    text = str(sample.labels.get(gold_field) or "").strip()
    if not text and gold_field == "gold_text":
        text = str(sample.labels.get("label") or "").strip()
    if not text:
        text = str(sample.get_transcript_text("gold") or "").strip()
    return text


def type_of(sample: Sample, type_field: str = "type") -> str:
    value = sample.labels.get(type_field)
    if value is None or not str(value).strip():
        value = sample.labels.get("classification_bucket") or "(空)"
    return str(value).strip() or "(空)"


@dataclass
class EvalReadiness:
    path: Path
    total: int
    with_gold: list[str] = field(default_factory=list)
    without_gold: list[str] = field(default_factory=list)
    duplicates: list[str] = field(default_factory=list)
    empty_ids: int = 0
    unique_ids: int = 0
    missing_audio: list[str] = field(default_factory=list)
    type_counts: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def gold_ratio(self) -> float:
        return (len(self.with_gold) / self.total) if self.total else 0.0


def inspect_eval_manifest(
    path: Path,
    *,
    gold_field: str = "gold_text",
    type_field: str = "type",
    min_gold_ratio: float = 0.0,
    require_audio_key: str = "resampled_16k",
) -> EvalReadiness:
    """Inspect a Manifest for evaluation readiness. Does not raise on failures."""
    manifest = Manifest.load(path)
    samples = list(manifest)
    report = EvalReadiness(path=path, total=len(samples))
    if report.total == 0:
        report.errors.append(f"manifest is empty: {path}")
        return report

    seen: set[str] = set()
    for sample in samples:
        sample_id = sample.id
        if not str(sample_id).strip():
            report.empty_ids += 1
        elif sample_id in seen:
            if sample_id not in report.duplicates:
                report.duplicates.append(sample_id)
        else:
            seen.add(sample_id)

        if gold_text_of(sample, gold_field):
            report.with_gold.append(sample.id)
        else:
            report.without_gold.append(sample.id)

        key = type_of(sample, type_field)
        report.type_counts[key] = report.type_counts.get(key, 0) + 1

        if require_audio_key and require_audio_key not in sample.audio:
            report.missing_audio.append(sample.id)

    report.unique_ids = len(seen)
    if report.empty_ids:
        report.errors.append(f"{report.empty_ids} samples have empty id")
    if report.duplicates:
        report.errors.append(f"{len(report.duplicates)} duplicate ids")
    if require_audio_key and report.missing_audio:
        report.errors.append(
            f"{len(report.missing_audio)} samples missing audio key {require_audio_key!r}"
        )
    if report.gold_ratio < min_gold_ratio:
        report.errors.append(
            f"gold ratio {report.gold_ratio:.2%} < --min-gold-ratio {min_gold_ratio:.2%}"
        )
    return report
