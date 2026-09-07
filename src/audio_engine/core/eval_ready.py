"""Shared evaluation-set readiness inspection (CLI check / register)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from audio_engine.core.manifest import Manifest
from audio_engine.core.sample import Sample
from audio_engine.core.selection_v2.types import (
    EMPTY_GOLD_TYPES,
    LABEL_TIER_GOLD,
    PSEUDO_GOLD_TYPES,
)

TRUSTED_LABEL_SOURCES = frozenset({"human", "external"})
PSEUDO_LABEL_TIERS = frozenset({"pseudo_high", "pseudo_medium"})
PSEUDO_TYPES = frozenset(PSEUDO_GOLD_TYPES) | frozenset(
    {"auto_gold", "consensus_gold"}
)
# Unreviewed risk buckets must not serve as formal eval gold by themselves.
BLOCKED_UNVERIFIED_TYPES = frozenset(
    {
        "semantic_inversion",
        "critical_token_conflict",
        "semantic_sanitization",
        "possible_vad_miss",
        "short_utterance_risk",
    }
)


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


def label_tier_of(sample: Sample) -> str:
    return str(sample.labels.get("label_tier") or "").strip().lower()


def label_source_of(sample: Sample) -> str:
    raw = sample.labels.get("label_source") or sample.labels.get("gold_source") or ""
    return str(raw).strip().lower()


def is_human_verified(sample: Sample) -> bool:
    value = sample.labels.get("is_human_verified")
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def is_trusted_external(sample: Sample) -> bool:
    """External gold with explicit trusted marker or gold_mode=external."""
    if str(sample.labels.get("type") or "").strip() == "trusted_external_gold":
        return True
    if str(sample.labels.get("gold_mode") or "").strip().lower() == "external":
        return True
    if label_source_of(sample) == "external" and label_tier_of(sample) == LABEL_TIER_GOLD:
        return True
    # Legacy external_gold_v1 path: gold_source=external
    if str(sample.labels.get("gold_source") or "").strip().lower() == "external":
        return True
    return False


def is_formal_gold_sample(sample: Sample) -> bool:
    """Whether a sample may contribute gold_text to a formal eval Release."""
    if is_trusted_external(sample):
        return True
    state = str(sample.labels.get("annotation_state") or "").strip().lower()
    if state == "human_accepted":
        return True
    if is_human_verified(sample):
        return True
    if label_tier_of(sample) == LABEL_TIER_GOLD and label_source_of(sample) in TRUSTED_LABEL_SOURCES:
        return True
    return False


def is_pseudo_gold_sample(sample: Sample) -> bool:
    if is_formal_gold_sample(sample):
        return False
    tier = label_tier_of(sample)
    if tier in PSEUDO_LABEL_TIERS:
        return True
    bucket = type_of(sample)
    if bucket in PSEUDO_TYPES:
        return True
    return False


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
    subtype_counts: dict[str, int] = field(default_factory=dict)
    label_tier_counts: dict[str, int] = field(default_factory=dict)
    label_source_counts: dict[str, int] = field(default_factory=dict)
    release_version: str | None = None
    eval_trust: str = "unknown"
    pseudo_gold_ids: list[str] = field(default_factory=list)
    unverified_risk_ids: list[str] = field(default_factory=list)
    leak_audio_ids: list[str] = field(default_factory=list)
    leak_duplicate_groups: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def gold_ratio(self) -> float:
        return (len(self.with_gold) / self.total) if self.total else 0.0


def _sample_release_version(sample: Sample) -> str | None:
    for key in ("release_version", "eval_release", "dataset_release"):
        value = sample.labels.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def inspect_eval_manifest(
    path: Path,
    *,
    gold_field: str = "gold_text",
    type_field: str = "type",
    min_gold_ratio: float = 0.0,
    require_audio_key: str = "resampled_16k",
    require_formal_gold: bool = False,
    allow_pseudo_gold: bool = False,
    train_manifest: Path | None = None,
) -> EvalReadiness:
    """Inspect a Manifest for evaluation readiness. Does not raise on failures."""
    manifest = Manifest.load(path)
    samples = list(manifest)
    report = EvalReadiness(path=path, total=len(samples))
    if report.total == 0:
        report.errors.append(f"manifest is empty: {path}")
        return report

    release_versions: set[str] = set()
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
        subtype = str(sample.labels.get("subtype") or "").strip() or "(空)"
        report.subtype_counts[subtype] = report.subtype_counts.get(subtype, 0) + 1
        tier = label_tier_of(sample) or "(空)"
        report.label_tier_counts[tier] = report.label_tier_counts.get(tier, 0) + 1
        source = label_source_of(sample) or "(空)"
        report.label_source_counts[source] = report.label_source_counts.get(source, 0) + 1

        rv = _sample_release_version(sample)
        if rv:
            release_versions.add(rv)

        if is_pseudo_gold_sample(sample):
            report.pseudo_gold_ids.append(sample.id)

        bucket = type_of(sample, type_field)
        if bucket in BLOCKED_UNVERIFIED_TYPES and not is_formal_gold_sample(sample):
            # Only flag if it still carries gold_text pretending to be formal.
            if gold_text_of(sample, gold_field):
                report.unverified_risk_ids.append(sample.id)

        if require_audio_key and require_audio_key not in sample.audio:
            report.missing_audio.append(sample.id)

        # Empty-gold allowed types: track but do not treat as missing-gold error here.
        if (
            not gold_text_of(sample, gold_field)
            and bucket not in EMPTY_GOLD_TYPES
            and bucket not in {"noise", "invalid_audio", "true_silence"}
        ):
            pass

    report.unique_ids = len(seen)
    if len(release_versions) == 1:
        report.release_version = next(iter(release_versions))
    elif len(release_versions) > 1:
        report.release_version = ",".join(sorted(release_versions))
        report.warnings.append(
            f"multiple release_version values in manifest: {sorted(release_versions)}"
        )

    if allow_pseudo_gold:
        report.eval_trust = "pseudo_debug"
    elif report.pseudo_gold_ids and require_formal_gold:
        report.eval_trust = "rejected_pseudo"
    elif require_formal_gold:
        report.eval_trust = "formal"
    else:
        # Heuristic for check display
        if report.pseudo_gold_ids and not any(
            is_formal_gold_sample(s) for s in samples[: min(50, len(samples))]
        ):
            report.eval_trust = "legacy_or_pseudo"
        else:
            report.eval_trust = "legacy_compatible"

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

    if require_formal_gold and not allow_pseudo_gold:
        if report.pseudo_gold_ids:
            report.errors.append(
                f"{len(report.pseudo_gold_ids)} pseudo-gold samples "
                f"(label_tier in pseudo_* / type in auto_gold|consensus_gold|pseudo_gold_*); "
                f"formal eval register rejected (pass --allow-pseudo-gold for debug only)"
            )
        # Every sample with gold_text must be formal; empty gold only for allowlisted types.
        informal: list[str] = []
        empty_ok = EMPTY_GOLD_TYPES | {"true_silence", "invalid_audio", "noise"}
        for sample in samples:
            bucket = type_of(sample, type_field)
            has_gold = bool(gold_text_of(sample, gold_field))
            if has_gold and not is_formal_gold_sample(sample):
                informal.append(sample.id)
            elif not has_gold and bucket not in empty_ok:
                informal.append(sample.id)
        seen_inf: set[str] = set()
        informal_unique: list[str] = []
        for item in informal:
            if item not in seen_inf:
                seen_inf.add(item)
                informal_unique.append(item)
        if informal_unique:
            report.errors.append(
                f"{len(informal_unique)} samples lack formal gold "
                f"(need human_accepted / is_human_verified / trusted external, "
                f"or empty-gold allowlist type); preview={informal_unique[:10]}"
            )
        if report.unverified_risk_ids:
            report.errors.append(
                f"{len(report.unverified_risk_ids)} unverified risk-bucket samples "
                f"still carry gold_text (semantic_inversion etc. must be human-verified)"
            )

    if allow_pseudo_gold:
        report.warnings.append(
            "eval_trust=pseudo_debug: pseudo-gold allowed; do not write formal stage3 reports"
        )

    if train_manifest is not None:
        _check_train_leakage(report, samples, train_manifest)

    return report


def _check_train_leakage(
    report: EvalReadiness,
    eval_samples: list[Sample],
    train_path: Path,
) -> None:
    try:
        train = Manifest.load(train_path)
    except Exception as exc:  # noqa: BLE001 — surface as readiness error
        report.errors.append(f"failed to load train manifest for leak check: {exc}")
        return
    train_ids = {sample.id for sample in train if str(sample.id).strip()}
    train_dups: set[str] = set()
    for sample in train:
        dup = str(sample.labels.get("duplicate_group_id") or "").strip()
        if dup:
            train_dups.add(dup)

    for sample in eval_samples:
        if sample.id in train_ids:
            report.leak_audio_ids.append(sample.id)
        dup = str(sample.labels.get("duplicate_group_id") or "").strip()
        if dup and dup in train_dups:
            report.leak_duplicate_groups.append(dup)

    if report.leak_audio_ids:
        report.errors.append(
            f"train/eval audio_id leakage: {len(report.leak_audio_ids)} ids "
            f"(preview={report.leak_audio_ids[:10]})"
        )
    if report.leak_duplicate_groups:
        unique = sorted(set(report.leak_duplicate_groups))
        report.errors.append(
            f"train/eval duplicate_group_id leakage: {len(unique)} groups "
            f"(preview={unique[:10]})"
        )
