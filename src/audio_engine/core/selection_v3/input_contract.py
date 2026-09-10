"""Eight-route input contract: align, status, conservation (selection_v3 stage A)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from audio_engine.core.sample import Sample
from audio_engine.core.selection_v3.config import SelectionV3Config
from audio_engine.core.selection_v3.types import (
    RUN_STATUS_FAILED,
    RUN_STATUS_MISSING,
    RUN_STATUS_SUCCESS_EMPTY,
    RUN_STATUS_SUCCESS_TEXT,
    SAMPLE_CLASSIFIABLE,
    SAMPLE_INFERENCE_INCOMPLETE,
    SAMPLE_INVALID_AUDIO,
)


def original_audio_sha256(sample: Sample) -> str:
    """Original (unpadded) audio content hash when available."""
    for container in (sample.labels, sample.quality):
        for key in ("original_audio_sha256", "source_audio_sha256", "raw_audio_sha256"):
            value = container.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
    # Lineage / provenance nested dicts
    provenance = sample.labels.get("provenance")
    if isinstance(provenance, dict):
        for key in ("original_audio_sha256", "source_audio_sha256"):
            value = provenance.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
    return str(sample.sha256 or "").strip()


def join_key(sample: Sample) -> tuple[str, str]:
    """Canonical join key: sample id + original audio hash (not pad-file hash alone)."""
    return (str(sample.id).strip(), original_audio_sha256(sample))


def _transcript_entry(sample: Sample, key: str) -> dict[str, Any] | None:
    entry = sample.transcripts.get(key)
    if entry is None:
        return None
    if isinstance(entry, dict):
        return entry
    return {"text": str(entry)}


def _explicit_failed(entry: dict[str, Any], sample: Sample, key: str) -> bool:
    status = str(entry.get("status") or "").strip().lower()
    if status in {"failed", "error", "fail"}:
        return True
    extra = entry.get("extra") if isinstance(entry.get("extra"), dict) else {}
    inf = str(extra.get("inference_status") or extra.get("status") or "").strip().lower()
    if inf in {"failed", "error", "fail"}:
        return True
    # Operator-level failure recorded against this transcript key / asr family
    for err_key, err_val in (sample.errors or {}).items():
        if not err_val:
            continue
        lowered = str(err_key).lower()
        if key.lower() in lowered or "asr" in lowered:
            # Only treat as this run's failure when status also marks failed
            if sample.status.get(err_key) == "failed":
                return True
    if entry.get("failed") is True:
        return True
    return False


def classify_run_status(sample: Sample, transcript_key: str) -> str:
    """Map one ASR route to success_text | success_empty | failed | missing."""
    entry = _transcript_entry(sample, transcript_key)
    if entry is None:
        return RUN_STATUS_MISSING
    if _explicit_failed(entry, sample, transcript_key):
        return RUN_STATUS_FAILED
    if str(entry.get("status") or "").lower() in {"missing", "pending", "running"}:
        return RUN_STATUS_MISSING
    text = entry.get("text")
    if text is None:
        return RUN_STATUS_MISSING
    from audio_engine.core.selection_v3.text import raw_transcript_text
    if raw_transcript_text(entry).strip() == "":
        return RUN_STATUS_SUCCESS_EMPTY
    return RUN_STATUS_SUCCESS_TEXT


def is_physically_invalid(sample: Sample) -> bool:
    if sample.labels.get("broken") is True:
        return True
    if sample.labels.get("invalid_audio") is True:
        return True
    duration = sample.duration
    if duration is not None and float(duration) <= 0:
        return True
    return False


@dataclass
class SampleContractResult:
    sample_id: str
    original_audio_sha256: str
    readiness: str
    run_statuses: dict[str, str] = field(default_factory=dict)
    missing_runs: list[str] = field(default_factory=list)
    failed_runs: list[str] = field(default_factory=list)
    reason_codes: list[str] = field(default_factory=list)

    def to_labels(self) -> dict[str, Any]:
        return {
            "contract_readiness": self.readiness,
            "run_statuses": dict(self.run_statuses),
            "original_audio_sha256": self.original_audio_sha256,
            "inference_incomplete": self.readiness == SAMPLE_INFERENCE_INCOMPLETE,
            "contract_reason_codes": list(self.reason_codes),
        }


@dataclass
class ConservationReport:
    input_count: int = 0
    physically_invalid: int = 0
    inference_incomplete: int = 0
    classifiable: int = 0
    quality_score_failed: int = 0  # cross-stat placeholder for later sidecar merge
    extra_ids_from_inference: list[str] = field(default_factory=list)
    retry_list: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def assert_conserved(self) -> None:
        total = self.physically_invalid + self.inference_incomplete + self.classifiable
        if total != self.input_count:
            raise ValueError(
                f"conservation broken: input={self.input_count} != "
                f"invalid({self.physically_invalid}) + incomplete({self.inference_incomplete}) "
                f"+ classifiable({self.classifiable}) = {total}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_count": self.input_count,
            "physically_invalid": self.physically_invalid,
            "inference_incomplete": self.inference_incomplete,
            "classifiable": self.classifiable,
            "quality_score_failed": self.quality_score_failed,
            "extra_ids_from_inference": list(self.extra_ids_from_inference),
            "retry_list_count": len(self.retry_list),
            "retry_list_preview": self.retry_list[:50],
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "conserved": (
                self.physically_invalid
                + self.inference_incomplete
                + self.classifiable
                == self.input_count
            ),
        }


@dataclass
class EightRouteAlignmentReport:
    """Integrity report for configured multi-run / multi-family join."""

    join_key: str = "id+original_audio_sha256"
    base_count: int = 0
    expected_runs: list[str] = field(default_factory=list)
    run_reports: list[dict[str, Any]] = field(default_factory=list)
    conservation: ConservationReport = field(default_factory=ConservationReport)
    aligned: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "join_key": self.join_key,
            "base_count": self.base_count,
            "expected_runs": list(self.expected_runs),
            "run_reports": list(self.run_reports),
            "conservation": self.conservation.to_dict(),
            "aligned": self.aligned,
        }


def validate_base_snapshot(samples: Iterable[Sample]) -> dict[str, Sample]:
    """Index audio snapshot by id; fail-fast on duplicate id or id→hash conflict."""
    by_id: dict[str, Sample] = {}
    id_to_hash: dict[str, str] = {}
    for sample in samples:
        sid = str(sample.id).strip()
        if not sid:
            raise ValueError("sample has empty id")
        ohash = original_audio_sha256(sample)
        if sid in by_id:
            raise ValueError(f"duplicate sample id in base snapshot: {sid}")
        if sid in id_to_hash and id_to_hash[sid] != ohash and id_to_hash[sid] and ohash:
            raise ValueError(
                f"same id {sid!r} maps to different original_audio_sha256: "
                f"{id_to_hash[sid]!r} vs {ohash!r}"
            )
        by_id[sid] = sample
        if ohash:
            id_to_hash[sid] = ohash
    return by_id


def align_run_manifest(
    base: dict[str, Sample],
    incoming: list[Sample],
    *,
    transcript_key: str,
    path: str = "",
    id_policy: str = "exact",
) -> dict[str, Any]:
    """Align one run manifest to the audio base by id + original audio hash."""
    if id_policy not in {"exact", "left"}:
        raise ValueError("id_policy must be 'exact' or 'left'")
    indexed: dict[str, Sample] = {}
    for sample in incoming:
        sid = str(sample.id).strip()
        if not sid:
            raise ValueError(f"manifest {path} contains empty id")
        if sid in indexed:
            raise ValueError(f"manifest {path} contains duplicate ids: {sid}")
        indexed[sid] = sample

    expected = set(base)
    missing = sorted(expected - indexed.keys())
    extra = sorted(indexed.keys() - expected)
    hash_mismatches: list[str] = []
    unchecked = 0
    for sid in sorted(expected & indexed.keys()):
        base_hash = original_audio_sha256(base[sid])
        join_hash = original_audio_sha256(indexed[sid])
        if base_hash and join_hash and base_hash != join_hash:
            hash_mismatches.append(sid)
        elif not base_hash or not join_hash:
            unchecked += 1

    report = {
        "transcript_key": transcript_key,
        "path": path,
        "count": len(incoming),
        "missing_ids": len(missing),
        "extra_ids": len(extra),
        "original_audio_sha256_mismatches": len(hash_mismatches),
        "original_audio_sha256_unchecked": unchecked,
        "missing_id_examples": missing[:20],
        "extra_id_examples": extra[:20],
        "hash_mismatch_examples": hash_mismatches[:20],
    }
    if id_policy == "exact" and (missing or extra):
        raise ValueError(
            f"run {transcript_key!r} ({path}) ids not aligned: "
            f"missing={len(missing)}, extra={len(extra)}"
        )
    if hash_mismatches:
        raise ValueError(
            f"run {transcript_key!r} ({path}) original_audio_sha256 mismatches="
            f"{len(hash_mismatches)}, examples={hash_mismatches[:5]}"
        )
    return report


def evaluate_sample_contract(
    sample: Sample,
    config: SelectionV3Config,
) -> SampleContractResult:
    """Classify one sample against the configured multi-run contract."""
    ohash = original_audio_sha256(sample)
    statuses: dict[str, str] = {}
    missing: list[str] = []
    failed: list[str] = []
    for key in config.all_transcript_keys():
        status = classify_run_status(sample, key)
        statuses[key] = status
        if status == RUN_STATUS_MISSING:
            missing.append(key)
        elif status == RUN_STATUS_FAILED:
            failed.append(key)

    reasons: list[str] = []
    if is_physically_invalid(sample):
        readiness = SAMPLE_INVALID_AUDIO
        reasons.append("physically_invalid")
    elif missing or failed:
        readiness = SAMPLE_INFERENCE_INCOMPLETE
        if missing:
            reasons.append("missing_runs")
        if failed:
            reasons.append("failed_runs")
    else:
        readiness = SAMPLE_CLASSIFIABLE

    return SampleContractResult(
        sample_id=sample.id,
        original_audio_sha256=ohash,
        readiness=readiness,
        run_statuses=statuses,
        missing_runs=missing,
        failed_runs=failed,
        reason_codes=reasons,
    )


def apply_contract_to_samples(
    samples: list[Sample],
    config: SelectionV3Config,
    *,
    quality_failed_ids: set[str] | None = None,
) -> tuple[list[Sample], ConservationReport, EightRouteAlignmentReport]:
    """Annotate samples with contract fields and build conservation report.

    Does not drop samples: incomplete rows stay as inference_incomplete with retry list.
    """
    config.validate_family_config()
    validate_base_snapshot(samples)

    report = ConservationReport(input_count=len(samples))
    alignment = EightRouteAlignmentReport(
        base_count=len(samples),
        expected_runs=config.all_transcript_keys(),
    )
    quality_failed = quality_failed_ids or set()
    updated: list[Sample] = []

    for source in samples:
        sample = source.model_copy(deep=True)
        result = evaluate_sample_contract(sample, config)
        sample.labels.update(result.to_labels())
        from dataclasses import asdict
        from audio_engine.core.dataset_v3.audit_plan import digest_payload
        sample.labels["run_identities_digest"] = digest_payload([asdict(r) for r in config.runs]) if config.runs else None
        sample.labels["run_identities_verified"] = False
        # Preserve stable provenance fields when present
        if result.original_audio_sha256 and not sample.labels.get("original_audio_sha256"):
            sample.labels["original_audio_sha256"] = result.original_audio_sha256

        if result.readiness == SAMPLE_INVALID_AUDIO:
            report.physically_invalid += 1
        elif result.readiness == SAMPLE_INFERENCE_INCOMPLETE:
            report.inference_incomplete += 1
            report.retry_list.append(
                {
                    "id": sample.id,
                    "original_audio_sha256": result.original_audio_sha256,
                    "missing_runs": list(result.missing_runs),
                    "failed_runs": list(result.failed_runs),
                }
            )
        else:
            report.classifiable += 1

        if sample.id in quality_failed:
            report.quality_score_failed += 1

        updated.append(sample)

    report.assert_conserved()
    alignment.conservation = report
    alignment.aligned = True
    return updated, report, alignment


def merge_field_by_join_key(
    base_samples: list[Sample],
    sidecar_by_key: dict[tuple[str, str], dict[str, Any]],
    *,
    target: str = "quality",
) -> list[Sample]:
    """Merge sidecar fields by (id, original_audio_sha256); never by row position."""
    if target not in {"quality", "labels"}:
        raise ValueError("target must be 'quality' or 'labels'")
    out: list[Sample] = []
    for source in base_samples:
        sample = source.model_copy(deep=True)
        key = join_key(sample)
        payload = sidecar_by_key.get(key)
        if payload:
            getattr(sample, target).update(payload)
        out.append(sample)
    return out
