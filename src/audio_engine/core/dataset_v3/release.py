"""Atomic dataset_policy_v3 Release publish (stage D).

Gates: dual-reviewed eval gold, cross-split leakage, quota shortfalls,
pseudo-audit for pseudo_high train. Staging → validate → catalog → publish.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from audio_engine.core.artifacts import atomic_write_json
from audio_engine.core.catalog import (
    ArtifactCatalog,
    DatasetRelease,
    ProducerRecord,
    current_git_commit,
    is_dataset_policy_v3,
    register_manifest_output,
)
from audio_engine.core.dataset_v3.reservation import ReservationArtifact
from audio_engine.core.dataset_v3.sampling import (
    SamplingConfig,
    SamplingPlan,
    apply_sampling_plan_to_samples,
    build_sampling_plan,
)
from audio_engine.core.manifest import Manifest, file_sha256
from audio_engine.core.sample import Sample
from audio_engine.core.selection_v3.types import (
    RESERVATION_EVAL_CORE_RESERVE,
    RESERVATION_EVAL_RANDOM,
)


class ReleaseBuildError(ValueError):
    """Formal Release refused (shortfall, leakage, incomplete annotation, etc.)."""


@dataclass
class LeakageReport:
    ok: bool
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    blocked_eval_reserve_in_train: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "conflicts": list(self.conflicts),
            "blocked_eval_reserve_in_train": list(self.blocked_eval_reserve_in_train),
            "notes": list(self.notes),
        }


@dataclass
class ReleasePublishResult:
    release_id: str
    release_dir: Path
    sampling_digest: str
    reservation_digest: str | None
    counts: dict[str, int]
    outputs: dict[str, str]
    idempotent_hit: bool = False
    leakage: LeakageReport | None = None
    shortfalls: list[dict[str, Any]] = field(default_factory=list)
    release: DatasetRelease | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "release_id": self.release_id,
            "release_dir": str(self.release_dir),
            "sampling_digest": self.sampling_digest,
            "reservation_digest": self.reservation_digest,
            "counts": dict(self.counts),
            "outputs": dict(self.outputs),
            "idempotent_hit": self.idempotent_hit,
            "leakage": self.leakage.to_dict() if self.leakage else None,
            "shortfalls": list(self.shortfalls),
        }


def _group_of(sample: Sample, reservation: ReservationArtifact | None) -> str:
    if reservation is not None and sample.id in reservation.group_mapping:
        return reservation.group_mapping[sample.id]
    return str(
        sample.labels.get("leakage_group_id")
        or sample.labels.get("duplicate_group_id")
        or sample.id
    )


def _content_keys(sample: Sample) -> set[str]:
    keys: set[str] = set()
    for field_name in (
        "original_audio_sha256",
        "pcm_sha256",
        "normalized_pcm_sha256",
        "source_audio_id",
        "near_duplicate_group_id",
        "call_id",
        "conversation_id",
    ):
        value = sample.labels.get(field_name)
        if value:
            keys.add(f"{field_name}:{value}")
    if sample.sha256:
        keys.add(f"sha256:{sample.sha256}")
    return keys


def validate_cross_split_leakage(
    stamped: Sequence[Sample],
    plan: SamplingPlan,
    reservation: ReservationArtifact | None,
) -> LeakageReport:
    """Block ID / hash / source / call / confirmed near-dup / leakage-group overlap."""
    by_id = {s.id: s for s in stamped}
    split_ids = {
        "eval_random": set(plan.eval_random_ids),
        "eval_core": set(plan.eval_core_ids),
        "dev": set(plan.dev_ids),
        "train": set(plan.train_ids),
    }
    report = LeakageReport(ok=True)

    # Direct ID overlap
    names = list(split_ids)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = split_ids[left] & split_ids[right]
            if overlap:
                report.ok = False
                report.conflicts.append(
                    {
                        "type": "sample_id_overlap",
                        "left": left,
                        "right": right,
                        "ids": sorted(overlap)[:50],
                        "count": len(overlap),
                    }
                )

    # Leakage group overlap across formal splits
    groups: dict[str, dict[str, set[str]]] = {
        name: {} for name in names
    }
    for name, ids in split_ids.items():
        for sid in ids:
            sample = by_id.get(sid)
            if sample is None:
                continue
            gid = _group_of(sample, reservation)
            groups[name].setdefault(gid, set()).add(sid)

    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            shared = set(groups[left]) & set(groups[right])
            if shared:
                report.ok = False
                report.conflicts.append(
                    {
                        "type": "leakage_group_overlap",
                        "left": left,
                        "right": right,
                        "groups": sorted(shared)[:50],
                        "count": len(shared),
                    }
                )

    # Content-key overlap (file/pcm hash, source_audio, call, confirmed near-dup)
    key_owners: dict[str, dict[str, str]] = {}
    for name, ids in split_ids.items():
        for sid in ids:
            sample = by_id.get(sid)
            if sample is None:
                continue
            for key in _content_keys(sample):
                prev = key_owners.get(key)
                if prev and prev["split"] != name:
                    report.ok = False
                    report.conflicts.append(
                        {
                            "type": "content_key_overlap",
                            "key": key,
                            "left": prev["split"],
                            "left_id": prev["id"],
                            "right": name,
                            "right_id": sid,
                        }
                    )
                else:
                    key_owners[key] = {"split": name, "id": sid}

    # Eval reserve (selected or not) must not appear in train/dev
    if reservation is not None:
        reserved_eval_groups = {
            gid
            for gid, role in reservation.group_role.items()
            if role in {RESERVATION_EVAL_RANDOM, RESERVATION_EVAL_CORE_RESERVE}
        }
        for name in ("train", "dev"):
            for sid in split_ids[name]:
                sample = by_id.get(sid)
                if sample is None:
                    continue
                gid = _group_of(sample, reservation)
                if gid in reserved_eval_groups:
                    report.ok = False
                    report.blocked_eval_reserve_in_train.append(sid)
        if report.blocked_eval_reserve_in_train:
            report.conflicts.append(
                {
                    "type": "eval_reserve_group_in_train_or_dev",
                    "ids": report.blocked_eval_reserve_in_train[:50],
                    "count": len(report.blocked_eval_reserve_in_train),
                }
            )
        report.notes.append(
            "training checks the full frozen eval reserve groups, not only selected IDs"
        )

    return report


def _split_manifest(stamped: Sequence[Sample], split: str) -> Manifest:
    return Manifest([s for s in stamped if s.labels.get("split") == split])


def _content_fingerprint(paths: dict[str, Path], meta: dict[str, Any]) -> str:
    pieces: dict[str, Any] = {"meta": meta, "files": {}}
    for name, path in sorted(paths.items()):
        if path.is_file():
            pieces["files"][name] = file_sha256(path)
    raw = json.dumps(pieces, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _publish_release_v3(
    samples: Sequence[Sample],
    *,
    config: SamplingConfig,
    reservation: ReservationArtifact | None,
    catalog_dir: str | Path,
    output_dir: str | Path = "data/releases",
    source_artifact_id: str | None = None,
    source_manifest_path: str | Path | None = None,
    run_dir: str | Path | None = None,
    audit_report: dict[str, Any] | None = None,
    plan: SamplingPlan | None = None,
) -> ReleasePublishResult:
    """Sample → gate → stage → catalog → atomic publish.

    Idempotent: same release_id + identical content returns existing reference.
    Different content for an existing release_id fails fast without overwrite.
    """
    if not config.release_id:
        raise ReleaseBuildError("release_id is required")
    ArtifactCatalog._validate_name(config.release_id)
    if reservation is None:
        raise ReleaseBuildError("formal v3 Release requires frozen reservation")
    if not is_dataset_policy_v3(config.policy_version):
        raise ReleaseBuildError(
            f"publish_release_v3 requires dataset_policy_v3.*, got {config.policy_version!r}"
        )

    catalog = ArtifactCatalog(catalog_dir)
    release_root = Path(output_dir)
    final_dir = release_root / config.release_id
    run_path = Path(run_dir) if run_dir else release_root / ".runs" / config.release_id / uuid.uuid4().hex
    run_path.mkdir(parents=True, exist_ok=True)

    # Resolve / register source artifact
    if source_artifact_id:
        source_id = source_artifact_id
        catalog.get(source_id, verify=True)
    elif source_manifest_path:
        source_id = catalog.register_file(
            Path(source_manifest_path),
            kind="manifest",
            producer=ProducerRecord(pipeline="build_dataset_v3", run_id=run_path.name),
            metadata={"sample_count": len(samples)},
        ).artifact_id
    else:
        # Synthetic source from in-memory digest
        from audio_engine.core.dataset_v3.audit_plan import digest_payload
        source_digest = digest_payload([s.model_dump(mode="json") for s in samples])
        tmp_source = run_path / f"source_snapshot_{source_digest}.jsonl"
        Manifest(list(samples)).save(tmp_source)
        source_id = catalog.register_file(
            tmp_source,
            kind="manifest",
            producer=ProducerRecord(pipeline="build_dataset_v3", run_id=run_path.name),
            metadata={"sample_count": len(samples)},
        ).artifact_id

    sampling_plan = build_sampling_plan(samples, reservation, config)
    if plan is not None and plan.compute_digest() != sampling_plan.compute_digest():
        raise ReleaseBuildError("supplied sampling plan differs from recomputed plan")
    stamped = apply_sampling_plan_to_samples(samples, sampling_plan, reservation)
    leakage = validate_cross_split_leakage(stamped, sampling_plan, reservation)

    shortfall_dicts = [s.to_dict() for s in sampling_plan.shortfalls]
    blocking_reasons: list[str] = []
    from audio_engine.core.annotation_v3.gold import has_formal_gold_evidence
    eval_ids = set(sampling_plan.eval_core_ids + sampling_plan.eval_random_ids)
    if any(s.id in eval_ids and not has_formal_gold_evidence(s, require_dual=True) for s in samples):
        blocking_reasons.append("incomplete_eval_gold_evidence")
    from audio_engine.core.annotation_v3.queue import spot_check_stratum
    import math
    selected_single_layers = {spot_check_stratum(s) for s in samples
        if s.id in sampling_plan.train_ids and s.labels.get("annotation_state") == "annotated"}
    for layer in selected_single_layers:
        pool = [s for s in samples if spot_check_stratum(s) == layer
                and s.labels.get("annotation_state") == "annotated"
                and reservation.sample_role.get(s.id) == "train_pool"]
        checked = [s for s in pool if s.labels.get("spot_check_passed")
                   and s.labels.get("spot_checker_id") and s.labels["spot_checker_id"] != s.labels.get("annotator_id")]
        if len(checked) < math.ceil(len(pool) * .10):
            blocking_reasons.append(f"spot_check_incomplete:{layer}")
    from audio_engine.core.dataset_v3.audit_plan import validate_publish_audit
    pseudo_ids = {sid for sid in sampling_plan.train_ids
                  if sampling_plan.train_pool_assignment.get(sid) == "pseudo_high_audited"}
    if pseudo_ids:
        try:
            validate_publish_audit([s for s in samples if s.id in pseudo_ids], audit_report)
        except ValueError as exc:
            blocking_reasons.append(f"pseudo_audit_evidence:{exc}")
    if sampling_plan.has_blocking_shortfall():
        blocking_reasons.append("quota_shortfall")
    if not leakage.ok:
        blocking_reasons.append("leakage_conflict")
    if audit_report is not None:
        if audit_report.get("stop_publish") or not audit_report.get("passed", True):
            # Pseudo train may still be empty; only block if plan includes pseudo
            if any(
                sampling_plan.train_pool_assignment.get(sid) == "pseudo_high_audited"
                for sid in sampling_plan.train_ids
            ):
                blocking_reasons.append("pseudo_audit_failed")

    if blocking_reasons:
        # Write diagnostic reports under run_dir only — never a half release.
        atomic_write_json(run_path / "sampling.json", sampling_plan.to_dict())
        atomic_write_json(run_path / "leakage_report.json", leakage.to_dict())
        if audit_report is not None:
            atomic_write_json(run_path / "audit_report.json", audit_report)
        atomic_write_json(
            run_path / "build_failed.json",
            {
                "release_id": config.release_id,
                "reasons": blocking_reasons,
                "shortfalls": shortfall_dicts,
                "leakage_ok": leakage.ok,
            },
        )
        raise ReleaseBuildError(
            "formal Release refused: "
            + ", ".join(blocking_reasons)
            + f"; shortfalls={len(shortfall_dicts)}; leakage_conflicts={len(leakage.conflicts)}"
        )

    # Stage under a unique directory; publish only after full validation.
    staging = release_root / f".staging_{config.release_id}_{uuid.uuid4().hex}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    published = False
    registered_outputs: list[str] = []

    try:
        split_paths: dict[str, Path] = {}
        counts: dict[str, int] = {}
        for split in ("train", "dev", "eval_core", "eval_random", "excluded"):
            manifest = _split_manifest(stamped, split)
            parquet_path = staging / f"{split}.parquet"
            jsonl_path = staging / f"{split}.jsonl"
            manifest.save(parquet_path)
            manifest.save(jsonl_path)
            split_paths[split] = parquet_path
            counts[split] = len(manifest)

        sampling_path = staging / "sampling.json"
        leakage_path = staging / "leakage_report.json"
        audit_path = staging / "audit_report.json"
        release_json_path = staging / "release.json"
        atomic_write_json(sampling_path, sampling_plan.to_dict())
        atomic_write_json(leakage_path, leakage.to_dict())
        if audit_report is not None:
            atomic_write_json(audit_path, audit_report)
        elif config.require_pseudo_audit_for_pseudo_train:
            atomic_write_json(
                audit_path,
                {
                    "passed": None,
                    "note": "not_applicable: no pseudo train samples selected",
                    "stop_publish": False,
                },
            )

        meta = {
            "release_id": config.release_id,
            "policy_version": config.policy_version,
            "sampling_digest": sampling_plan.sampling_digest,
            "reservation_digest": reservation.content_digest if reservation else None,
            "counts": counts,
            "train_size": config.train_size,
            "sampling_seed": config.sampling_seed,
            "config": asdict(config),
        }
        fingerprint_paths = {**split_paths, "audit_report": audit_path,
                             "sampling_report": sampling_path, "leakage_report": leakage_path}
        fingerprint_paths.update({f"{split}_jsonl": staging / f"{split}.jsonl" for split in split_paths})
        fingerprint = _content_fingerprint(fingerprint_paths, meta)

        # Idempotency: existing release with same fingerprint → return reference
        if final_dir.exists() and (final_dir / "release.json").is_file():
            existing_meta = json.loads((final_dir / "release.json").read_text(encoding="utf-8"))
            existing_fp = existing_meta.get("content_fingerprint")
            if existing_fp == fingerprint:
                existing_paths = {name: final_dir / path.name for name, path in fingerprint_paths.items()}
                if _content_fingerprint(existing_paths, meta) != fingerprint:
                    raise ReleaseBuildError("existing release files failed content verification")
                try:
                    existing_release = catalog.get_release(config.release_id)
                except KeyError as exc:
                    raise ReleaseBuildError("existing release missing catalog record; repair required") from exc
                return ReleasePublishResult(
                    release_id=config.release_id,
                    release_dir=final_dir,
                    sampling_digest=sampling_plan.sampling_digest,
                    reservation_digest=reservation.content_digest if reservation else None,
                    counts=dict(existing_meta.get("counts") or counts),
                    outputs=dict(existing_release.outputs) if existing_release else {},
                    idempotent_hit=True,
                    leakage=leakage,
                    shortfalls=shortfall_dicts,
                    release=existing_release,
                )
            raise ReleaseBuildError(
                f"release_id {config.release_id!r} already exists with different content; "
                "refusing overwrite (fail-fast)"
            )

        # Also check catalog-only collision
        try:
            existing_release = catalog.get_release(config.release_id)
        except KeyError:
            existing_release = None
        if existing_release is not None:
            # Catalog says it exists — verify file fingerprint if present
            raise ReleaseBuildError(
                f"release_id {config.release_id!r} already registered in catalog; "
                "refusing different content"
            )

        release_payload = {
            **meta,
            "content_fingerprint": fingerprint,
            "normalization_version": config.normalization_version,
            "gold_revision": config.gold_revision,
            "group_key": config.group_key,
            "git_commit": current_git_commit(),
            "splits": list(split_paths),
        }
        atomic_write_json(release_json_path, release_payload)

        # Register manifests from staging paths (will remain valid after rename)
        # Publish staging → final first so URIs point at final locations.
        release_root.mkdir(parents=True, exist_ok=True)
        if final_dir.exists():
            raise ReleaseBuildError(f"release dir unexpectedly exists: {final_dir}")
        staging.rename(final_dir)
        published = True

        outputs: dict[str, str] = {}
        for split in ("train", "dev", "eval_core", "eval_random", "excluded"):
            path = final_dir / f"{split}.parquet"
            record = register_manifest_output(
                path,
                catalog_dir=catalog_dir,
                pipeline="build_dataset_v3",
                run_dir=run_path / split,
                sample_count=counts[split],
            )
            outputs[split] = record.artifact_id
            registered_outputs.append(record.artifact_id)

        release = catalog.put_release(
            DatasetRelease(
                release_id=config.release_id,
                source_artifact_id=source_id,
                outputs={
                    "train": outputs["train"],
                    "dev": outputs["dev"],
                    "eval_core": outputs["eval_core"],
                    "eval_random": outputs["eval_random"],
                    "excluded": outputs["excluded"],
                },
                policy_version=config.policy_version,
                normalization_version=config.normalization_version,
                gold_revision=config.gold_revision,
                split_seed=config.sampling_seed,
                group_key=config.group_key,
                counts=counts,
                git_commit=current_git_commit(),
                sampling_digest=sampling_plan.sampling_digest,
                reservation_digest=reservation.content_digest if reservation else None,
                leakage_report_uri=str((final_dir / "leakage_report.json").resolve()),
                sampling_report_uri=str((final_dir / "sampling.json").resolve()),
                audit_report_uri=str((final_dir / "audit_report.json").resolve()),
            )
        )
        # Refresh release.json with artifact ids
        release_payload["outputs"] = outputs
        atomic_write_json(final_dir / "release.json", release_payload)

        return ReleasePublishResult(
            release_id=config.release_id,
            release_dir=final_dir,
            sampling_digest=sampling_plan.sampling_digest,
            reservation_digest=reservation.content_digest if reservation else None,
            counts=counts,
            outputs=outputs,
            idempotent_hit=False,
            leakage=leakage,
            shortfalls=shortfall_dicts,
            release=release,
        )
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        if published:
            # This release ID is exclusively locked by the public entry point.
            # Roll back records created by this attempt together with its files.
            (catalog.releases_dir / f"{config.release_id}.json").unlink(missing_ok=True)
            for artifact_id in registered_outputs:
                (catalog.records_dir / f"{artifact_id}.json").unlink(missing_ok=True)
            shutil.rmtree(final_dir, ignore_errors=True)
        raise


def publish_release_v3(samples: Sequence[Sample], *, config: SamplingConfig, **kwargs) -> ReleasePublishResult:
    """Serialize writers for one release ID; never publish over a competing run."""
    ArtifactCatalog._validate_name(config.release_id)
    root = Path(kwargs.get("output_dir", "data/releases"))
    lock = root / ".locks" / f"{config.release_id}.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = lock.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise ReleaseBuildError(f"release publication locked: {lock}") from exc
    try:
        with handle:
            import os
            handle.write(json.dumps({"pid": os.getpid(), "release_id": config.release_id}))
            handle.flush()
            return _publish_release_v3(sorted(samples, key=lambda s: s.id), config=config, **kwargs)
    finally:
        lock.unlink(missing_ok=True)
