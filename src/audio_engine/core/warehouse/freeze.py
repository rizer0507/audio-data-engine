"""Atomic batch-unique warehouse publish (033).

Distinct from dataset_v3 Release. One batch ↔ one formal warehouse_id.
Staging → Manifest-layer gates → catalog bind → atomic rename.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from audio_engine.core.annotation_v3.config import AnnotationConfig
from audio_engine.core.artifacts import atomic_write_json
from audio_engine.core.catalog import (
    ArtifactCatalog,
    BatchWarehouse,
    ProducerRecord,
    current_git_commit,
    register_manifest_output,
)
from audio_engine.core.manifest import Manifest, file_sha256
from audio_engine.core.sample import Sample
from audio_engine.core.source_naming import validate_source_name
from audio_engine.core.warehouse.completion import (
    WarehouseGateError,
    assert_batch_complete,
    assert_snapshot_alignment,
    final_category,
    is_explicitly_excluded,
    raise_if_incomplete,
)
from audio_engine.core.warehouse.identity import (
    classified_snapshot_digest,
    original_audio_sha,
    resolve_audio_ref,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def warehouse_id_for_batch(batch: str) -> str:
    batch = validate_source_name(batch)
    return f"wh_{batch}"


@dataclass
class WarehousePublishResult:
    warehouse_id: str
    batch: str
    warehouse_dir: Path
    content_fingerprint: str
    classified_digest: str
    counts: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)
    idempotent_hit: bool = False
    warehouse: BatchWarehouse | None = None
    gate_report: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "warehouse_id": self.warehouse_id,
            "batch": self.batch,
            "warehouse_dir": str(self.warehouse_dir),
            "content_fingerprint": self.content_fingerprint,
            "classified_digest": self.classified_digest,
            "counts": dict(self.counts),
            "outputs": dict(self.outputs),
            "idempotent_hit": self.idempotent_hit,
            "gate_report": dict(self.gate_report),
        }


def _enrich_for_warehouse(samples: Sequence[Sample]) -> list[Sample]:
    """Attach audio refs and final_category without mutating caller's objects."""
    out: list[Sample] = []
    for sample in samples:
        s = sample.model_copy(deep=True)
        ref = resolve_audio_ref(s)
        s.labels["warehouse_audio_key"] = ref["audio_key"]
        s.labels["warehouse_audio_path"] = ref["audio_path"]
        s.labels["warehouse_audio_content_sha256"] = ref["audio_content_sha256"]
        s.labels["warehouse_audio_size_bytes"] = ref["audio_size_bytes"]
        cat = final_category(s)
        if cat is not None:
            s.labels["final_category"] = cat
        s.labels["warehouse_excluded"] = bool(is_explicitly_excluded(s))
        if is_explicitly_excluded(s) and not s.labels.get("warehouse_keep_reason"):
            kind = str(s.labels.get("gold_kind") or "")
            state = str(s.labels.get("annotation_state") or "")
            s.labels["warehouse_keep_reason"] = kind or state or "explicit_exclude"
        out.append(s)
    return out


def _count_stats(samples: Sequence[Sample]) -> dict[str, Any]:
    by_final: dict[str, int] = {}
    by_state: dict[str, int] = {}
    by_kind: dict[str, int] = {}
    excluded = 0
    for s in samples:
        cat = str(s.labels.get("final_category") or final_category(s) or "_none_")
        by_final[cat] = by_final.get(cat, 0) + 1
        state = str(s.labels.get("annotation_state") or "_none_")
        by_state[state] = by_state.get(state, 0) + 1
        kind = str(s.labels.get("gold_kind") or "_none_")
        by_kind[kind] = by_kind.get(kind, 0) + 1
        if is_explicitly_excluded(s) or s.labels.get("warehouse_excluded"):
            excluded += 1
    return {
        "total": len(samples),
        "excluded_kept": excluded,
        "by_final_category": dict(sorted(by_final.items())),
        "by_annotation_state": dict(sorted(by_state.items())),
        "by_gold_kind": dict(sorted(by_kind.items())),
    }


def _content_fingerprint(manifest_path: Path, meta: dict[str, Any]) -> str:
    pieces = {
        "meta": {
            k: meta[k]
            for k in (
                "warehouse_id",
                "batch",
                "classified_digest",
                "classified_manifest",
                "rule_version",
                "annotation_version",
                "counts",
            )
            if k in meta
        },
        "manifest_sha256": file_sha256(manifest_path),
    }
    raw = json.dumps(pieces, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _verify_review_evidence(paths: Sequence[str | Path]) -> list[dict[str, str]]:
    refs: list[dict[str, str]] = []
    for raw in paths:
        path = Path(raw)
        if not path.is_file():
            raise WarehouseGateError(f"review evidence missing or not a file: {path}")
        refs.append(
            {
                "path": str(path),
                "sha256": file_sha256(path),
            }
        )
    return refs


def _publish_warehouse(
    reviewed: Sequence[Sample],
    *,
    batch: str,
    classified: Sequence[Sample],
    classified_path: str | Path,
    config: AnnotationConfig,
    allowed_categories: Iterable[str] | None,
    catalog_dir: str | Path,
    output_dir: str | Path,
    run_dir: str | Path | None,
    review_evidence: Sequence[str | Path] | None,
    rule_version: str | None,
    annotation_config_path: str | None,
) -> WarehousePublishResult:
    batch = validate_source_name(batch)
    warehouse_id = warehouse_id_for_batch(batch)
    ArtifactCatalog._validate_name(warehouse_id)

    classified_path = Path(classified_path)
    classified_digest = classified_snapshot_digest(classified)

    align_issues = assert_snapshot_alignment(classified, reviewed)
    if align_issues:
        preview = "; ".join(f"{i.sample_id}:{i.code}" for i in align_issues[:12])
        raise WarehouseGateError(
            f"warehouse freeze refused: snapshot alignment failed ({len(align_issues)}); {preview}"
        )

    gate = assert_batch_complete(
        reviewed,
        config=config,
        allowed_categories=allowed_categories,
        require_audio=True,
    )
    raise_if_incomplete(gate)

    evidence_refs = _verify_review_evidence(review_evidence or [])

    enriched = _enrich_for_warehouse(reviewed)
    counts = _count_stats(enriched)

    catalog = ArtifactCatalog(catalog_dir)
    root = Path(output_dir)
    final_dir = root / batch
    run_path = Path(run_dir) if run_dir else root / ".runs" / batch / uuid.uuid4().hex
    run_path.mkdir(parents=True, exist_ok=True)

    staging = root / f".staging_{batch}_{uuid.uuid4().hex}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    published = False
    registered_outputs: list[str] = []

    try:
        manifest_path = staging / "manifest.parquet"
        Manifest(list(enriched)).save(manifest_path)
        Manifest(list(enriched)).save(staging / "manifest.jsonl")

        meta: dict[str, Any] = {
            "warehouse_id": warehouse_id,
            "batch": batch,
            "frozen_at": _utc_now(),
            "classified_manifest": str(classified_path),
            "classified_digest": classified_digest,
            "rule_version": rule_version
            or next(
                (str(s.labels.get("rule_version") or "") for s in classified if s.labels.get("rule_version")),
                None,
            ),
            "annotation_version": config.annotation_version,
            "annotation_config": annotation_config_path,
            "review_evidence": evidence_refs,
            "counts": counts,
            "protocol": "warehouse_batch_v1",
            "git_commit": current_git_commit(),
        }
        fingerprint = _content_fingerprint(manifest_path, meta)
        meta["content_fingerprint"] = fingerprint
        meta["manifest_sha256"] = file_sha256(manifest_path)
        atomic_write_json(staging / "warehouse.json", meta)
        atomic_write_json(run_path / "gate_report.json", gate.to_dict())

        # Idempotent hit: same batch + same fingerprint.
        if final_dir.exists() and (final_dir / "warehouse.json").is_file():
            existing_meta = json.loads((final_dir / "warehouse.json").read_text(encoding="utf-8"))
            existing_fp = existing_meta.get("content_fingerprint")
            if existing_fp == fingerprint:
                existing_manifest = final_dir / "manifest.parquet"
                if not existing_manifest.is_file():
                    raise WarehouseGateError("existing warehouse missing manifest.parquet; repair required")
                if file_sha256(existing_manifest) != existing_meta.get("manifest_sha256"):
                    raise WarehouseGateError("existing warehouse manifest digest mismatch; refusing")
                try:
                    existing = catalog.get_warehouse_by_batch(batch)
                except KeyError as exc:
                    raise WarehouseGateError(
                        "existing warehouse dir present but catalog binding missing; repair required"
                    ) from exc
                if existing.content_fingerprint != fingerprint:
                    raise WarehouseGateError(
                        "catalog fingerprint diverges from on-disk warehouse; refusing"
                    )
                # Drop unused staging so failed/idempotent retries leave no consumable half-product.
                if staging.exists():
                    shutil.rmtree(staging, ignore_errors=True)
                return WarehousePublishResult(
                    warehouse_id=existing.warehouse_id,
                    batch=batch,
                    warehouse_dir=final_dir,
                    content_fingerprint=fingerprint,
                    classified_digest=classified_digest,
                    counts=dict(existing_meta.get("counts") or counts),
                    outputs={"manifest": existing.manifest_artifact_id},
                    idempotent_hit=True,
                    warehouse=existing,
                    gate_report=gate.to_dict(),
                )
            raise WarehouseGateError(
                f"batch {batch!r} already has a formal warehouse with different content; "
                "refusing overwrite (no multi-version warehouse in v1)"
            )

        try:
            existing = catalog.get_warehouse_by_batch(batch)
        except KeyError:
            existing = None
        if existing is not None:
            raise WarehouseGateError(
                f"batch {batch!r} already registered as warehouse {existing.warehouse_id!r}; "
                "refusing different content"
            )

        root.mkdir(parents=True, exist_ok=True)
        if final_dir.exists():
            raise WarehouseGateError(f"warehouse dir unexpectedly exists: {final_dir}")
        staging.rename(final_dir)
        published = True

        record = register_manifest_output(
            final_dir / "manifest.parquet",
            catalog_dir=catalog_dir,
            pipeline="warehouse.freeze",
            run_dir=run_path / "manifest",
            sample_count=len(enriched),
        )
        registered_outputs.append(record.artifact_id)

        warehouse = catalog.put_warehouse(
            BatchWarehouse(
                warehouse_id=warehouse_id,
                batch=batch,
                manifest_artifact_id=record.artifact_id,
                classified_digest=classified_digest,
                content_fingerprint=fingerprint,
                uri=str((final_dir / "warehouse.json").resolve()),
                counts={
                    "total": int(counts["total"]),
                    "excluded_kept": int(counts["excluded_kept"]),
                },
                git_commit=current_git_commit(),
                metadata={
                    "classified_manifest": str(classified_path),
                    "rule_version": meta.get("rule_version"),
                    "annotation_version": meta.get("annotation_version"),
                    "by_final_category": counts.get("by_final_category") or {},
                },
            )
        )
        meta["manifest_artifact_id"] = record.artifact_id
        atomic_write_json(final_dir / "warehouse.json", meta)

        return WarehousePublishResult(
            warehouse_id=warehouse_id,
            batch=batch,
            warehouse_dir=final_dir,
            content_fingerprint=fingerprint,
            classified_digest=classified_digest,
            counts=counts,
            outputs={"manifest": record.artifact_id},
            idempotent_hit=False,
            warehouse=warehouse,
            gate_report=gate.to_dict(),
        )
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        if published:
            (catalog.warehouses_dir / f"{batch}.json").unlink(missing_ok=True)
            for artifact_id in registered_outputs:
                (catalog.records_dir / f"{artifact_id}.json").unlink(missing_ok=True)
            shutil.rmtree(final_dir, ignore_errors=True)
        raise


def publish_warehouse(
    reviewed: Sequence[Sample],
    *,
    batch: str,
    classified: Sequence[Sample],
    classified_path: str | Path,
    config: AnnotationConfig,
    allowed_categories: Iterable[str] | None = None,
    catalog_dir: str | Path = "data/catalog",
    output_dir: str | Path = "datasets/stage1/warehouses",
    run_dir: str | Path | None = None,
    review_evidence: Sequence[str | Path] | None = None,
    rule_version: str | None = None,
    annotation_config_path: str | None = None,
) -> WarehousePublishResult:
    """Serialize writers for one batch; never publish a second formal warehouse."""
    batch = validate_source_name(batch)
    root = Path(output_dir)
    lock = root / ".locks" / f"{batch}.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = lock.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise WarehouseGateError(f"warehouse publication locked: {lock}") from exc
    try:
        with handle:
            handle.write(json.dumps({"pid": os.getpid(), "batch": batch}))
            handle.flush()
            return _publish_warehouse(
                sorted(reviewed, key=lambda s: s.id),
                batch=batch,
                classified=sorted(classified, key=lambda s: s.id),
                classified_path=classified_path,
                config=config,
                allowed_categories=allowed_categories,
                catalog_dir=catalog_dir,
                output_dir=output_dir,
                run_dir=run_dir,
                review_evidence=review_evidence,
                rule_version=rule_version,
                annotation_config_path=annotation_config_path,
            )
    finally:
        lock.unlink(missing_ok=True)
