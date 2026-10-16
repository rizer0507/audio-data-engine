"""Batch/input preflight for tempo augmentation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from audio_engine.core.augmentation.ffmpeg_atempo import atempo_filter_graph, probe_ffmpeg
from audio_engine.core.augmentation.params import TempoParams, resolve_tempo_params
from audio_engine.core.augmentation.recipe import resolve_product_layout
from audio_engine.core.manifest import Manifest
from audio_engine.core.sample import Sample
from audio_engine.core.source_naming import validate_source_name


class AugmentPreflightError(ValueError):
    """Raised when batch/input/params cannot start processing."""


def _batch_from_sample(sample: Sample) -> str | None:
    labels = sample.labels or {}
    for key in ("batch", "batch_id", "source_batch", "warehouse_batch"):
        value = labels.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def detect_already_augmented(sample: Sample) -> bool:
    labels = sample.labels or {}
    aug = labels.get("augmentation")
    if isinstance(aug, dict) and str(aug.get("type") or "").lower() == "tempo":
        return True
    # Derived id / batch naming from this pipeline.
    if str(sample.id).startswith("tempo_"):
        return True
    if "__tempo_" in str(labels.get("batch") or labels.get("derived_batch") or ""):
        return True
    return False


def validate_input_snapshot(
    samples: list[Sample],
    *,
    source_batch: str,
    input_manifest: Path,
    allow_missing_batch_field: bool = True,
) -> dict[str, Any]:
    batch = validate_source_name(source_batch)
    if not samples:
        raise AugmentPreflightError(
            f"input snapshot is empty ({input_manifest}); nothing to augment "
            "(do not mark as successful augmented set)"
        )

    ids = [s.id for s in samples]
    if len(ids) != len(set(ids)):
        dup = sorted({i for i in ids if ids.count(i) > 1})[:10]
        raise AugmentPreflightError(f"duplicate sample ids in input snapshot: {dup}")

    batches_found: set[str] = set()
    missing_batch = 0
    for sample in samples:
        b = _batch_from_sample(sample)
        if b is None:
            missing_batch += 1
            continue
        batches_found.add(b)

    if len(batches_found) > 1:
        raise AugmentPreflightError(
            f"input snapshot mixes multiple batches {sorted(batches_found)}; "
            "refusing to process. Bind one batch via a single-stage Manifest."
        )
    if batches_found and batch not in batches_found:
        raise AugmentPreflightError(
            f"--source-name {batch!r} does not match Manifest batch field "
            f"{sorted(batches_found)[0]!r}"
        )
    if not batches_found and not allow_missing_batch_field:
        raise AugmentPreflightError(
            "Manifest has no batch field; pass --source-name to bind explicitly "
            "(recorded as binding_source=cli_source_name)"
        )

    already = [s.id for s in samples if detect_already_augmented(s)]
    if already:
        raise AugmentPreflightError(
            f"input already contains tempo-augmented samples (e.g. {already[:5]}); "
            "cascade augmentation is not supported in the first version"
        )

    binding = "manifest_batch" if batches_found else "cli_source_name"
    return {
        "source_batch": batch,
        "row_count": len(samples),
        "binding_source": binding,
        "batches_found": sorted(batches_found),
        "missing_batch_field_rows": missing_batch,
        "input_manifest": str(Path(input_manifest).resolve()),
    }


def preflight_tempo(
    *,
    params_raw: dict[str, Any],
    source_batch: str,
    input_manifest: Path | str,
    samples: list[Sample] | None = None,
    mock: bool = False,
) -> dict[str, Any]:
    """Validate params/backend/input and resolve product layout."""
    params = resolve_tempo_params(params_raw)
    batch = validate_source_name(source_batch)
    manifest_path = Path(input_manifest)
    if not manifest_path.is_file():
        raise AugmentPreflightError(
            f"default/input Manifest not found: {manifest_path}. "
            f"Expected datasets/stage1/segments/segments_{batch}.parquet "
            "or pass --input-manifest for this batch's other snapshot."
        )

    if samples is None:
        samples = list(Manifest.load(manifest_path))
    snap = validate_input_snapshot(
        samples,
        source_batch=batch,
        input_manifest=manifest_path,
    )

    if mock:
        ffmpeg_version = "mock"
        filter_graph = atempo_filter_graph(params.speed_factor)
        backend = {"binary": "mock", "version": "mock", "atempo_available": True}
    else:
        info = probe_ffmpeg(params.ffmpeg_bin)
        ffmpeg_version = info.version
        filter_graph = atempo_filter_graph(params.speed_factor)
        backend = info.to_dict()

    product_root, digest, derived_batch = resolve_product_layout(
        source_batch=batch,
        input_manifest=manifest_path,
        params=params,
        ffmpeg_version=ffmpeg_version,
        filter_graph=filter_graph,
    )
    return {
        "params": params,
        "source_batch": batch,
        "input_manifest": str(manifest_path.resolve()),
        "snapshot": snap,
        "ffmpeg": backend,
        "filter_graph": filter_graph,
        "ffmpeg_version": ffmpeg_version,
        "product_root": product_root,
        "recipe_digest": digest,
        "derived_batch": derived_batch,
        "samples": samples,
    }
