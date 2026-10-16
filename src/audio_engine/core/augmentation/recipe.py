"""Recipe identity and product directory layout for tempo augmentation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from audio_engine.core.augmentation.params import (
    OPERATOR_VERSION,
    TempoParams,
    factor_tag,
)
from audio_engine.core.manifest import file_sha256

STAGE1_AUGMENTED_DIR = Path("datasets") / "stage1" / "augmented"


def recipe_digest(
    *,
    source_batch: str,
    input_manifest: str | Path,
    input_manifest_sha256: str,
    params: TempoParams,
    ffmpeg_version: str,
    filter_graph: str,
) -> str:
    payload = {
        "source_batch": source_batch,
        "input_manifest": str(Path(input_manifest).as_posix()),
        "input_manifest_sha256": input_manifest_sha256,
        "speed_factor": params.speed_factor,
        "input_audio_key": params.input_audio_key,
        "backend": params.backend,
        "ffmpeg_version": ffmpeg_version,
        "filter_graph": filter_graph,
        "output_format": params.output_format,
        "output_subtype": params.output_subtype,
        "channel_policy": params.channel_policy,
        "preserve_pitch": True,
        "operator_version": OPERATOR_VERSION,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def product_root_for(
    source_batch: str,
    params: TempoParams,
    recipe_id: str,
) -> Path:
    return STAGE1_AUGMENTED_DIR / source_batch / f"tempo_{factor_tag(params.speed_factor)}_{recipe_id}"


def build_recipe(
    *,
    source_batch: str,
    input_manifest: Path,
    input_row_count: int,
    params: TempoParams,
    ffmpeg_version: str,
    filter_graph: str,
    derived_batch: str,
    product_root: Path,
    binding_source: str,
) -> dict[str, Any]:
    manifest_path = Path(input_manifest).resolve()
    manifest_sha = file_sha256(manifest_path) if manifest_path.is_file() else ""
    digest = recipe_digest(
        source_batch=source_batch,
        input_manifest=manifest_path,
        input_manifest_sha256=manifest_sha,
        params=params,
        ffmpeg_version=ffmpeg_version,
        filter_graph=filter_graph,
    )
    return {
        "recipe_digest": digest,
        "source_batch": source_batch,
        "derived_batch": derived_batch,
        "binding_source": binding_source,
        "input_manifest": str(manifest_path),
        "input_manifest_sha256": manifest_sha,
        "input_row_count": input_row_count,
        "speed_factor": params.speed_factor,
        "factor_tag": factor_tag(params.speed_factor),
        "input_audio_key": params.input_audio_key,
        "output_audio_key": params.output_audio_key,
        "backend": params.backend,
        "preserve_pitch": True,
        "ffmpeg_version": ffmpeg_version,
        "filter_graph": filter_graph,
        "output_format": params.output_format,
        "output_subtype": params.output_subtype,
        "channel_policy": params.channel_policy,
        "backend_threads": params.backend_threads,
        "operator_version": OPERATOR_VERSION,
        "product_root": str(Path(product_root).resolve()),
    }


def resolve_product_layout(
    *,
    source_batch: str,
    input_manifest: Path,
    params: TempoParams,
    ffmpeg_version: str,
    filter_graph: str,
) -> tuple[Path, str, str]:
    """Return (product_root, recipe_digest, derived_batch_id)."""
    from audio_engine.core.augmentation.identity import derived_batch_id

    manifest_path = Path(input_manifest).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"input manifest not found: {manifest_path}. "
            f"Default expects datasets/stage1/segments/segments_{source_batch}.parquet "
            "or pass --input-manifest explicitly."
        )
    digest = recipe_digest(
        source_batch=source_batch,
        input_manifest=manifest_path,
        input_manifest_sha256=file_sha256(manifest_path),
        params=params,
        ffmpeg_version=ffmpeg_version,
        filter_graph=filter_graph,
    )
    root = product_root_for(source_batch, params, digest)
    batch_id = derived_batch_id(source_batch, params.speed_factor, digest)
    return root, digest, batch_id
