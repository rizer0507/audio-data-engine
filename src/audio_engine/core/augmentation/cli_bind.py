"""Bind --source-name / --speed-factor onto augment_batch_speed pipeline config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from audio_engine.core.augmentation.ffmpeg_atempo import atempo_filter_graph, probe_ffmpeg
from audio_engine.core.augmentation.params import (
    DEFAULT_SPEED_FACTOR,
    normalize_speed_factor,
    resolve_tempo_params,
)
from audio_engine.core.augmentation.recipe import resolve_product_layout
from audio_engine.core.source_naming import validate_source_name


def effective_speed_factor(
    *,
    cli_value: float | None,
    step_params: dict[str, Any],
) -> float:
    """CLI explicit value wins; otherwise YAML/step; otherwise 1.2."""
    if cli_value is not None:
        return normalize_speed_factor(cli_value)
    if "speed_factor" in step_params and step_params.get("speed_factor") is not None:
        return normalize_speed_factor(step_params.get("speed_factor"))
    return normalize_speed_factor(DEFAULT_SPEED_FACTOR)


def bind_tempo_pipeline(
    cfg: Any,
    *,
    source_batch: str | None,
    speed_factor_cli: float | None,
    input_manifest_override: str | Path | None = None,
    mock: bool = False,
    defer_publish: bool | None = None,
) -> dict[str, Any]:
    """Inject source_batch / speed_factor / product_root into augmentation.tempo steps.

    Returns a summary dict for logging.
    """
    batch = validate_source_name(source_batch) if source_batch else None
    tempo_steps = [s for s in cfg.steps if str(s.operator) == "augmentation.tempo"]
    if not tempo_steps:
        raise ValueError("augment_batch_speed requires an augmentation.tempo step")

    # Use first tempo step as parameter source.
    base_params = dict(tempo_steps[0].params or {})

    # Shard workers receive --input-manifest=<shard>, but recipe identity must stay
    # bound to the frozen full-batch snapshot already stored in step params.
    frozen_input = str(base_params.get("input_manifest") or "").strip()
    has_frozen_recipe = bool(
        frozen_input
        and base_params.get("product_root")
        and base_params.get("recipe_digest")
    )
    if has_frozen_recipe:
        input_manifest = Path(frozen_input)
        factor = normalize_speed_factor(base_params.get("speed_factor", DEFAULT_SPEED_FACTOR))
        if speed_factor_cli is not None and normalize_speed_factor(speed_factor_cli) != factor:
            raise ValueError(
                f"--speed-factor {speed_factor_cli} conflicts with frozen recipe "
                f"speed_factor={factor}; start a new run"
            )
    else:
        factor = effective_speed_factor(cli_value=speed_factor_cli, step_params=base_params)
        input_manifest = input_manifest_override or cfg.input_manifest
        if not input_manifest:
            raise ValueError("augment_batch_speed needs input.manifest or --input-manifest")
        input_manifest = Path(input_manifest)

    base_params["speed_factor"] = factor

    if batch is None:
        # Allow explicit manifests without --source-name only if batch already in params.
        batch = str(base_params.get("source_batch") or "").strip() or None
        if not batch:
            raise ValueError(
                "augment_batch_speed requires --source-name (batch id) "
                "or params.source_batch"
            )
        batch = validate_source_name(batch)

    params = resolve_tempo_params(base_params)
    if mock:
        ffmpeg_version = "mock"
    else:
        ffmpeg_version = probe_ffmpeg(params.ffmpeg_bin).version
    filter_graph = atempo_filter_graph(params.speed_factor)

    if has_frozen_recipe:
        product_root = Path(str(base_params["product_root"]))
        recipe_digest = str(base_params["recipe_digest"])
        derived_batch = str(
            base_params.get("derived_batch")
            or f"{batch}__tempo_{factor}"
        )
    else:
        product_root, recipe_digest, derived_batch = resolve_product_layout(
            source_batch=batch,
            input_manifest=input_manifest,
            params=params,
            ffmpeg_version=ffmpeg_version,
            filter_graph=filter_graph,
        )
    output_manifest = product_root / "manifest.parquet"

    for step in tempo_steps:
        merged = dict(step.params or {})
        merged.update(
            {
                "speed_factor": factor,
                "source_batch": batch,
                "input_manifest": str(Path(input_manifest).resolve()),
                "product_root": str(product_root.resolve()),
                "recipe_digest": recipe_digest,
                "derived_batch": derived_batch,
                "backend": params.backend,
                "preserve_pitch": True,
                "input_audio_key": params.input_audio_key,
                "output_audio_key": params.output_audio_key,
                "backend_threads": params.backend_threads,
                "timeout_seconds": params.timeout_seconds,
                "max_attempts": params.max_attempts,
                "workers": params.workers,
                "max_in_flight": params.max_in_flight,
                "channel_policy": params.channel_policy,
                "output_format": params.output_format,
                "output_subtype": params.output_subtype,
            }
        )
        if defer_publish is not None:
            merged["defer_publish"] = bool(defer_publish)
        step.params = merged

    # Parent bind freezes the full-batch input path into cfg; shard workers keep the
    # shard parquet on cfg.input_manifest (CLI --input-manifest) while steps still
    # reference the frozen full snapshot via params.input_manifest.
    if not has_frozen_recipe:
        cfg.input_manifest = str(Path(input_manifest).resolve())
        cfg.output_manifest = str(output_manifest)
    return {
        "source_batch": batch,
        "speed_factor": factor,
        "input_manifest": str(Path(input_manifest).resolve()),
        "product_root": str(product_root.resolve()),
        "recipe_digest": recipe_digest,
        "derived_batch": derived_batch,
        "output_manifest": str(output_manifest),
        "ffmpeg_version": ffmpeg_version,
        "filter_graph": filter_graph,
        "preserve_pitch": True,
        "backend": params.backend,
        "input_audio_key": params.input_audio_key,
        "workers": params.workers,
        "max_in_flight": params.max_in_flight,
        "backend_threads": params.backend_threads,
    }


def finalize_sharded_tempo(
    *,
    product_root: Path,
    prepared_input: Path,
    merged_samples: list[Any],
    recipe: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge results_parts after sharded run and publish formal/partial manifest."""
    import pandas as pd

    from audio_engine.core.augmentation.publish import (
        load_recipe,
        publish_batch_artifacts,
        write_results_parquet,
    )
    from audio_engine.core.manifest import Manifest

    root = Path(product_root)
    parts_dir = root / "results_parts"
    rows: list[dict[str, Any]] = []
    if parts_dir.is_dir():
        for part in sorted(parts_dir.glob("*.parquet")):
            df = pd.read_parquet(part)
            rows.extend(df.to_dict(orient="records"))

    prepared = Manifest.load(prepared_input)
    source_order = [s.id for s in prepared.samples]

    # Deduplicate rows by source_id (last wins).
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        by_id[str(row.get("source_id"))] = row
    # Ensure every prepared id has a row (missing → failed).
    for sid in source_order:
        if sid not in by_id:
            by_id[sid] = {
                "source_id": sid,
                "status": "failed",
                "error_code": "missing_result",
                "error_message": "no shard result row for source id",
            }
    ordered_rows = [by_id[sid] for sid in source_order]
    write_results_parquet(root / "results.parquet", ordered_rows)

    frozen = recipe or load_recipe(root) or {}
    success = [s for s in merged_samples if str(getattr(s, "id", "")).startswith("tempo_")]
    import time

    return publish_batch_artifacts(
        product_root=root,
        recipe=frozen,
        results_rows=ordered_rows,
        success_samples=success,
        source_order=source_order,
        started_at=time.perf_counter(),
        raise_on_failure=True,
    )
