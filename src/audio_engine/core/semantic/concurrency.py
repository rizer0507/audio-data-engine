"""Static concurrency budget checks for semantic.judge + sharding."""

from __future__ import annotations

from typing import Any


def effective_workers(
    *,
    pipeline_workers: int,
    step_workers: int | None,
    cli_workers: int | None,
    sharding_workers: int | None = None,
) -> int:
    if cli_workers is not None:
        return int(cli_workers)
    if step_workers is not None:
        return int(step_workers)
    if sharding_workers is not None:
        return int(sharding_workers)
    return int(pipeline_workers)


def validate_concurrency_budget(
    *,
    max_concurrency_total: int,
    workers_per_shard: int,
    parallel_shards: int = 1,
    context: str = "semantic.judge",
) -> dict[str, int]:
    """Reject configs where parallel_shards * workers exceeds the service budget."""
    if max_concurrency_total < 1:
        raise ValueError(f"{context}: max_concurrency_total must be >= 1")
    if workers_per_shard < 1:
        raise ValueError(f"{context}: workers_per_shard must be >= 1")
    if parallel_shards < 1:
        raise ValueError(f"{context}: parallel_shards must be >= 1")
    effective = int(parallel_shards) * int(workers_per_shard)
    if effective > int(max_concurrency_total):
        raise ValueError(
            f"{context}: concurrency budget exceeded: "
            f"parallel_shards({parallel_shards}) * workers({workers_per_shard}) "
            f"= {effective} > max_concurrency_total({max_concurrency_total}). "
            "Lower workers / parallel_shards or raise the service budget."
        )
    return {
        "parallel_shards": int(parallel_shards),
        "workers_per_shard": int(workers_per_shard),
        "effective_peak_requests": effective,
        "max_concurrency_total": int(max_concurrency_total),
    }


def budget_from_pipeline_config(
    pipeline_cfg: Any,
    *,
    max_concurrency_total: int,
    step_params: dict[str, Any] | None = None,
) -> dict[str, int]:
    """Derive workers/parallel from PipelineConfig (+ optional CLI overrides)."""
    step_params = step_params or {}
    override = getattr(pipeline_cfg, "execution_override", None) or {}
    execution = getattr(pipeline_cfg, "execution", None)
    sharding = getattr(pipeline_cfg, "sharding", None)

    pipeline_workers = int(getattr(execution, "workers", 4) or 4)
    cli_workers = override.get("workers")
    step_workers = step_params.get("workers")
    sharding_workers = getattr(sharding, "workers", None) if sharding else None

    workers = effective_workers(
        pipeline_workers=pipeline_workers,
        step_workers=int(step_workers) if step_workers is not None else None,
        cli_workers=int(cli_workers) if cli_workers is not None else None,
        sharding_workers=int(sharding_workers) if sharding_workers is not None else None,
    )
    parallel = 1
    if sharding is not None:
        parallel = int(getattr(sharding, "effective_parallel", 1) or 1)

    return validate_concurrency_budget(
        max_concurrency_total=max_concurrency_total,
        workers_per_shard=workers,
        parallel_shards=parallel,
    )
