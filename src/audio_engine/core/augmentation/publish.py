"""Results / recipe / summary / conditional formal manifest publish."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from audio_engine.core.artifacts import atomic_path, atomic_write_json
from audio_engine.core.manifest import Manifest
from audio_engine.core.sample import Sample


class AugmentBatchIncomplete(RuntimeError):
    """Raised when some rows failed so the formal manifest must not be published."""

    def __init__(self, message: str, *, summary: dict[str, Any]):
        super().__init__(message)
        self.summary = summary


def _results_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return list(rows)


def write_results_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(_results_rows(rows))
    with atomic_path(path) as tmp:
        df.to_parquet(tmp, index=False)


def order_success_samples(
    samples: list[Sample],
    *,
    source_order: list[str],
) -> list[Sample]:
    """Restore input relative order using source ids."""
    by_source: dict[str, Sample] = {}
    for sample in samples:
        aug = (sample.labels or {}).get("augmentation") or {}
        source_id = str(aug.get("source_id") or "")
        if not source_id:
            source = aug.get("source") if isinstance(aug, dict) else None
            if isinstance(source, dict):
                source_id = str(source.get("sample_id") or "")
        if source_id:
            by_source[source_id] = sample
    ordered: list[Sample] = []
    for source_id in source_order:
        sample = by_source.get(source_id)
        if sample is not None and sample.status.get("augmentation.tempo") == "completed":
            # Only successful derived samples enter the formal manifest.
            if str(sample.id).startswith("tempo_"):
                ordered.append(sample)
    return ordered


def publish_batch_artifacts(
    *,
    product_root: Path,
    recipe: dict[str, Any],
    results_rows: list[dict[str, Any]],
    success_samples: list[Sample],
    source_order: list[str],
    started_at: float,
    raise_on_failure: bool = True,
) -> dict[str, Any]:
    """Write results/recipe/summary; formal manifest only when N=M and F=0."""
    root = Path(product_root)
    root.mkdir(parents=True, exist_ok=True)

    n = len(source_order)
    by_source = {str(r.get("source_id")): r for r in results_rows}
    # Ensure coverage of every input id.
    covered = [sid for sid in source_order if sid in by_source]
    if len(covered) != n:
        missing = [sid for sid in source_order if sid not in by_source]
        raise RuntimeError(f"results missing {len(missing)} source ids, e.g. {missing[:5]}")

    ok_rows = [by_source[sid] for sid in source_order if by_source[sid].get("status") == "ok"]
    fail_rows = [by_source[sid] for sid in source_order if by_source[sid].get("status") != "ok"]
    cache_hits = sum(1 for r in ok_rows if r.get("cache_hit"))
    m = len(ok_rows)
    f = len(fail_rows)
    if m + f != n:
        raise RuntimeError(f"results conservation broken: N={n} M={m} F={f}")

    elapsed = max(0.0, time.perf_counter() - started_at)
    input_duration = sum(float(r.get("input_duration") or 0.0) for r in results_rows)
    output_duration = sum(
        float(r.get("output_duration") or 0.0) for r in ok_rows
    )
    summary: dict[str, Any] = {
        "status": "completed" if f == 0 and m == n else "partial" if m else "failed",
        "total": n,
        "success": m,
        "failed": f,
        "cache_hits": cache_hits,
        "input_duration_sec": input_duration,
        "output_duration_sec": output_duration,
        "elapsed_sec": elapsed,
        "samples_per_sec": (n / elapsed) if elapsed > 0 else 0.0,
        "product_root": str(root.resolve()),
        "recipe_digest": recipe.get("recipe_digest"),
        "speed_factor": recipe.get("speed_factor"),
        "source_batch": recipe.get("source_batch"),
        "derived_batch": recipe.get("derived_batch"),
    }

    write_results_parquet(root / "results.parquet", [by_source[sid] for sid in source_order])
    atomic_write_json(root / "recipe.json", recipe)
    atomic_write_json(root / "summary.json", summary)

    ordered = order_success_samples(success_samples, source_order=source_order)
    # Prefer reconstructing order strictly from results ok rows when possible.
    if len(ordered) != m:
        # Fall back: keep whatever success_samples we have, sorted by source_index.
        indexed = []
        for sample in success_samples:
            aug = (sample.labels or {}).get("augmentation") or {}
            if sample.status.get("augmentation.tempo") != "completed":
                continue
            if not str(sample.id).startswith("tempo_"):
                continue
            indexed.append((int(aug.get("source_index") or 10**12), sample))
        indexed.sort(key=lambda x: x[0])
        ordered = [s for _, s in indexed]

    partial_path = root / "manifest.partial.parquet"
    formal_path = root / "manifest.parquet"

    if f == 0 and m == n and len(ordered) == n:
        Manifest(ordered).save(formal_path)
        Manifest(ordered).save(formal_path.with_suffix(".jsonl"))
        if partial_path.exists():
            partial_path.unlink()
        summary["formal_manifest"] = str(formal_path.resolve())
        atomic_write_json(root / "summary.json", summary)
        return summary

    # Partial / failed: never leave a formal complete manifest.
    if formal_path.exists():
        formal_path.unlink()
    sibling_jsonl = formal_path.with_suffix(".jsonl")
    if sibling_jsonl.exists():
        sibling_jsonl.unlink()
    if ordered:
        Manifest(ordered).save(partial_path)
        summary["partial_manifest"] = str(partial_path.resolve())
    atomic_write_json(root / "summary.json", summary)

    message = (
        f"tempo augmentation incomplete: success={m} failed={f} total={n}; "
        f"formal manifest not published. See {root / 'results.parquet'}"
    )
    if raise_on_failure:
        raise AugmentBatchIncomplete(message, summary=summary)
    summary["error"] = message
    return summary


def load_recipe(product_root: Path) -> dict[str, Any] | None:
    path = Path(product_root) / "recipe.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
