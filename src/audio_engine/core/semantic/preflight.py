"""Input preflight for semantic.judge manifests."""

from __future__ import annotations

import json
from typing import Any

from audio_engine.core.sample import Sample


def preflight_samples(samples: list[Sample]) -> None:
    """Reject duplicate ids and empty identity fields before batch Chat calls."""
    if not samples:
        raise ValueError("semantic.judge: input manifest has 0 samples")
    seen: dict[str, int] = {}
    errors: list[str] = []
    for idx, sample in enumerate(samples):
        sid = str(sample.id or "").strip()
        if not sid:
            errors.append(f"row[{idx}]: missing id")
            continue
        if sid in seen:
            errors.append(f"duplicate id {sid!r} at rows {seen[sid]} and {idx}")
        else:
            seen[sid] = idx
        if not str(sample.source_path or "").strip():
            errors.append(f"id={sid}: missing source_path")
    if errors:
        preview = "; ".join(errors[:20])
        more = f" (+{len(errors) - 20} more)" if len(errors) > 20 else ""
        raise ValueError(f"semantic.judge input preflight failed: {preview}{more}")


def diagnose_parquet_json_columns(path: str) -> list[str]:
    """Detect JSON canonical columns that decode to empty due to corruption."""
    import pandas as pd

    issues: list[str] = []
    df = pd.read_parquet(path)
    for col in ("transcripts", "labels", "quality", "audio", "status", "errors"):
        if col not in df.columns:
            continue
        for i, value in enumerate(df[col].tolist()):
            if value is None:
                continue
            if isinstance(value, str):
                try:
                    parsed = json.loads(value)
                except json.JSONDecodeError as exc:
                    issues.append(f"row[{i}].{col}: invalid JSON ({exc})")
                    continue
                if parsed == {} and value.strip() not in {"{}", ""}:
                    issues.append(f"row[{i}].{col}: JSON decoded empty unexpectedly")
    return issues
