from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_sources(manifest_path: str | Path = "resources/manifest.yaml") -> list[dict[str, Any]]:
    path = Path(manifest_path)
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return list(data.get("sources") or [])


def lookup_source(
    source_id: str,
    *,
    manifest_path: str | Path = "resources/manifest.yaml",
) -> dict[str, Any]:
    """Look up a registered logical source by source_id."""
    for entry in load_sources(manifest_path):
        if entry.get("source_id") == source_id:
            return entry
    raise KeyError(
        f"Source '{source_id}' not found in {manifest_path}. "
        "Register it in resources/manifest.yaml first."
    )


def resolve_source_input(
    source_id: str,
    *,
    resources_manifest: str | Path = "resources/manifest.yaml",
) -> dict[str, str]:
    """Resolve source_id to pipeline input.

    Preference order for an existing sample index:
      1. resources/sources/{id}/samples.jsonl
      2. resources/sources/{id}/samples.parquet
      3. datasets/manifests/{id}.parquet
      4. datasets/batches/{id}/manifest.parquet

    If none exist, fall back to the registered external path as source_dir
    (pipeline should start with ingest.scan).
    """
    entry = lookup_source(source_id, manifest_path=resources_manifest)
    candidates = [
        Path(f"resources/sources/{source_id}/samples.jsonl"),
        Path(f"resources/sources/{source_id}/samples.parquet"),
        Path(f"datasets/manifests/{source_id}.parquet"),
        Path(f"datasets/batches/{source_id}/manifest.parquet"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return {"manifest": str(candidate), "source_id": source_id}

    path = entry.get("path")
    if not path:
        raise ValueError(f"Source '{source_id}' has no path and no sample index")
    return {"source_dir": str(path), "source_id": source_id, "source_meta": entry}
