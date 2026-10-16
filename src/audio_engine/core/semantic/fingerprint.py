"""Cache / checkpoint fingerprints for semantic.judge."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from audio_engine.core.checkpoint import digest_payload
from audio_engine.core.sample import Sample
from audio_engine.core.semantic.config import FrozenSemanticBundle
from audio_engine.core.semantic.evidence import evidence_fingerprint_payload
from audio_engine.core.semantic.types import ENGINE_VERSION


def input_context_digest(samples: list[Sample], bundle: FrozenSemanticBundle) -> str:
    """Digest template-readable sample context + evidence for checkpoint identity."""
    rows = []
    for sample in samples:
        rows.append(
            {
                "id": sample.id,
                "evidence": evidence_fingerprint_payload(sample, bundle.task.evidence),
            }
        )
    return digest_payload({"rows": rows, "count": len(rows)})


def semantic_cache_payload(
    sample: Sample,
    bundle: FrozenSemanticBundle,
    *,
    mock: bool,
) -> dict[str, Any]:
    """Full cache identity. Excludes API keys and prior semantic_results."""
    return {
        "engine_version": ENGINE_VERSION,
        "task_id": bundle.task.task_id,
        "task_version": bundle.task.task_version,
        "content_fingerprint": bundle.content_fingerprint,
        "mock": bool(mock),
        "cache_namespace": bundle.service.cache_namespace,
        "model": bundle.service.model,
        "model_revision": bundle.service.model_revision,
        "generation": bundle.service.generation.fingerprint(),
        "structured_output_mode": bundle.service.structured_output_mode,
        "enable_thinking": bundle.service.enable_thinking,
        "evidence": evidence_fingerprint_payload(sample, bundle.task.evidence),
    }


def semantic_cache_key(
    sample: Sample,
    bundle: FrozenSemanticBundle,
    *,
    mock: bool,
) -> str:
    raw = json.dumps(
        semantic_cache_payload(sample, bundle, mock=mock),
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def result_fingerprint(result: dict[str, Any] | None) -> str | None:
    if result is None:
        return None
    return digest_payload(result)
