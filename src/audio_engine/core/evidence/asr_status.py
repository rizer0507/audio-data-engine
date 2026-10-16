"""ASR run-status resolution (extracted from selection_v3.input_contract, 044).

Only the neutral per-route status contract lives here: success_text / success_empty /
failed / missing. No sample readiness queues, conservation reports or business
categories.
"""

from __future__ import annotations

from typing import Any

from audio_engine.core.evidence.route_config import RouteConfig
from audio_engine.core.evidence.text import raw_transcript_text
from audio_engine.core.evidence.types import (
    RUN_STATUS_FAILED,
    RUN_STATUS_MISSING,
    RUN_STATUS_SUCCESS_EMPTY,
    RUN_STATUS_SUCCESS_TEXT,
)
from audio_engine.core.sample import Sample


def original_audio_sha256(sample: Sample) -> str:
    """Working (unpadded) audio content hash for the current Sample.

    For VAD segments (041), this is the *segment* PCM identity written to
    ``labels.original_audio_sha256`` / ``segment_pcm_sha256``. Parent PCM hashes
    live in ``source_pcm_sha256`` / ``parent_pcm_sha256`` and must not be used
    here — otherwise sibling segments share cache keys and warehouse identity.
    """
    for container in (sample.labels, sample.quality):
        seg = container.get("segment_pcm_sha256")
        if seg is not None and str(seg).strip() and container.get("is_vad_segment"):
            return str(seg).strip()
    for container in (sample.labels, sample.quality):
        for key in ("original_audio_sha256", "source_audio_sha256", "raw_audio_sha256"):
            value = container.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
    provenance = sample.labels.get("provenance")
    if isinstance(provenance, dict):
        for key in ("original_audio_sha256", "source_audio_sha256"):
            value = provenance.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
    return str(sample.sha256 or "").strip()


def transcript_entry(sample: Sample, key: str) -> dict[str, Any] | None:
    entry = sample.transcripts.get(key)
    if entry is None:
        return None
    if isinstance(entry, dict):
        return entry
    return {"text": str(entry)}


def explicit_failed(entry: dict[str, Any], sample: Sample, key: str) -> bool:
    status = str(entry.get("status") or "").strip().lower()
    if status in {"failed", "error", "fail"}:
        return True
    extra = entry.get("extra") if isinstance(entry.get("extra"), dict) else {}
    inf = str(extra.get("inference_status") or extra.get("status") or "").strip().lower()
    if inf in {"failed", "error", "fail"}:
        return True
    # Operator-level failure recorded against this transcript key / asr family
    for err_key, err_val in (sample.errors or {}).items():
        if not err_val:
            continue
        lowered = str(err_key).lower()
        if key.lower() in lowered or "asr" in lowered:
            # Only treat as this run's failure when status also marks failed
            if sample.status.get(err_key) == "failed":
                return True
    if entry.get("failed") is True:
        return True
    return False


def classify_run_status(
    sample: Sample,
    transcript_key: str,
    config: RouteConfig | None = None,
) -> str:
    """Map one ASR route to success_text | success_empty | failed | missing."""
    entry = transcript_entry(sample, transcript_key)
    if entry is None:
        return RUN_STATUS_MISSING
    if explicit_failed(entry, sample, transcript_key):
        return RUN_STATUS_FAILED
    if str(entry.get("status") or "").lower() in {"missing", "pending", "running"}:
        return RUN_STATUS_MISSING
    text = entry.get("text")
    if text is None:
        return RUN_STATUS_MISSING
    raw = raw_transcript_text(entry)
    if config is not None and config.uses_chinese_only_text():
        from audio_engine.core.evidence.classify_text import prepare_classify_text

        family = config.family_of_key(transcript_key)
        prepared = prepare_classify_text(
            raw,
            echo=config.echo_table_for(family),
            keep_digits=config.classify_text_keep_digits,
            policy=config.classify_text_policy,
        )
        return prepared.status
    if raw.strip() == "":
        return RUN_STATUS_SUCCESS_EMPTY
    return RUN_STATUS_SUCCESS_TEXT
