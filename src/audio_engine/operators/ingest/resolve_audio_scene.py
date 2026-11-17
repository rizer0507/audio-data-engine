"""Tag each ingested parent with labels.audio_scene before PCM decode (050)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

from audio_engine.core.audio_scene import (
    AUDIO_SCENE_RULE_VERSION,
    SCENE_FENSHEN,
    SCENE_SHENPI_INTERVIEW,
    SCENE_SHENPI_OUTBOUND,
    SCENE_UNKNOWN,
    resolve_audio_scene,
)
from audio_engine.core.operator import ManifestOperator, OperatorConfig
from audio_engine.core.registry import register_operator
from audio_engine.core.sample import Sample
from audio_engine.operators.audio.pcm import looks_like_wav

_PCM_EXTS = {".pcm", ".raw"}
_SHENPI_SCENES = {SCENE_SHENPI_INTERVIEW, SCENE_SHENPI_OUTBOUND}


def _reject(sample: Sample, reason: str) -> Sample:
    updated = sample.model_copy(deep=True)
    updated.labels["ingest_rejected"] = True
    updated.labels["ingest_reject_reason"] = reason
    updated.labels["audio_pass"] = False
    updated.quality["broken"] = True
    updated.quality["ingest_reject_reason"] = reason
    return updated


@register_operator
class ResolveAudioSceneOperator(ManifestOperator):
    """Resolve scene + effective headerless rate; reject contract-breaking parents."""

    name = "resolve_audio_scene"
    version = "1.0.0"
    category = "ingest"

    def run(self, samples: list[Sample], config: OperatorConfig) -> list[Sample]:
        drop_rejected = bool(config.params.get("drop_rejected", True))
        stats = {
            "total": 0,
            SCENE_FENSHEN: 0,
            SCENE_SHENPI_INTERVIEW: 0,
            SCENE_SHENPI_OUTBOUND: 0,
            SCENE_UNKNOWN: 0,
            "rejected": 0,
            "reject_reasons": {},
        }
        kept: list[Sample] = []
        for sample in samples:
            stats["total"] += 1
            resolution = resolve_audio_scene(sample.id)
            updated = sample.model_copy(deep=True)
            updated.labels.update(resolution.as_labels())
            updated.add_lineage(
                operator=self.full_name,
                version=self.version,
                params={
                    "rule_version": AUDIO_SCENE_RULE_VERSION,
                    "scene": resolution.scene,
                    "source_sample_rate": resolution.source_sample_rate,
                },
            )
            updated.mark_completed(self.full_name)

            if resolution.rejected:
                updated = _reject(updated, resolution.reject_reason or "audio_scene_contract")
                stats["rejected"] += 1
                reason = resolution.reject_reason or "audio_scene_contract"
                stats["reject_reasons"][reason] = stats["reject_reasons"].get(reason, 0) + 1
                if not drop_rejected:
                    kept.append(updated)
                continue

            # Approval inputs: first release only accepts headerless .pcm/.raw.
            # A RIFF/WAVE container (even with .pcm suffix) is a contract reject.
            if resolution.scene in _SHENPI_SCENES:
                raw_path = None
                try:
                    raw_path = Path(updated.audio_path("raw"))
                except Exception:
                    raw_path = Path(updated.source_path) if updated.source_path else None
                if raw_path is None or not raw_path.is_file():
                    updated = _reject(updated, "shenpi_raw_missing")
                    stats["rejected"] += 1
                    stats["reject_reasons"]["shenpi_raw_missing"] = (
                        stats["reject_reasons"].get("shenpi_raw_missing", 0) + 1
                    )
                    if not drop_rejected:
                        kept.append(updated)
                    continue
                suffix = raw_path.suffix.lower()
                if suffix not in _PCM_EXTS:
                    updated = _reject(updated, "shenpi_non_pcm_extension")
                    stats["rejected"] += 1
                    stats["reject_reasons"]["shenpi_non_pcm_extension"] = (
                        stats["reject_reasons"].get("shenpi_non_pcm_extension", 0) + 1
                    )
                    if not drop_rejected:
                        kept.append(updated)
                    continue
                if looks_like_wav(raw_path):
                    updated = _reject(updated, "shenpi_container_header_as_pcm")
                    stats["rejected"] += 1
                    stats["reject_reasons"]["shenpi_container_header_as_pcm"] = (
                        stats["reject_reasons"].get("shenpi_container_header_as_pcm", 0) + 1
                    )
                    if not drop_rejected:
                        kept.append(updated)
                    continue
                # Odd / empty byte length is checked again in pcm_to_wav; pre-check here.
                size = raw_path.stat().st_size
                if size <= 0:
                    updated = _reject(updated, "shenpi_empty_pcm")
                    stats["rejected"] += 1
                    stats["reject_reasons"]["shenpi_empty_pcm"] = (
                        stats["reject_reasons"].get("shenpi_empty_pcm", 0) + 1
                    )
                    if not drop_rejected:
                        kept.append(updated)
                    continue
                if size % 2 != 0:
                    updated = _reject(updated, "shenpi_odd_pcm_bytes")
                    stats["rejected"] += 1
                    stats["reject_reasons"]["shenpi_odd_pcm_bytes"] = (
                        stats["reject_reasons"].get("shenpi_odd_pcm_bytes", 0) + 1
                    )
                    if not drop_rejected:
                        kept.append(updated)
                    continue

            stats[resolution.scene] = stats.get(resolution.scene, 0) + 1
            kept.append(updated)

        if config.run_dir:
            from audio_engine.core.artifacts import atomic_write_json

            atomic_write_json(
                Path(config.run_dir) / "audio_scene_stats.json",
                {
                    "rule_version": AUDIO_SCENE_RULE_VERSION,
                    **stats,
                    "kept": len(kept),
                },
            )
        logger.info(
            "ingest.resolve_audio_scene: total={} kept={} rejected={} "
            "fenshen={} interview={} outbound={} unknown={}",
            stats["total"],
            len(kept),
            stats["rejected"],
            stats[SCENE_FENSHEN],
            stats[SCENE_SHENPI_INTERVIEW],
            stats[SCENE_SHENPI_OUTBOUND],
            stats[SCENE_UNKNOWN],
        )
        return kept
