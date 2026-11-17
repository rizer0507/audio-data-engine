"""Unified audio-scene resolution for stage-1 ingest (050).

Rules (stem = Sample.id = original filename without extension):
- starts with ``fenshen`` → digital twin, 16000 Hz for headerless PCM
- known approval prefixes ``shenpi_ai-`` / ``shepi_ai-``:
  - suffix after the second ASCII ``-`` contains uppercase ``V`` or ``P``
    → interview, 8000 Hz
  - otherwise → outbound, 16000 Hz
  - missing second separator / empty mid or suffix → contract reject
- anything else → ``unknown`` (still normalized with default headerless rate)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

SCENE_FENSHEN = "fenshen"
SCENE_SHENPI_INTERVIEW = "shenpi_interview"
SCENE_SHENPI_OUTBOUND = "shenpi_outbound"
SCENE_UNKNOWN = "unknown"

SHENPI_SCENES = frozenset({SCENE_SHENPI_INTERVIEW, SCENE_SHENPI_OUTBOUND})
VAD_FENSHEN_SCENES = frozenset({SCENE_FENSHEN})
VAD_SHENPI_SCENES = SHENPI_SCENES

AUDIO_SCENE_RULE_VERSION = "audio_scene/v1"

SHENPI_PREFIXES = ("shenpi_ai-", "shepi_ai-")

SceneName = Literal["fenshen", "shenpi_interview", "shenpi_outbound", "unknown"]


@dataclass(frozen=True)
class AudioSceneResolution:
    scene: SceneName
    source_sample_rate: int | None
    rule_version: str = AUDIO_SCENE_RULE_VERSION
    rejected: bool = False
    reject_reason: str | None = None
    matched_prefix: str | None = None

    def as_labels(self) -> dict[str, Any]:
        labels: dict[str, Any] = {
            "audio_scene": self.scene,
            "audio_scene_rule_version": self.rule_version,
        }
        if self.source_sample_rate is not None:
            labels["source_sample_rate"] = int(self.source_sample_rate)
        if self.matched_prefix:
            labels["audio_scene_prefix"] = self.matched_prefix
        if self.rejected:
            labels["ingest_rejected"] = True
            labels["ingest_reject_reason"] = self.reject_reason or "audio_scene_contract"
        return labels


def _match_shenpi_prefix(stem: str) -> str | None:
    for prefix in SHENPI_PREFIXES:
        if stem.startswith(prefix):
            return prefix
    return None


def resolve_audio_scene(stem: str) -> AudioSceneResolution:
    """Resolve scene + effective headerless source rate from a file stem / Sample.id."""
    text = str(stem or "")
    if text.startswith(SCENE_FENSHEN):
        return AudioSceneResolution(
            scene=SCENE_FENSHEN,
            source_sample_rate=16000,
            matched_prefix=SCENE_FENSHEN,
        )

    prefix = _match_shenpi_prefix(text)
    if prefix is None:
        return AudioSceneResolution(scene=SCENE_UNKNOWN, source_sample_rate=None)

    remainder = text[len(prefix) :]
    # First segment is already consumed by the prefix (ends with '-');
    # remainder is "<mid>-<suffix...>" with at least one ASCII '-'.
    if "-" not in remainder:
        return AudioSceneResolution(
            scene=SCENE_UNKNOWN,
            source_sample_rate=None,
            rejected=True,
            reject_reason="shenpi_missing_second_separator",
            matched_prefix=prefix,
        )
    mid, suffix = remainder.split("-", 1)
    if not mid or not suffix:
        return AudioSceneResolution(
            scene=SCENE_UNKNOWN,
            source_sample_rate=None,
            rejected=True,
            reject_reason="shenpi_empty_mid_or_suffix",
            matched_prefix=prefix,
        )
    if any(ch in suffix for ch in ("V", "P")):
        return AudioSceneResolution(
            scene=SCENE_SHENPI_INTERVIEW,
            source_sample_rate=8000,
            matched_prefix=prefix,
        )
    return AudioSceneResolution(
        scene=SCENE_SHENPI_OUTBOUND,
        source_sample_rate=16000,
        matched_prefix=prefix,
    )


def scene_from_sample_labels(labels: dict[str, Any] | None) -> str | None:
    if not labels:
        return None
    value = labels.get("audio_scene")
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def infer_legacy_fenshen_scene(sample_id: str, labels: dict[str, Any] | None) -> str | None:
    """Historical twin parents without audio_scene: keep FSMN compatible via fenshen prefix."""
    existing = scene_from_sample_labels(labels)
    if existing:
        return existing
    if str(sample_id or "").startswith(SCENE_FENSHEN):
        return SCENE_FENSHEN
    return None


def requires_reingest_for_shenpi(sample_id: str, labels: dict[str, Any] | None) -> bool:
    """Known approval prefix without scene label must not be consumed silently."""
    if scene_from_sample_labels(labels):
        return False
    return _match_shenpi_prefix(str(sample_id or "")) is not None
