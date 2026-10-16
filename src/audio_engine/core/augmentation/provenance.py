"""Archive source evidence so derived samples do not fake completed ASR/quality."""

from __future__ import annotations

import copy
from typing import Any

from audio_engine.core.sample import Sample

# Fields that describe the *current* audio identity and must not keep source values
# unless recomputed for the augmented file.
_CURRENT_IDENTITY_LABEL_KEYS = {
    "original_audio_sha256",
    "segment_pcm_sha256",
    "segment_wav_sha256",
    "normalized_audio_sha256",
    "source_pcm_sha256",
    "parent_pcm_sha256",
    "pcm_sha256",
    "wav_sha256",
}

# Quality keys tied to the source waveform.
_SOURCE_QUALITY_KEYS = {
    "dnsmos",
    "dnsmos_p835",
    "snr",
    "speech_rate",
    "vad",
    "silero",
    "energy",
    "acoustic",
}


def _archive_transcripts(transcripts: dict[str, Any]) -> dict[str, Any]:
    archived: dict[str, Any] = {}
    for key, value in (transcripts or {}).items():
        entry = copy.deepcopy(value) if isinstance(value, dict) else {"text": value}
        if not isinstance(entry, dict):
            entry = {"value": entry}
        entry = dict(entry)
        entry["inherited_from_source"] = True
        entry["applies_to"] = "source_audio_only"
        # Strip any completed-like markers that downstream might misread.
        entry.pop("status", None)
        archived[key] = entry
    return archived


def build_derived_sample(
    source: Sample,
    *,
    derived_id: str,
    output_path: str,
    output_sha256: str,
    sample_rate: int,
    channels: int,
    duration: float,
    augmentation_label: dict[str, Any],
    operator_name: str,
    operator_version: str,
) -> Sample:
    """Create a new Sample for the augmented audio with archived source evidence."""
    source_labels = copy.deepcopy(dict(source.labels or {}))
    source_quality = copy.deepcopy(dict(source.quality or {}))

    archived_identity = {
        key: source_labels.pop(key)
        for key in list(source_labels.keys())
        if key in _CURRENT_IDENTITY_LABEL_KEYS
    }
    archived_quality = {
        key: source_quality.pop(key)
        for key in list(source_quality.keys())
        if key in _SOURCE_QUALITY_KEYS or key.startswith("dnsmos")
    }

    # Gold / use-restriction / anti-leakage: keep on the derived sample.
    keep_label_keys = {
        "gold_text",
        "gold",
        "human_transcript",
        "train_exclude",
        "exclude_from_train",
        "do_not_train",
        "license",
        "usage_restriction",
        "split",
        "dataset_split",
        "group_id",
        "leakage_group",
        "speaker_id",
        "conversation_id",
        "parent_id",
        "parent_source_path",
        "vad_start_ms",
        "vad_end_ms",
        "segment_start_ms",
        "segment_end_ms",
        "source_start_ms",
        "source_end_ms",
    }
    retained_labels = {
        key: value
        for key, value in source_labels.items()
        if key in keep_label_keys
        or key.startswith("gold")
        or key.startswith("split")
        or key.startswith("group")
        or key.startswith("leak")
        or key.startswith("usage")
        or key.startswith("license")
        or key.startswith("parent")
        or key.startswith("vad_")
        or key.startswith("segment_")
    }
    # Mark gold text as inherited, not re-annotated on the new audio.
    for key in list(retained_labels.keys()):
        if "gold" in key or key == "human_transcript":
            if isinstance(retained_labels[key], dict):
                retained_labels[key] = {
                    **retained_labels[key],
                    "inherited_from_source": True,
                }
            else:
                retained_labels[f"{key}_inherited_from_source"] = True

    source_ns = {
        "sample_id": source.id,
        "source_path": source.source_path,
        "sha256": source.sha256,
        "sample_rate": source.sample_rate,
        "channels": source.channels,
        "duration": source.duration,
        "audio": copy.deepcopy(dict(source.audio or {})),
        "transcripts": _archive_transcripts(dict(source.transcripts or {})),
        "quality": archived_quality,
        "identity_labels": archived_identity,
        "status": copy.deepcopy(dict(source.status or {})),
        "labels_extra": {
            key: value
            for key, value in source_labels.items()
            if key not in retained_labels and key != "augmentation"
        },
    }

    labels = dict(retained_labels)
    labels["augmentation"] = dict(augmentation_label)
    labels["augmentation"]["source"] = source_ns

    audio = {
        "raw": output_path,
        augmentation_label.get("output_audio_key", "aug_speed"): output_path,
    }
    if sample_rate == 16000:
        audio["resampled_16k"] = output_path

    # Drop source completed/failed statuses so downstream does not think ASR etc. ran
    # on the new audio. Keep lineage history for audit.
    derived = Sample(
        id=derived_id,
        source_path=output_path,
        sha256=output_sha256,
        sample_rate=sample_rate,
        channels=channels,
        duration=duration,
        audio=audio,
        transcripts={},  # ASR archived under labels.augmentation.source
        quality={},  # acoustic scores archived under source namespace
        labels=labels,
        lineage=list(source.lineage),
        status={},
        errors={},
    )
    derived.add_lineage(
        operator=operator_name,
        version=operator_version,
        params={
            "speed_factor": augmentation_label.get("speed_factor"),
            "preserve_pitch": True,
            "backend": augmentation_label.get("backend"),
            "filter_graph": augmentation_label.get("filter_graph"),
            "processing_fingerprint": augmentation_label.get("processing_fingerprint"),
        },
        input_key=augmentation_label.get("input_audio_key"),
        output_key=augmentation_label.get("output_audio_key"),
        output_path=output_path,
        cache_key=augmentation_label.get("cache_key"),
    )
    derived.mark_completed(operator_name)
    return derived
