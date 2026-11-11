"""Browse manifests using existing batch naming and publication metadata."""

from pathlib import PurePosixPath
from typing import Any

from audio_engine.core.source_naming import infer_batch_name


def manifest_path(value: str) -> PurePosixPath:
    return PurePosixPath(value.replace("\\", "/"))


def manifest_batch(value: str) -> str | None:
    path = manifest_path(value)
    stem = path.stem.removesuffix(".annotation_work")
    for folder in ("warehouses", "augmented", "eval_sets"):
        if folder in path.parts[:-2]:
            return path.parts[path.parts.index(folder) + 1]
    if stem.startswith("segments_"):
        stem = stem.removesuffix("_vad_all")
    return infer_batch_name(stem)


def task_batch(request: dict[str, Any]) -> str | None:
    source = request.get("input") or {}
    params = request.get("parameters") or {}
    return source.get("batch") or params.get("batch") or manifest_batch(source.get("dataset_id") or "")


def manifest_stage(value: str, request: dict[str, Any]) -> tuple[str | None, str | None]:
    stem = manifest_path(value).stem
    for prefix, stage in (
        ("ingested_", "ingest"), ("segments_", "segments"), ("dnsmos_", "dnsmos"),
        ("prepared_evidence_", "evidence"), ("tagged_free_tags_", "tags"),
    ):
        if stem.startswith(prefix):
            if stage == "segments" and stem.endswith("_vad_all"):
                return "segments_all", None
            return stage, None
    if "_asr_" in stem and not stem.startswith(("multi_asr_", "prepared_asr_")):
        return "asr", stem.split("_asr_", 1)[0]
    plugin = (request.get("root_plugin") or {}).get("id")
    for family in ("qwen", "glm", "sensevoice"):
        if plugin == f"{family}_asr_batch":
            return "asr", (request.get("parameters") or {}).get("asr_run") or family
    stage = {
        "audio_ingest": "ingest", "vad_segment_batch": "segments", "dnsmos_score_batch": "dnsmos",
        "evidence_join": "evidence", "tag_dataset_free_tags_v2": "tags",
    }.get(plugin)
    return stage, None
