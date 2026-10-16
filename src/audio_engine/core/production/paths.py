"""Canonical dataset paths for the 045 production chain."""

from __future__ import annotations

from pathlib import Path

from audio_engine.core.source_naming import (
    DATASETS_ROOT,
    STAGE1_ASR_DIR,
    STAGE1_DERIVED_DIR,
    _posix,
    manifest_path,
    model_asr_kind,
    staged_manifest_path,
    validate_asr_run,
    validate_source_name,
)

STAGE1_INGESTED_DIR = DATASETS_ROOT / "stage1" / "ingested"
STAGE1_SEGMENTS_DIR = DATASETS_ROOT / "stage1" / "segments"
STAGE1_QUALITY_DIR = DATASETS_ROOT / "stage1" / "quality"


class ProductionPaths:
    """Stable product locations for one batch (source_name)."""

    def __init__(self, source_name: str):
        self.source_name = validate_source_name(source_name)

    @property
    def ingested(self) -> Path:
        return STAGE1_INGESTED_DIR / f"ingested_{self.source_name}.parquet"

    @property
    def segments(self) -> Path:
        return STAGE1_SEGMENTS_DIR / f"segments_{self.source_name}.parquet"

    def asr(self, route_id: str) -> Path:
        alias = validate_asr_run(route_id)
        return STAGE1_ASR_DIR / f"{model_asr_kind(alias)}_{self.source_name}.parquet"

    @property
    def dnsmos(self) -> Path:
        return STAGE1_QUALITY_DIR / f"dnsmos_{self.source_name}.parquet"

    @property
    def prepared_evidence(self) -> Path:
        return STAGE1_DERIVED_DIR / f"prepared_evidence_{self.source_name}.parquet"

    @property
    def tagged_free_tags(self) -> Path:
        return STAGE1_DERIVED_DIR / f"tagged_free_tags_v2_{self.source_name}.parquet"

    @property
    def summary_xlsx(self) -> Path:
        return Path("data/exports") / f"summary_free_tags_v2_{self.source_name}.xlsx"

    def as_posix(self, path: Path) -> str:
        return _posix(path)


def register_production_stems(stem: str) -> Path | None:
    """Return staged directory for new 045 stems, or None if not a production stem."""
    text = str(stem or "").strip()
    if text.lower().endswith((".parquet", ".jsonl")):
        text = Path(text).stem
    if text.startswith("ingested_"):
        return STAGE1_INGESTED_DIR
    if text.startswith("segments_"):
        return STAGE1_SEGMENTS_DIR
    if text.startswith("dnsmos_"):
        return STAGE1_QUALITY_DIR
    if text.startswith("prepared_evidence_"):
        return STAGE1_DERIVED_DIR
    if text.startswith("tagged_free_tags_"):
        return STAGE1_DERIVED_DIR
    return None


def legacy_quality_sidecar_path(source_name: str) -> Path:
    """Historical sidecar path (readable only; not used by new production chain)."""
    return staged_manifest_path(f"quality_sidecar_{validate_source_name(source_name)}")


def dnsmos_manifest_path(source_name: str) -> Path:
    return ProductionPaths(source_name).dnsmos


def prepared_evidence_path(source_name: str) -> Path:
    return ProductionPaths(source_name).prepared_evidence
