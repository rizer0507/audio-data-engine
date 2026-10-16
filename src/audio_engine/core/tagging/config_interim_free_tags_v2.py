"""Structural validators for free_tags_v2 YAML (044).

``load_tagging_config`` already accepts ``evidence_config`` alone; this module
remains for lightweight schema checks without loading rules/lexicons.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from audio_engine.core.tagging.migrate import DEFAULT_EVIDENCE_REL, FREE_TAGS_V2_POLICY


def is_free_tags_v2_raw(raw: dict[str, Any]) -> bool:
    return str(raw.get("policy_version") or "").strip() == FREE_TAGS_V2_POLICY


def validate_free_tags_v2_raw(raw: dict[str, Any], *, path: str = "") -> list[str]:
    """Structural checks for the new tagging schema (no SelectionV3 load)."""
    errors: list[str] = []
    prefix = f"{path}: " if path else ""
    if not is_free_tags_v2_raw(raw):
        errors.append(f"{prefix}policy_version must be {FREE_TAGS_V2_POLICY!r}")
    if raw.get("selection_config") or raw.get("selection_config_path"):
        errors.append(f"{prefix}selection_config must not be present on free_tags_v2")
    if not str(raw.get("evidence_config") or "").strip():
        errors.append(f"{prefix}evidence_config is required")
    if not raw.get("select_tags") and not raw.get("tags"):
        errors.append(f"{prefix}select_tags or inline tags required")
    return errors


def load_free_tags_v2_raw(path: str | Path) -> dict[str, Any]:
    p = Path(path).resolve()
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"tagging config must be mapping: {p}")
    errors = validate_free_tags_v2_raw(raw, path=str(p))
    if errors:
        raise ValueError("; ".join(errors))
    return raw


def interim_loader_requirements() -> dict[str, Any]:
    """Document the free_tags_v2 contract (now implemented by ``load_tagging_config``)."""
    return {
        "policy_version": FREE_TAGS_V2_POLICY,
        "required_keys": ["evidence_config", "evidence_mode", "select_tags"],
        "forbidden_keys": ["selection_config", "selection_config_path"],
        "default_evidence_config": DEFAULT_EVIDENCE_REL,
        "loader_note": (
            "Implemented: load_tagging_config reads evidence_config / inline evidence "
            "via TagEvidenceConfig; legacy selection_config is rejected with migrate hint."
        ),
        "status": "loader_ready",
    }
