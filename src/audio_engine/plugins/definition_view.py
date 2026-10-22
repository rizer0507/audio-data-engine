"""Description-only view/edit for plugin definitions.

Never calls LLMs or inference. Editing description does not change rules,
pipeline steps, status, or business version.
"""

from __future__ import annotations

import copy
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from audio_engine.plugins.schema_util import validate_against_schema

_EDIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "title": "解释编辑",
    "properties": {
        "description": {
            "type": "string",
            "title": "解释",
            "minLength": 1,
            "description": "普通人能看懂的说明；不改变实际判定或执行过程",
        },
    },
    "required": ["description"],
}

_FORBIDDEN_DRAFT_KEYS = frozenset(
    {
        "id",
        "tag_id",
        "adapter",
        "definition",
        "definition_path",
        "status",
        "version",
        "module",
        "command",
        "name",
        "purpose",
        "notes",
        "rule_params",
        "input_note",
        "output_note",
        "flow_notes",
        "default_parameters",
        "step_params",
        "display",
        "description_updated_at",
        "rules",
        "business_definition",
    }
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def normalize_description(text: Any) -> str:
    return str(text or "").strip()


def read_description_updated_at(raw: dict[str, Any] | None) -> str | None:
    if not isinstance(raw, dict):
        return None
    value = raw.get("description_updated_at")
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _source_raw(plugin: Any) -> dict[str, Any]:
    if hasattr(plugin, "entry_raw"):
        return dict(plugin.entry_raw or {})
    if hasattr(plugin, "registration"):
        return dict(plugin.registration or {})
    return {}


def build_tag_definition_view(plugin: Any) -> dict[str, Any]:
    entry = dict(plugin.entry_raw or {})
    description = normalize_description(entry.get("description"))
    name = str(entry.get("name") or entry.get("tag_id") or "")
    return {
        "summary": {
            "kind": "tag",
            "name": name,
            "purpose": description,
        },
        "editable": {"description": description},
        "edit_schema": copy.deepcopy(_EDIT_SCHEMA),
        "readonly_fields": [
            "tag_id",
            "name",
            "version",
            "status",
            "rules",
            "description_updated_at",
        ],
    }


def build_pipeline_definition_view(plugin: Any) -> dict[str, Any]:
    reg = dict(plugin.registration or {})
    description = normalize_description(reg.get("description"))
    name = str(reg.get("name") or reg.get("id") or "")
    return {
        "summary": {
            "kind": "pipeline",
            "name": name,
            "purpose": description,
        },
        "editable": {"description": description},
        "edit_schema": copy.deepcopy(_EDIT_SCHEMA),
        "readonly_fields": [
            "id",
            "name",
            "version",
            "status",
            "adapter",
            "definition",
            "description_updated_at",
            "DAG / 步骤参数",
        ],
    }


def build_definition_view(plugin: Any) -> dict[str, Any]:
    if hasattr(plugin, "entry_raw"):
        return build_tag_definition_view(plugin)
    if hasattr(plugin, "registration"):
        return build_pipeline_definition_view(plugin)
    return {
        "summary": {"kind": "unknown", "purpose": "", "name": ""},
        "editable": {},
        "edit_schema": {"type": "object", "properties": {}},
        "readonly_fields": [],
    }


def enrich_descriptor_dict(desc_dict: dict[str, Any], plugin: Any) -> dict[str, Any]:
    view = build_definition_view(plugin)
    out = dict(desc_dict)
    out["revision"] = out.get("catalog_generation") or 0
    out["summary"] = view["summary"]
    out["editable"] = view["editable"]
    out["edit_schema"] = view["edit_schema"]
    out["readonly_fields"] = view.get("readonly_fields") or []
    raw = _source_raw(plugin)
    # Prefer adapter-provided field; fall back to source config.
    if not out.get("description_updated_at"):
        out["description_updated_at"] = read_description_updated_at(raw)
    # Ensure description shown in dialog is the persisted explanation body.
    if hasattr(plugin, "entry_raw"):
        out["description"] = normalize_description(raw.get("description"))
    elif hasattr(plugin, "registration"):
        out["description"] = normalize_description(raw.get("description"))
    return out


def _reject_forbidden(draft: dict[str, Any]) -> list[dict[str, Any]]:
    smuggled = sorted(set(draft) & _FORBIDDEN_DRAFT_KEYS)
    if not smuggled:
        return []
    return [
        {
            "code": "EDIT_FIELD_READONLY",
            "message": f"cannot edit readonly fields: {', '.join(smuggled)}",
            "field_path": smuggled[0],
        }
    ]


def validate_tag_draft(plugin: Any, draft: dict[str, Any]) -> dict[str, Any]:
    field_errors = _reject_forbidden(draft)
    view = build_tag_definition_view(plugin)
    if not field_errors:
        report = validate_against_schema(draft, view["edit_schema"], plugin_ref=None)
        field_errors = [e.as_dict() for e in report.errors]
        warnings = [w.as_dict() for w in report.warnings]
    else:
        warnings = []

    entry = copy.deepcopy(dict(plugin.entry_raw))
    new_desc = normalize_description(draft.get("description"))
    if not new_desc and not any(e.get("field_path") == "description" for e in field_errors):
        field_errors.append(
            {
                "code": "DESCRIPTION_EMPTY",
                "message": "解释不能为空",
                "field_path": "description",
            }
        )
    if new_desc:
        entry["description"] = new_desc
    # Timestamp is applied only on real save when text changes.
    ok = len(field_errors) == 0
    return {
        "ok": ok,
        "field_errors": field_errors,
        "warnings": warnings,
        "changes_execution_logic": False,
        "affected_references": [],
        "normalized_draft": {"description": new_desc},
        "preview_entry": entry if ok else None,
        "description_changed": new_desc != normalize_description(plugin.entry_raw.get("description")),
    }


def validate_pipeline_draft(plugin: Any, draft: dict[str, Any]) -> dict[str, Any]:
    field_errors = _reject_forbidden(draft)
    view = build_pipeline_definition_view(plugin)
    if not field_errors:
        report = validate_against_schema(draft, view["edit_schema"], plugin_ref=None)
        field_errors = [e.as_dict() for e in report.errors]
        warnings = [w.as_dict() for w in report.warnings]
    else:
        warnings = []

    reg = copy.deepcopy(dict(plugin.registration))
    new_desc = normalize_description(draft.get("description"))
    if not new_desc and not any(e.get("field_path") == "description" for e in field_errors):
        field_errors.append(
            {
                "code": "DESCRIPTION_EMPTY",
                "message": "解释不能为空",
                "field_path": "description",
            }
        )
    if new_desc:
        reg["description"] = new_desc
    # Drop legacy display.* explanation mirrors; description is the single body.
    if isinstance(reg.get("display"), dict):
        display = dict(reg["display"])
        display.pop("purpose", None)
        if not display:
            reg.pop("display", None)
        else:
            reg["display"] = display

    ok = len(field_errors) == 0
    return {
        "ok": ok,
        "field_errors": field_errors,
        "warnings": warnings,
        "changes_execution_logic": False,
        "affected_references": [],
        "normalized_draft": {"description": new_desc},
        "preview_registration": reg if ok else None,
        "preview_meta": None,
        "description_changed": new_desc
        != normalize_description(plugin.registration.get("description")),
    }


def validate_definition_draft(plugin: Any, draft: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(draft, dict):
        return {
            "ok": False,
            "field_errors": [
                {"code": "DRAFT_INVALID", "message": "draft must be an object"}
            ],
            "warnings": [],
            "changes_execution_logic": False,
            "affected_references": [],
        }
    if hasattr(plugin, "entry_raw"):
        return validate_tag_draft(plugin, draft)
    if hasattr(plugin, "registration"):
        return validate_pipeline_draft(plugin, draft)
    return {
        "ok": False,
        "field_errors": [
            {
                "code": "ADAPTER_UNSUPPORTED",
                "message": "plugin adapter has no definition editor",
            }
        ],
        "warnings": [],
        "changes_execution_logic": False,
        "affected_references": [],
    }


def apply_description_timestamp(
    *,
    preview: dict[str, Any],
    previous: dict[str, Any],
    changed: bool,
) -> None:
    """Mutate preview: set UTC timestamp only when explanation text changed."""

    if changed:
        preview["description_updated_at"] = utc_now_iso()
        return
    existing = read_description_updated_at(previous)
    if existing:
        preview["description_updated_at"] = existing
    else:
        preview.pop("description_updated_at", None)


def bump_version_string(version: str) -> str:
    text = str(version or "1").strip() or "1"
    if re.fullmatch(r"\d+", text):
        return str(int(text) + 1)
    m = re.fullmatch(r"(.*?)(\d+)$", text)
    if m:
        return f"{m.group(1)}{int(m.group(2)) + 1}"
    return f"{text}.1"


def find_select_tag_refs(
    project_root: Path,
    runtime_configs: list[Path],
    *,
    tag_id: str,
    version: str,
) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for cfg in runtime_configs:
        path = cfg if cfg.is_absolute() else (project_root / cfg)
        if not path.is_file():
            continue
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:  # noqa: BLE001
            continue
        select = raw.get("select_tags") or []
        if not isinstance(select, list):
            continue
        for i, item in enumerate(select):
            if not isinstance(item, dict):
                continue
            if str(item.get("tag_id") or "") == tag_id and str(item.get("version") or "") == version:
                refs.append(
                    {
                        "path": str(path.relative_to(project_root).as_posix())
                        if path.is_relative_to(project_root)
                        else str(path),
                        "index": i,
                        "tag_id": tag_id,
                        "version": version,
                    }
                )
    return refs
