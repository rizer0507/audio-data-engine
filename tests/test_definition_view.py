"""Unit tests for description-only definition views (no inference)."""

from __future__ import annotations

from pathlib import Path

import yaml

from audio_engine.plugins.adapters import create_production_pipeline, create_tag_definition
from audio_engine.plugins.definition_view import build_definition_view, validate_definition_draft

ROOT = Path(__file__).resolve().parents[1]


def test_critical_short_description_is_plain() -> None:
    path = ROOT / "configs/tagging/library/critical_short.yaml"
    entry = yaml.safe_load(path.read_text(encoding="utf-8"))
    plugin = create_tag_definition(
        entry_raw=entry,
        library_root=ROOT / "configs/tagging/library",
        runtime_config=ROOT / "configs/tagging/zh_asr_free_tags_v2.yaml",
        project_root=ROOT,
        catalog_generation=1,
        source_path=str(path.relative_to(ROOT).as_posix()),
    )
    view = build_definition_view(plugin)
    summary = view["summary"]
    assert summary["kind"] == "tag"
    purpose = summary["purpose"]
    assert "关键" in purpose or "短应答" in purpose
    assert "同时满足" in purpose or "才打" in purpose
    assert "capability:" not in purpose
    assert "家族状态" not in purpose
    assert "代表文本" not in purpose
    assert view["editable"] == {"description": purpose}
    assert "rule_params" not in view["edit_schema"]["properties"]


def test_production_pipeline_description_explains_outcome() -> None:
    reg = {
        "id": "audio_ingest_vad_asr_dnsmos_tags",
        "version": "1",
        "adapter": "production_pipeline",
        "name": "入库打标",
        "description": "把原始录音切成片段，识别说话内容并检查音质，再按规则打标签，最后生成可以查看和导出的数据清单。",
        "definition": "pipelines/audio_ingest_vad_asr_dnsmos_tags.yaml",
        "status": "enabled",
        "input_kinds": ["source"],
        "output_kinds": ["manifest", "report", "annotation_package"],
        "description_updated_at": "2026-10-19T06:00:00Z",
    }
    plugin = create_production_pipeline(reg, project_root=ROOT, catalog_generation=1)
    view = build_definition_view(plugin)
    assert view["summary"]["kind"] == "pipeline"
    assert "打标签" in view["summary"]["purpose"] or "清单" in view["summary"]["purpose"]
    assert view["editable"]["description"] == view["summary"]["purpose"]
    desc = plugin.describe()
    assert desc.description_updated_at == "2026-10-19T06:00:00Z"


def test_tag_draft_description_does_not_change_execution() -> None:
    path = ROOT / "configs/tagging/library/critical_short.yaml"
    entry = yaml.safe_load(path.read_text(encoding="utf-8"))
    plugin = create_tag_definition(
        entry_raw=entry,
        library_root=ROOT / "configs/tagging/library",
        runtime_config=ROOT / "configs/tagging/zh_asr_free_tags_v2.yaml",
        project_root=ROOT,
        catalog_generation=1,
        source_path=str(path.relative_to(ROOT).as_posix()),
    )
    draft = {"description": "标记关键的简短应答。"}
    result = validate_definition_draft(plugin, draft)
    assert result["ok"] is True
    assert result["changes_execution_logic"] is False
    assert result["description_changed"] is True
    assert result["preview_entry"]["description"] == "标记关键的简短应答。"
    # Rules untouched
    assert result["preview_entry"]["rules"] == entry["rules"]


def test_reject_readonly_id_in_draft() -> None:
    path = ROOT / "configs/tagging/library/critical_short.yaml"
    entry = yaml.safe_load(path.read_text(encoding="utf-8"))
    plugin = create_tag_definition(
        entry_raw=entry,
        library_root=ROOT / "configs/tagging/library",
        runtime_config=ROOT / "configs/tagging/zh_asr_free_tags_v2.yaml",
        project_root=ROOT,
        catalog_generation=1,
        source_path=str(path.relative_to(ROOT).as_posix()),
    )
    result = validate_definition_draft(
        plugin,
        {"description": "合法说明", "tag_id": "hijack"},
    )
    assert result["ok"] is False
    assert any(e.get("code") == "EDIT_FIELD_READONLY" for e in result["field_errors"])


def test_unchanged_description_flag() -> None:
    path = ROOT / "configs/tagging/library/silence_strict_zero.yaml"
    entry = yaml.safe_load(path.read_text(encoding="utf-8"))
    plugin = create_tag_definition(
        entry_raw=entry,
        library_root=ROOT / "configs/tagging/library",
        runtime_config=ROOT / "configs/tagging/zh_asr_free_tags_v2.yaml",
        project_root=ROOT,
        catalog_generation=1,
        source_path=str(path.relative_to(ROOT).as_posix()),
    )
    same = str(entry.get("description") or "").strip()
    result = validate_definition_draft(plugin, {"description": same})
    assert result["ok"] is True
    assert result["description_changed"] is False
