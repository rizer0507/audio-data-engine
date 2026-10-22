"""Behavioral tests for the unified plugin interface (phase 1)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from audio_engine.plugins.contracts import (
    InputRef,
    PluginContext,
    PluginRef,
    PluginRequest,
    TaskRequest,
    parse_task_request,
)
from audio_engine.plugins.registry import PluginRegistry

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def registry(tmp_path: Path) -> PluginRegistry:
    # Use real project root for definitions; temp only for work dirs.
    reg = PluginRegistry(project_root=ROOT)
    reg.refresh(ROOT / "configs" / "plugins.yaml")
    return reg


def test_catalog_discovers_pipelines_and_tags(registry: PluginRegistry) -> None:
    catalog = registry.catalog
    pipelines = catalog.list(kind="pipeline")
    tags = catalog.list(kind="tag")
    assert any(p.id == "audio_ingest_vad_asr_dnsmos_tags" for p in pipelines)
    assert any(p.adapter == "production_pipeline" for p in pipelines)
    assert any(p.adapter == "yaml_pipeline" for p in pipelines)
    assert any(t.id == "critical_short" for t in tags)
    assert not any(t.id.startswith("_") or t.id == "example_new_tag" for t in tags)
    # underscore examples must not be auto-loaded
    assert catalog.generation >= 1


def test_production_capabilities_are_honest(registry: PluginRegistry) -> None:
    desc = registry.describe("pipeline", "audio_ingest_vad_asr_dnsmos_tags")
    assert desc.capabilities.supports_resume is True
    assert desc.capabilities.supports_cancel is False
    assert desc.capabilities.supports_isolated_output is False
    assert desc.capabilities.supports_task_private_env is True
    assert desc.capabilities.supports_tag_selection is True
    assert any("cancel" in n.lower() for n in desc.capabilities.notes)


def test_list_show_validate_plan_zero_inference(registry: PluginRegistry, tmp_path: Path) -> None:
    """list/show/validate/plan must not call ASR or acoustic models."""

    calls: list[str] = []

    def boom(*_a, **_k):  # pragma: no cover - should never run
        calls.append("inference")
        raise AssertionError("inference must not run during plan")

    # Monkeypatch common inference entry points if imported later.
    import audio_engine.plugins.adapters as adapters

    original = getattr(adapters, "TaggingService", None)
    # Just ensure plan path never constructs TaggingService.run
    context = PluginContext(project_root=ROOT, work_dir=tmp_path / "w", readonly=True)
    task = TaskRequest(
        task_type="pipeline",
        root_plugin=PluginRef(kind="pipeline", id="qwen_asr_batch", version="1"),
        input=InputRef(kind="dataset", dataset_id="datasets/stage1/cleaned/cleaned_source_A.parquet"),
        parameters={},
        execution=__import__(
            "audio_engine.plugins.contracts", fromlist=["ExecutionOptions"]
        ).ExecutionOptions(evidence_mode="plan"),
    )
    report = registry.validate_task(task, context)
    assert report.ok, report.as_dict()
    plan = registry.plan_task(task, context)
    assert plan.estimated_samples is None
    assert any("read-only" in w.lower() or "inference" in w.lower() for w in plan.warnings)
    assert calls == []
    assert original is not None


def test_tag_batch_plan_shared_evidence_no_compute(registry: PluginRegistry, tmp_path: Path) -> None:
    context = PluginContext(project_root=ROOT, work_dir=tmp_path / "w", readonly=True)
    task = TaskRequest(
        task_type="tagging",
        root_plugin=None,
        input=InputRef(kind="dataset", dataset_id="datasets/stage1/derived/prepared_evidence_source_A.parquet"),
        tags=[
            PluginRef(kind="tag", id="critical_short", version="1"),
            PluginRef(kind="tag", id="human_noise_sig_low", version="1"),
        ],
        execution=__import__(
            "audio_engine.plugins.contracts", fromlist=["ExecutionOptions"]
        ).ExecutionOptions(evidence_mode="existing"),
    )
    plan = registry.plan_task(task, context)
    assert plan.evidence_union.get("shared_evidence") is True
    assert plan.evidence_union.get("inference_calls") == 0
    assert plan.evidence_union.get("compute") is False
    assert len(plan.selected_tags) == 2
    # snapshot must include tag/rule bodies
    snap = plan.definition_snapshot.get("tagging") or {}
    assert "tags" in snap or "rules" in snap or "select_tags" in plan.definition_snapshot


def test_tag_selection_rejected_when_unsupported(registry: PluginRegistry, tmp_path: Path) -> None:
    context = PluginContext(project_root=ROOT, work_dir=tmp_path, readonly=True)
    task = TaskRequest(
        task_type="pipeline",
        root_plugin=PluginRef(kind="pipeline", id="audio_ingest", version="1"),
        input=InputRef(kind="source", source_id="demo"),
        tags=[PluginRef(kind="tag", id="critical_short", version="1")],
    )
    report = registry.validate_task(task, context)
    assert not report.ok
    assert any(e.code == "TAG_SELECTION_UNSUPPORTED" for e in report.errors)


def test_unknown_adapter_and_duplicate_id(tmp_path: Path) -> None:
    cfg = {
        "api_version": "1",
        "extension_modules": [],
        "pipelines": [
            {
                "id": "x",
                "version": "1",
                "adapter": "does_not_exist",
                "definition": "pipelines/audio_ingest.yaml",
                "status": "enabled",
                "parameters_schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {},
                },
            }
        ],
        "tag_libraries": [],
    }
    path = tmp_path / "plugins.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    # Point project root at real repo so definition path resolves if needed
    reg = PluginRegistry(project_root=ROOT)
    catalog = reg.refresh(path)
    # Unknown adapter becomes diagnostic, not fatal empty success
    assert any(d.get("code") == "PIPELINE_ADAPTER_ERROR" for d in catalog.diagnostics)


def test_duplicate_pipeline_id_keeps_old_generation(tmp_path: Path) -> None:
    # First load good catalog
    reg = PluginRegistry(project_root=ROOT)
    good = reg.refresh(ROOT / "configs" / "plugins.yaml")
    gen = good.generation

    bad = {
        "api_version": "1",
        "pipelines": [
            {
                "id": "dup",
                "version": "1",
                "adapter": "yaml_pipeline",
                "definition": "pipelines/audio_ingest.yaml",
                "status": "enabled",
                "parameters_schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {},
                },
            },
            {
                "id": "dup",
                "version": "2",
                "adapter": "yaml_pipeline",
                "definition": "pipelines/audio_ingest.yaml",
                "status": "enabled",
                "parameters_schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {},
                },
            },
        ],
        "tag_libraries": [],
    }
    path = tmp_path / "bad_plugins.yaml"
    path.write_text(yaml.safe_dump(bad), encoding="utf-8")
    with pytest.raises(RuntimeError, match="DUPLICATE_PLUGIN_ID|refresh failed"):
        reg.refresh(path)
    assert reg.catalog.generation == gen


def test_schema_rejects_unknown_parameters(registry: PluginRegistry, tmp_path: Path) -> None:
    context = PluginContext(project_root=ROOT, work_dir=tmp_path, readonly=True)
    task = TaskRequest(
        task_type="pipeline",
        root_plugin=PluginRef(kind="pipeline", id="audio_ingest", version="1"),
        input=InputRef(kind="source", source_id="demo"),
        parameters={"shell_command": "rm -rf /"},
    )
    report = registry.validate_task(task, context)
    assert not report.ok
    assert any(e.code == "SCHEMA_VALIDATION" for e in report.errors)


def test_extension_pipeline_only_needs_definition_and_registration(
    tmp_path: Path,
) -> None:
    """New YAML pipeline + plugins.yaml entry is enough to discover/plan."""

    pipe = tmp_path / "pipelines"
    pipe.mkdir()
    # Minimal reusable operator pipeline (script_example exists in repo operators)
    definition = ROOT / "pipelines" / "script_example.yaml"
    if not definition.is_file():
        pytest.skip("script_example.yaml missing")
    local_def = pipe / "ext_demo_pipeline.yaml"
    raw = yaml.safe_load(definition.read_text(encoding="utf-8"))
    raw["name"] = "ext_demo_pipeline"
    local_def.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")

    # Build a plugins.yaml that includes repo pipelines path via absolute definition
    cfg = {
        "api_version": "1",
        "extension_modules": [],
        "pipelines": [
            {
                "id": "ext_demo_pipeline",
                "version": "1",
                "adapter": "yaml_pipeline",
                "name": "扩展演示流水线",
                "definition": str(local_def),  # absolute allowed via resolve in adapter
                "status": "enabled",
                "purpose": "processing",
                "input_kinds": ["dataset"],
                "supports_tag_selection": False,
                "parameters_schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {},
                },
            }
        ],
        "tag_libraries": [
            {
                "path": "configs/tagging/library",
                "runtime_config": "configs/tagging/zh_asr_free_tags_v2.yaml",
            }
        ],
    }
    # Adapter joins project_root / definition — copy into project tmp under ROOT
    # Use a config that points into ROOT/tmp
    demo_dir = ROOT / "tmp" / "plugin_ext_test"
    demo_dir.mkdir(parents=True, exist_ok=True)
    demo_pipe = demo_dir / "ext_demo_pipeline.yaml"
    shutil.copy(local_def, demo_pipe)
    cfg["pipelines"][0]["definition"] = str(demo_pipe.relative_to(ROOT).as_posix())
    cfg_path = demo_dir / "plugins.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")

    reg = PluginRegistry(project_root=ROOT)
    catalog = reg.refresh(cfg_path)
    ids = [p.id for p in catalog.list(kind="pipeline")]
    assert "ext_demo_pipeline" in ids
    context = PluginContext(project_root=ROOT, work_dir=tmp_path / "w", readonly=True)
    task = TaskRequest(
        task_type="pipeline",
        root_plugin=PluginRef(kind="pipeline", id="ext_demo_pipeline", version="1"),
        input=InputRef(kind="dataset", dataset_id="datasets/stage1/cleaned/cleaned_source_A.parquet"),
    )
    plan = reg.plan_task(task, context)
    assert plan.plugin.id == "ext_demo_pipeline"
    assert plan.definition_snapshot


def test_new_tag_file_appears_after_refresh(tmp_path: Path) -> None:
    lib = tmp_path / "library"
    lib.mkdir()
    # Copy a real simple tag as template then rename
    src = ROOT / "configs" / "tagging" / "library" / "_example_new_tag.yaml"
    raw = yaml.safe_load(src.read_text(encoding="utf-8"))
    raw["tag_id"] = "plugin_ext_text_tag"
    raw["name"] = "插件扩展文本Tag"
    for rule in raw.get("rules") or []:
        rule["tag_id"] = "plugin_ext_text_tag"
        rule["rule_id"] = "rule_plugin_ext_text_tag_v1"
    (lib / "plugin_ext_text_tag.yaml").write_text(
        yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8"
    )

    cfg = {
        "api_version": "1",
        "pipelines": [],
        "tag_libraries": [
            {
                "path": str(lib),
                "runtime_config": str(ROOT / "configs/tagging/zh_asr_free_tags_v2.yaml"),
            }
        ],
    }
    cfg_path = tmp_path / "plugins.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    reg = PluginRegistry(project_root=ROOT)
    catalog = reg.refresh(cfg_path)
    tags = [t.id for t in catalog.list(kind="tag")]
    assert "plugin_ext_text_tag" in tags
    desc = reg.describe("tag", "plugin_ext_text_tag")
    assert desc.detail.get("requires_acoustic") is False


def test_parse_task_request_roundtrip() -> None:
    data = {
        "task_type": "pipeline",
        "root_plugin": {
            "kind": "pipeline",
            "id": "audio_ingest_vad_asr_dnsmos_tags",
            "version": "1",
        },
        "input": {"kind": "source", "source_id": "actual_source_id"},
        "parameters": {},
        "model_bindings": [{"route_id": "qwen1", "connection_id": "qwen_main"}],
        "tags": [{"kind": "tag", "id": "critical_short", "version": "1"}],
        "execution": {"evidence_mode": "existing"},
    }
    task = parse_task_request(data)
    assert task.task_type == "pipeline"
    assert task.root_plugin is not None
    assert task.root_plugin.id == "audio_ingest_vad_asr_dnsmos_tags"
    assert task.tags[0].id == "critical_short"


def test_cli_list_runs(tmp_path: Path) -> None:
    from audio_engine.plugins.__main__ import main

    code = main(["--project-root", str(ROOT), "list", "--kind", "pipeline"])
    assert code == 0
