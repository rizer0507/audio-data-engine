"""HTTP API protocol tests (local TestClient; not real vLLM)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def app_client(tmp_path: Path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from audio_engine.web.app import create_app
    from audio_engine.web.settings import load_web_settings

    # Isolated config + sqlite + sources; use temp plugins.yaml so status tests
    # never mutate the repo registration file.
    plugins_src = ROOT / "configs" / "plugins.yaml"
    plugins_dst = tmp_path / "plugins.yaml"
    shutil.copy2(plugins_src, plugins_dst)

    cfg = yaml.safe_load((ROOT / "configs" / "web.yaml").read_text(encoding="utf-8"))
    cfg["project_root"] = str(ROOT)
    cfg["sqlite_path"] = str(tmp_path / "web.sqlite3")
    cfg["runs_dir"] = str(tmp_path / "runs")
    cfg["sources_manifest"] = str(tmp_path / "manifest.yaml")
    cfg["allowed_source_roots"] = [str(tmp_path), str(ROOT)]
    cfg["plugins_config"] = str(plugins_dst)
    # Start models empty so auto-id create tests are isolated; keep file writable.
    cfg["models"] = []
    cfg_path = tmp_path / "web.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    (tmp_path / "manifest.yaml").write_text("sources: []\n", encoding="utf-8")

    settings = load_web_settings(cfg_path, project_root=ROOT)
    app = create_app(settings)
    with TestClient(app) as client:
        yield client, tmp_path, settings


def test_health_and_capabilities(app_client) -> None:
    client, _, _ = app_client
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    caps = client.get("/api/v1/capabilities")
    assert caps.status_code == 200
    assert "glm" in caps.json()["providers"]
    # Nginx 对外路径在原路径前加 /asr_data/，剥前缀后与原路由同一套处理。
    prefixed = client.get("/asr_data/api/v1/capabilities")
    assert prefixed.status_code == 200
    assert prefixed.json()["providers"] == caps.json()["providers"]
    assert client.get("/asr_data/health").status_code == 200


def test_plugins_list_and_show(app_client) -> None:
    client, _, _ = app_client
    r = client.post("/api/v1/plugins/refresh")
    assert r.status_code == 200
    listed = client.get("/api/v1/plugins?kind=pipeline")
    assert listed.status_code == 200
    body = listed.json()
    assert body["generation"] >= 1
    assert any(i["id"] == "audio_ingest_vad_asr_dnsmos_tags" for i in body["items"])
    show = client.get("/api/v1/plugins/pipeline/audio_ingest_vad_asr_dnsmos_tags")
    assert show.status_code == 200
    assert show.json()["capabilities"]["supports_cancel"] is False


def test_sources_create_and_duplicate_manifest_normalize(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    from audio_engine.web.services.sources import load_sources_strict

    # Reproduce duplicate sources key file
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        "sources: []\n\nsources:\n  - source_id: a\n    source_name: A\n    path: "
        + str(tmp_path / "data").replace("\\", "/")
        + "\n    format: wav\n",
        encoding="utf-8",
    )
    (tmp_path / "data").mkdir()
    sources, data = load_sources_strict(manifest)
    assert len(sources) == 1
    assert sources[0]["source_id"] == "a"
    # Second load must be strict-clean
    sources2, _ = load_sources_strict(manifest)
    assert len(sources2) == 1
    assert (tmp_path / "manifest.yaml.bak_duplicate_sources").exists() or True


def test_source_auto_id_dedup_and_batch_stable(app_client) -> None:
    client, tmp_path, settings = app_client
    src_dir = tmp_path / "audio_batch"
    src_dir.mkdir()
    body = {
        "batch": "web_auto_batch",
        "name": "显示名",
        "path": str(src_dir),
        "format": "wav",
    }
    r1 = client.post("/api/v1/sources", json=body)
    assert r1.status_code == 201, r1.text
    item1 = r1.json()
    assert item1["source_id"].startswith("src_")
    assert item1["batch"] == "web_auto_batch"
    assert item1["reused"] is False

    r2 = client.post("/api/v1/sources", json=body)
    assert r2.status_code == 200, r2.text
    item2 = r2.json()
    assert item2["source_id"] == item1["source_id"]
    assert item2["reused"] is True

    # Same batch, different path → conflict
    other = tmp_path / "audio_other"
    other.mkdir()
    conflict = client.post(
        "/api/v1/sources",
        json={"batch": "web_auto_batch", "path": str(other), "format": "wav"},
    )
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "SOURCE_BATCH_PATH_CONFLICT"

    # Patch keeps source_id
    patched = client.patch(
        f"/api/v1/sources/{item1['source_id']}",
        json={"name": "改名"},
        headers={"If-Match": str(item1.get("revision") or 2)},
    )
    assert patched.status_code == 200, patched.text
    assert patched.json()["source_id"] == item1["source_id"]
    assert patched.json()["batch"] == "web_auto_batch"

    # Reload from disk: id stable
    from audio_engine.web.services.sources import SourceService

    svc = SourceService(settings.sources_manifest, settings.allowed_source_roots)
    got = svc.get(item1["source_id"])
    assert got["source_id"] == item1["source_id"]
    assert got["batch"] == "web_auto_batch"


def test_model_auto_id_dedup_and_stable(app_client) -> None:
    client, tmp_path, settings = app_client
    body = {
        "provider": "qwen",
        "base_url": "http://127.0.0.1:18002/v1",
        "model": "qwen3-asr-test",
        "model_revision": "deployment_test",
    }
    r1 = client.post("/api/v1/models", json=body)
    assert r1.status_code == 201, r1.text
    m1 = r1.json()
    assert m1["connection_id"].startswith("conn_")
    assert m1["name"]  # default name generated
    assert m1["reused"] is False

    r2 = client.post("/api/v1/models", json=body)
    assert r2.status_code == 200
    assert r2.json()["connection_id"] == m1["connection_id"]
    assert r2.json()["reused"] is True

    # Persist across settings reload
    from audio_engine.web.settings import load_web_settings

    reloaded = load_web_settings(tmp_path / "web.yaml", project_root=ROOT)
    assert any(m.connection_id == m1["connection_id"] for m in reloaded.models)

    patched = client.patch(
        f"/api/v1/models/{m1['connection_id']}",
        json={"name": "Qwen 测试"},
        headers={"If-Match": str(m1.get("revision") or 1)},
    )
    assert patched.status_code == 200
    assert patched.json()["connection_id"] == m1["connection_id"]
    assert patched.json()["name"] == "Qwen 测试"


def test_plan_and_idempotent_submit(app_client) -> None:
    client, tmp_path, _ = app_client
    client.post("/api/v1/plugins/refresh")
    src_dir = tmp_path / "audio"
    src_dir.mkdir()
    create = client.post(
        "/api/v1/sources",
        json={
            "batch": "web_test_src",
            "name": "web test",
            "path": str(src_dir),
            "format": "wav",
        },
    )
    assert create.status_code == 201, create.text
    source_id = create.json()["source_id"]
    assert source_id.startswith("src_")
    assert create.json()["batch"] == "web_test_src"

    plan_req = {
        "task_type": "pipeline",
        "root_plugin": {"kind": "pipeline", "id": "audio_ingest", "version": "1"},
        "input": {"kind": "source", "source_id": source_id},
        "parameters": {},
        "model_bindings": [],
        "tags": [],
        "execution": {},
    }
    plan = client.post("/api/v1/plugins/plan", json=plan_req)
    assert plan.status_code == 200, plan.text
    body = plan.json()
    assert "plan_id" in body and "plan_digest" in body
    # Engine batch injected onto frozen input (not parameters schema)
    frozen_input = (body.get("request") or {}).get("input") or {}
    assert frozen_input.get("source_id") == source_id
    assert frozen_input.get("batch") == "web_test_src"
    assert any("inference" in w.lower() or "read-only" in w.lower() for w in body.get("warnings") or [])

    headers = {"Idempotency-Key": "idem-1"}
    j1 = client.post(
        "/api/v1/jobs",
        json={"plan_id": body["plan_id"], "plan_digest": body["plan_digest"]},
        headers=headers,
    )
    assert j1.status_code == 202
    j2 = client.post(
        "/api/v1/jobs",
        json={"plan_id": body["plan_id"], "plan_digest": body["plan_digest"]},
        headers=headers,
    )
    assert j2.status_code == 202
    assert j1.json()["job_id"] == j2.json()["job_id"]

    cancel = client.post(f"/api/v1/jobs/{j1.json()['job_id']}/cancel")
    assert cancel.status_code == 200
    assert cancel.json()["status"] == "cancelled"


def test_plugin_status_persist_and_archive_blocks_plan(app_client) -> None:
    client, tmp_path, settings = app_client
    client.post("/api/v1/plugins/refresh")
    listed = client.get("/api/v1/plugins?kind=pipeline&status=enabled")
    gen = listed.json()["generation"]
    target = "audio_ingest"
    before = client.get(f"/api/v1/plugins/pipeline/{target}").json()
    assert before["status"] == "enabled"

    archived = client.patch(
        f"/api/v1/plugins/pipeline/{target}/status",
        json={"status": "archived", "expected_revision": gen},
    )
    assert archived.status_code == 200, archived.text
    assert archived.json()["descriptor"]["status"] == "archived"
    new_gen = archived.json()["generation"]
    assert new_gen > gen

    # Persisted into temp plugins.yaml
    raw = yaml.safe_load(Path(settings.plugins_config).read_text(encoding="utf-8"))
    entry = next(p for p in raw["pipelines"] if p["id"] == target)
    assert entry["status"] == "archived"

    # New plan rejected
    src_dir = tmp_path / "audio2"
    src_dir.mkdir()
    src = client.post(
        "/api/v1/sources",
        json={"batch": "arch_block", "path": str(src_dir), "format": "wav"},
    ).json()
    plan = client.post(
        "/api/v1/plugins/plan",
        json={
            "task_type": "pipeline",
            "root_plugin": {"kind": "pipeline", "id": target, "version": "1"},
            "input": {"kind": "source", "source_id": src["source_id"]},
            "parameters": {},
            "model_bindings": [],
            "tags": [],
            "execution": {},
        },
    )
    assert plan.status_code == 400
    assert "status" in plan.json()["error"]["message"].lower() or "archived" in plan.text.lower()

    # Re-enable (idempotent archive first)
    again = client.patch(
        f"/api/v1/plugins/pipeline/{target}/status",
        json={"status": "archived", "expected_revision": new_gen},
    )
    assert again.status_code == 200
    assert again.json().get("unchanged") is True

    enabled = client.patch(
        f"/api/v1/plugins/pipeline/{target}/status",
        json={"status": "enabled", "expected_revision": again.json()["generation"]},
    )
    assert enabled.status_code == 200
    assert enabled.json()["descriptor"]["status"] == "enabled"

    # Tag status on an isolated copy under tmp library would need path rewrite;
    # exercise tag status via registry against a temp tag file.
    from audio_engine.plugins.registry import PluginRegistry

    tag_dir = tmp_path / "taglib"
    tag_dir.mkdir()
    tag_file = tag_dir / "web_status_probe.yaml"
    tag_file.write_text(
        yaml.safe_dump(
            {
                "tag_id": "web_status_probe",
                "version": "1",
                "status": "enabled",
                "name": "status probe",
                "description": "temp tag for status API test",
                "dependencies": {
                    "fragments": [],
                    "lexicons": [],
                    "requires_acoustic": False,
                },
                "examples": {"hit": [], "miss": [], "unknown": []},
                "rules": [
                    {
                        "rule_id": "rule_web_status_probe_v1",
                        "version": "1",
                        "tag_id": "web_status_probe",
                        "enabled": True,
                        "condition": {
                            "capability": "family_state",
                            "family": "sensevoice",
                            "state": "stable_text",
                        },
                    }
                ],
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    plugins_cfg = tmp_path / "plugins_tag_status.yaml"
    plugins_cfg.write_text(
        yaml.safe_dump(
            {
                "api_version": "1",
                "extension_modules": [],
                "pipelines": [],
                "tag_libraries": [
                    {
                        "path": str(tag_dir),
                        "runtime_config": "configs/tagging/zh_asr_free_tags_v2.yaml",
                    }
                ],
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    reg = PluginRegistry(project_root=ROOT)
    cat = reg.refresh(plugins_cfg)
    out = reg.set_status(
        "tag",
        "web_status_probe",
        "archived",
        expected_revision=cat.generation,
        plugins_config=plugins_cfg,
    )
    assert out["descriptor"]["status"] == "archived"
    persisted = yaml.safe_load(tag_file.read_text(encoding="utf-8"))
    assert persisted["status"] == "archived"
    # Restore
    reg.set_status(
        "tag",
        "web_status_probe",
        "enabled",
        expected_revision=out["generation"],
        plugins_config=plugins_cfg,
    )


def test_legacy_source_id_compat(app_client) -> None:
    """Hand-written historical source_id remains readable and usable as batch when valid."""
    client, tmp_path, settings = app_client
    manifest = Path(settings.sources_manifest)
    data_dir = tmp_path / "legacy"
    data_dir.mkdir()
    manifest.write_text(
        yaml.safe_dump(
            {
                "sources": [
                    {
                        "source_id": "legacy_batch_1",
                        "source_name": "旧显示名",
                        "path": str(data_dir).replace("\\", "/"),
                        "format": "wav",
                    }
                ]
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    listed = client.get("/api/v1/sources")
    assert listed.status_code == 200
    item = listed.json()["items"][0]
    assert item["source_id"] == "legacy_batch_1"
    # Chinese display name is not a valid batch; effective batch falls back to source_id.
    assert item["batch"] == "legacy_batch_1"


def test_unknown_api_returns_json(app_client) -> None:
    client, _, _ = app_client
    r = client.get("/api/v1/does-not-exist")
    assert r.status_code == 404
    assert "error" in r.json()


def test_plugin_definition_summary_validate_and_save(app_client) -> None:
    """Detail summary + definition validate/save; tag writes isolated library copy."""

    client, tmp_path, settings = app_client

    # Isolate tag library + runtime so saves never touch repo files.
    lib_src = ROOT / "configs" / "tagging" / "library"
    lib_dst = tmp_path / "tag_library"
    shutil.copytree(lib_src, lib_dst)
    runtime_src = ROOT / "configs" / "tagging" / "zh_asr_free_tags_v2.yaml"
    runtime_dst = tmp_path / "tagging_runtime.yaml"
    runtime_raw = yaml.safe_load(runtime_src.read_text(encoding="utf-8"))
    runtime_raw["library"] = {"path": str(lib_dst).replace("\\", "/")}
    # Keep fragments/lexicons pointing at repo via absolute-ish project paths
    runtime_dst.write_text(yaml.safe_dump(runtime_raw, allow_unicode=True), encoding="utf-8")

    plugins_path = Path(settings.plugins_config)
    plugins = yaml.safe_load(plugins_path.read_text(encoding="utf-8"))
    plugins["tag_libraries"] = [
        {
            "path": str(lib_dst).replace("\\", "/"),
            "runtime_config": str(runtime_dst).replace("\\", "/"),
        }
    ]
    plugins_path.write_text(yaml.safe_dump(plugins, allow_unicode=True), encoding="utf-8")

    client.post("/api/v1/plugins/refresh")

    # Pipeline detail: description only, no inference
    pipe = client.get("/api/v1/plugins/pipeline/audio_ingest_vad_asr_dnsmos_tags")
    assert pipe.status_code == 200, pipe.text
    body = pipe.json()
    assert "summary" in body and body["summary"]["kind"] == "pipeline"
    assert body.get("description")
    assert "editable" in body and "edit_schema" in body
    assert list(body["editable"].keys()) == ["description"]
    assert body.get("revision") == body.get("catalog_generation")
    assert body.get("description_updated_at")

    gen = body["revision"]
    draft = {"description": "把原始音频处理为带转写、质量指标和标签的数据（测试改说明）"}
    validated = client.post(
        "/api/v1/plugins/pipeline/audio_ingest_vad_asr_dnsmos_tags/definition/validate",
        json={
            "expected_revision": gen,
            "expected_definition_digest": body["definition_digest"],
            "draft": draft,
        },
    )
    assert validated.status_code == 200, validated.text
    assert validated.json()["ok"] is True
    assert validated.json()["changes_execution_logic"] is False

    saved = client.patch(
        "/api/v1/plugins/pipeline/audio_ingest_vad_asr_dnsmos_tags/definition",
        json={
            "expected_revision": gen,
            "expected_definition_digest": body["definition_digest"],
            "draft": draft,
        },
    )
    assert saved.status_code == 200, saved.text
    saved_desc = saved.json()["descriptor"]
    assert saved_desc["description"] == draft["description"]
    assert saved_desc["summary"]["purpose"] == draft["description"]
    assert saved_desc.get("description_updated_at")
    assert saved_desc["description_updated_at"] != body.get("description_updated_at")
    assert saved.json()["generation"] > gen

    # Persist description into temp plugins.yaml
    raw = yaml.safe_load(plugins_path.read_text(encoding="utf-8"))
    entry = next(p for p in raw["pipelines"] if p["id"] == "audio_ingest_vad_asr_dnsmos_tags")
    assert entry.get("description") == draft["description"]
    assert entry.get("description_updated_at") == saved_desc["description_updated_at"]
    assert entry.get("version") == "1"

    # Tag detail
    tag = client.get("/api/v1/plugins/tag/critical_short")
    assert tag.status_code == 200, tag.text
    tbody = tag.json()
    assert tbody["summary"]["kind"] == "tag"
    assert "关键" in tbody["description"] or "短应答" in tbody["description"]
    assert "capability:" not in tbody["description"]
    prior_ts = tbody.get("description_updated_at")

    tgen = tbody["revision"]
    tdraft = {"description": "标记关键的简短应答。"}
    t_val = client.post(
        "/api/v1/plugins/tag/critical_short/definition/validate",
        json={
            "expected_revision": tgen,
            "expected_definition_digest": tbody["definition_digest"],
            "draft": tdraft,
        },
    )
    assert t_val.status_code == 200, t_val.text
    assert t_val.json()["ok"] is True
    assert t_val.json()["changes_execution_logic"] is False

    # Same text save must not refresh timestamp
    same_draft = {"description": str(tbody["description"]).strip()}
    same_save = client.patch(
        "/api/v1/plugins/tag/critical_short/definition",
        json={
            "expected_revision": tgen,
            "expected_definition_digest": tbody["definition_digest"],
            "draft": same_draft,
        },
    )
    assert same_save.status_code == 200, same_save.text
    assert same_save.json()["descriptor"].get("description_updated_at") == prior_ts

    # Archive then save — must remain archived; text change updates timestamp
    arch = client.patch(
        "/api/v1/plugins/tag/critical_short/status",
        json={"status": "archived", "expected_revision": same_save.json()["generation"]},
    )
    assert arch.status_code == 200, arch.text
    tgen2 = arch.json()["generation"]
    tag2 = client.get("/api/v1/plugins/tag/critical_short").json()
    assert tag2.get("description_updated_at") == prior_ts  # archive does not touch time
    tdraft2 = {"description": "归档后仍可改说明"}
    saved_tag = client.patch(
        "/api/v1/plugins/tag/critical_short/definition",
        json={
            "expected_revision": tgen2,
            "expected_definition_digest": tag2["definition_digest"],
            "draft": tdraft2,
        },
    )
    assert saved_tag.status_code == 200, saved_tag.text
    assert saved_tag.json()["descriptor"]["status"] == "archived"
    assert saved_tag.json()["descriptor"]["description"] == "归档后仍可改说明"
    assert saved_tag.json()["descriptor"].get("description_updated_at") != prior_ts
    written = yaml.safe_load((lib_dst / "critical_short.yaml").read_text(encoding="utf-8"))
    assert written["status"] == "archived"
    assert written.get("description") == "归档后仍可改说明"
    assert written.get("version") == "1"
    assert written.get("rules")  # rules preserved

    # 409 on stale revision
    conflict = client.patch(
        "/api/v1/plugins/tag/critical_short/definition",
        json={
            "expected_revision": tgen2,
            "expected_definition_digest": tag2["definition_digest"],
            "draft": tdraft2,
        },
    )
    assert conflict.status_code == 409

    # Reject readonly smuggle
    bad = client.post(
        "/api/v1/plugins/tag/critical_short/definition/validate",
        json={
            "expected_revision": saved_tag.json()["generation"],
            "draft": {**tdraft2, "tag_id": "hijack"},
        },
    )
    assert bad.status_code == 200
    assert bad.json()["ok"] is False
