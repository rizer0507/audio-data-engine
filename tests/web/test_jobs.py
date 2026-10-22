"""Job store / dual-claim / cancel / publish / resume behavior tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _seed_plan(store, plan_id: str = "plan_x", digest: str = "digest_x", **extra):
    store.save_plan(
        plan_id=plan_id,
        plan_digest=digest,
        task_request={"task_type": "pipeline", "input": {"kind": "source", "source_id": "s"}},
        plan={"plan_id": plan_id, "supports_resume": True, **extra},
        catalog_generation=1,
        definition_digest=extra.get("definition_digest", "d"),
        snapshot_path=extra.get("snapshot_path"),
        ttl_seconds=900,
    )
    return store.get_plan(plan_id)


def test_dual_worker_claim_is_atomic(tmp_path: Path) -> None:
    from audio_engine.web.store import JobStore

    store = JobStore(tmp_path / "web.sqlite3")
    row = _seed_plan(store)
    job_id, _ = store.create_job_from_plan(
        plan_row=row, idempotency_key=None, idempotency_hash=None
    )
    a = store.claim_next_job(worker_id="w1", lease_seconds=30)
    b = store.claim_next_job(worker_id="w2", lease_seconds=30)
    assert a is not None
    assert a["job_id"] == job_id
    assert b is None
    # Exactly one attempt row
    assert len(store.list_attempts(job_id)) == 1


def test_cancel_queued_and_running_flags(tmp_path: Path) -> None:
    from audio_engine.web.store import JobStore

    store = JobStore(tmp_path / "web.sqlite3")
    row = _seed_plan(store, "p1", "d1")
    job_id, _ = store.create_job_from_plan(
        plan_row=row, idempotency_key="k1", idempotency_hash="h1"
    )
    out = store.request_cancel(job_id)
    assert out["status"] == "cancelled"

    row2 = _seed_plan(store, "p2", "d2")
    job2, _ = store.create_job_from_plan(
        plan_row=row2, idempotency_key=None, idempotency_hash=None
    )
    claimed = store.claim_next_job(worker_id="w", lease_seconds=30)
    assert claimed is not None
    out2 = store.request_cancel(job2)
    assert out2["status"] == "cancelling"
    # Idempotent cancel
    out3 = store.request_cancel(job2)
    assert out3["status"] == "cancelling"
    assert out3["changed"] is False


def test_idempotency_conflict(tmp_path: Path) -> None:
    from audio_engine.web.store import ConflictError, JobStore

    store = JobStore(tmp_path / "web.sqlite3")
    r1 = _seed_plan(store, "p1", "d1")
    r2 = _seed_plan(store, "p2", "d2")
    store.create_job_from_plan(plan_row=r1, idempotency_key="same", idempotency_hash="h1")
    with pytest.raises(ConflictError):
        store.create_job_from_plan(plan_row=r2, idempotency_key="same", idempotency_hash="h2")


def test_late_publish_rejected_after_lease_loss(tmp_path: Path) -> None:
    from audio_engine.web.store import JobStore

    store = JobStore(tmp_path / "web.sqlite3")
    row = _seed_plan(store)
    job_id, _ = store.create_job_from_plan(
        plan_row=row, idempotency_key=None, idempotency_hash=None
    )
    claimed = store.claim_next_job(worker_id="w1", lease_seconds=30)
    assert claimed is not None
    old_token = claimed["lease_token"]
    attempt_id = claimed["attempt_id"]

    # Simulate lease takeover: rotate token on attempt (new worker identity).
    with store.connection() as conn:
        conn.execute(
            "UPDATE job_attempts SET lease_token=? WHERE attempt_id=?",
            ("new_lease_token", attempt_id),
        )

    late = store.try_publish(
        attempt_id,
        lease_token=old_token,
        artifacts=[{"path": "late.json"}],
        outcome="succeeded",
    )
    assert late["accepted"] is False
    assert late["reason"] == "LEASE_MISMATCH"

    ok = store.try_publish(
        attempt_id,
        lease_token="new_lease_token",
        artifacts=[{"path": "good.json"}],
        outcome="succeeded",
    )
    assert ok["accepted"] is True
    job = store.get_job(job_id)
    assert job["status"] == "succeeded"
    assert store.get_publication(job_id) is not None

    # Second late publisher with different attempt cannot overwrite.
    row2 = _seed_plan(store, "p_other", "d_other")
    # Force another attempt-like publish against already published job via forged attempt:
    # create a fake attempt with matching job but different id — should hit ALREADY_PUBLISHED
    with store.connection() as conn:
        conn.execute(
            """
            INSERT INTO job_attempts(
              attempt_id, job_id, attempt_no, status, worker_id, lease_token, started_at
            ) VALUES ('att_late', ?, 99, 'running', 'wX', 'tokX', datetime('now'))
            """,
            (job_id,),
        )
    late2 = store.try_publish(
        "att_late",
        lease_token="tokX",
        artifacts=[{"path": "evil.json"}],
        outcome="succeeded",
    )
    assert late2["accepted"] is False
    assert late2["reason"] == "ALREADY_PUBLISHED_BY_OTHER_ATTEMPT"


def test_publication_reconcile_after_crash(tmp_path: Path) -> None:
    from audio_engine.web.store import JobStore

    store = JobStore(tmp_path / "web.sqlite3")
    row = _seed_plan(store)
    job_id, _ = store.create_job_from_plan(
        plan_row=row, idempotency_key=None, idempotency_hash=None
    )
    claimed = store.claim_next_job(worker_id="w", lease_seconds=30)
    assert claimed is not None
    # Insert publication without finishing job (crash between publish and status update)
    with store.connection() as conn:
        conn.execute(
            """
            INSERT INTO publications(job_id, attempt_id, lease_token, published_at, outcome, artifacts_json)
            VALUES (?, ?, ?, datetime('now'), 'succeeded', '[]')
            """,
            (job_id, claimed["attempt_id"], claimed["lease_token"]),
        )
    # Job still running
    assert store.get_job(job_id)["status"] == "running"
    fixed = store.reconcile_publications()
    assert any(f["job_id"] == job_id for f in fixed)
    assert store.get_job(job_id)["status"] == "succeeded"


def test_stale_lease_alive_process_not_interrupted(tmp_path: Path) -> None:
    import os

    from audio_engine.web.store import JobStore

    store = JobStore(tmp_path / "web.sqlite3")
    row = _seed_plan(store)
    store.create_job_from_plan(plan_row=row, idempotency_key=None, idempotency_hash=None)
    claimed = store.claim_next_job(worker_id="w", lease_seconds=1)
    assert claimed is not None
    pid = os.getpid()
    token = f"{pid}:12345"
    store.set_attempt_process(
        claimed["attempt_id"],
        lease_token=claimed["lease_token"],
        pid=pid,
        process_start_token=token,
        work_dir=str(tmp_path),
        log_path=str(tmp_path / "x.log"),
    )
    # Force lease expiry
    with store.connection() as conn:
        conn.execute(
            "UPDATE job_attempts SET lease_expires_at='2000-01-01T00:00:00Z' WHERE attempt_id=?",
            (claimed["attempt_id"],),
        )
    actions = store.reconcile_stale_attempts()
    assert any(a["action"] == "leave_running" for a in actions)
    job = store.get_job(claimed["job_id"])
    assert job["status"] == "running"


def test_stale_lease_dead_process_interrupted(tmp_path: Path) -> None:
    from audio_engine.web.store import JobStore

    store = JobStore(tmp_path / "web.sqlite3")
    row = _seed_plan(store)
    store.create_job_from_plan(plan_row=row, idempotency_key=None, idempotency_hash=None)
    claimed = store.claim_next_job(worker_id="w", lease_seconds=1)
    assert claimed is not None
    # Use a very unlikely live PID with matching token format
    dead_pid = 999_999_991
    store.set_attempt_process(
        claimed["attempt_id"],
        lease_token=claimed["lease_token"],
        pid=dead_pid,
        process_start_token=f"{dead_pid}:1",
        work_dir=str(tmp_path),
        log_path=str(tmp_path / "x.log"),
    )
    with store.connection() as conn:
        conn.execute(
            "UPDATE job_attempts SET lease_expires_at='2000-01-01T00:00:00Z' WHERE attempt_id=?",
            (claimed["attempt_id"],),
        )
    actions = store.reconcile_stale_attempts()
    assert any(a["action"] == "interrupted" for a in actions)
    assert store.get_job(claimed["job_id"])["status"] == "interrupted"


def test_resume_incompatible_marker(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    import yaml
    from fastapi.testclient import TestClient

    from audio_engine.web.app import create_app
    from audio_engine.web.settings import load_web_settings

    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "RESUME_INCOMPATIBLE").write_text("blocked", encoding="utf-8")
    (snap / "definition_snapshot.digest").write_text("frozen_d", encoding="utf-8")

    cfg = {
        "bind_host": "127.0.0.1",
        "bind_port": 0,
        "project_root": str(ROOT),
        "sqlite_path": str(tmp_path / "db.sqlite3"),
        "runs_dir": str(tmp_path / "runs"),
        "plugins_config": str(ROOT / "configs" / "plugins.yaml"),
        "sources_manifest": str(tmp_path / "manifest.yaml"),
        "allowed_source_roots": [str(tmp_path)],
        "models": [],
    }
    (tmp_path / "manifest.yaml").write_text("sources: []\n", encoding="utf-8")
    cfg_path = tmp_path / "web.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    settings = load_web_settings(cfg_path, project_root=ROOT)
    app = create_app(settings)
    store = app.state.store
    row = _seed_plan(store, snapshot_path=str(snap), definition_digest="frozen_d")
    job_id, _ = store.create_job_from_plan(
        plan_row=row, idempotency_key=None, idempotency_hash=None
    )
    store.update_job_status(job_id, status="failed", expected_statuses={"queued", "running", "failed"})
    # Force failed (may already be queued only — update without expected if needed)
    with store.connection() as conn:
        conn.execute(
            "UPDATE jobs SET status='failed', snapshot_path=? WHERE job_id=?",
            (str(snap), job_id),
        )

    with TestClient(app) as client:
        r = client.post(f"/api/v1/jobs/{job_id}/retry", json={"mode": "resume"})
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "RESUME_INCOMPATIBLE"


def test_api_restart_preserves_job(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    import yaml
    from fastapi.testclient import TestClient

    from audio_engine.web.app import create_app
    from audio_engine.web.settings import load_web_settings

    cfg = {
        "bind_host": "127.0.0.1",
        "bind_port": 0,
        "project_root": str(ROOT),
        "sqlite_path": str(tmp_path / "db.sqlite3"),
        "runs_dir": str(tmp_path / "runs"),
        "plugins_config": str(ROOT / "configs" / "plugins.yaml"),
        "sources_manifest": str(tmp_path / "manifest.yaml"),
        "allowed_source_roots": [str(tmp_path)],
        "models": [],
    }
    (tmp_path / "manifest.yaml").write_text("sources: []\n", encoding="utf-8")
    cfg_path = tmp_path / "web.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    settings = load_web_settings(cfg_path, project_root=ROOT)
    app1 = create_app(settings)
    store = app1.state.store
    row = _seed_plan(store)
    job_id, _ = store.create_job_from_plan(
        plan_row=row, idempotency_key="persist-1", idempotency_hash="h"
    )

    # New app instance = API restart, same sqlite
    app2 = create_app(load_web_settings(cfg_path, project_root=ROOT))
    with TestClient(app2) as client:
        r = client.get(f"/api/v1/jobs/{job_id}")
    assert r.status_code == 200
    assert r.json()["job_id"] == job_id
    assert r.json()["status"] == "queued"


def test_harness_plugin_plan_and_execute(tmp_path: Path) -> None:
    """Small test plugin covers plan/execute without real ASR."""

    import yaml

    from audio_engine.plugins.registry import PluginRegistry

    plugins_yaml = tmp_path / "plugins.yaml"
    plugins_yaml.write_text(
        yaml.safe_dump(
            {
                "api_version": "1",
                "extension_modules": ["audio_engine.web.testing.harness_adapter"],
                "pipelines": [
                    {
                        "id": "harness_ok",
                        "version": "1",
                        "adapter": "test_harness",
                        "name": "Harness",
                        "definition": "n/a",
                        "status": "enabled",
                        "purpose": "processing",
                        "input_kinds": ["source"],
                        "supports_tag_selection": False,
                        "parameters_schema": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "mode": {"type": "string", "default": "ok"},
                            },
                        },
                    }
                ],
                "tag_libraries": [],
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    reg = PluginRegistry(project_root=ROOT)
    catalog = reg.refresh(plugins_yaml)
    assert any(e.descriptor.id == "harness_ok" for e in catalog.entries.values())

    from audio_engine.plugins.contracts import PluginContext, parse_task_request

    task = parse_task_request(
        {
            "task_type": "pipeline",
            "root_plugin": {"kind": "pipeline", "id": "harness_ok", "version": "1"},
            "input": {"kind": "source", "source_id": "s"},
            "parameters": {"mode": "ok"},
            "model_bindings": [],
            "tags": [],
            "execution": {},
        }
    )
    ctx = PluginContext(project_root=ROOT, work_dir=tmp_path, readonly=True)
    plan = reg.plan_task(task, ctx)
    assert plan.supports_cancel is True
    exec_ctx = PluginContext(
        project_root=ROOT, work_dir=tmp_path / "run", job_id="j", attempt_id="a", readonly=False
    )
    (tmp_path / "run").mkdir()
    result = reg.execute_plan(plan, exec_ctx)
    assert result.outcome == "succeeded"
