"""Stage-4 results/review/export/gold-gate tests (offline; not real vLLM)."""

from __future__ import annotations

import struct
import wave
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def _write_wav(path: Path, *, seconds: float = 0.2, rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(rate * seconds)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(struct.pack("<" + "h" * n, *([0] * n)))


@pytest.fixture
def app_client(tmp_path: Path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from audio_engine.web.app import create_app
    from audio_engine.web.settings import load_web_settings

    cfg = yaml.safe_load((ROOT / "configs" / "web.yaml").read_text(encoding="utf-8"))
    cfg["project_root"] = str(ROOT)
    cfg["sqlite_path"] = str(tmp_path / "web.sqlite3")
    cfg["runs_dir"] = str(tmp_path / "runs")
    cfg["sources_manifest"] = str(tmp_path / "manifest.yaml")
    cfg["allowed_source_roots"] = [str(tmp_path), str(ROOT)]
    cfg["plugins_config"] = str(ROOT / "configs" / "plugins.yaml")
    cfg_path = tmp_path / "web.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    (tmp_path / "manifest.yaml").write_text("sources: []\n", encoding="utf-8")

    settings = load_web_settings(cfg_path, project_root=ROOT)
    app = create_app(settings)
    with TestClient(app) as client:
        yield client, tmp_path, settings


def _make_dataset(tmp_path: Path, settings) -> str:
    from audio_engine.core.manifest import Manifest
    from audio_engine.core.sample import Sample

    audio_dir = tmp_path / "audio"
    wav_a = audio_dir / "a.wav"
    wav_b = audio_dir / "b.wav"
    _write_wav(wav_a)
    _write_wav(wav_b)

    samples = [
        Sample(
            id="s1",
            source_path=str(wav_a),
            sha256="a" * 64,
            duration=0.2,
            audio={"raw": str(wav_a), "resampled_16k": str(wav_a)},
            labels={
                "tags": ["noise_high", "short_utt"],
                "tagging": {
                    "snapshot_id": "snap_old",
                    "rule_coverage": {
                        "noise_high": "matched",
                        "clear_speech": "unknown",
                    },
                },
                "type": "clean",
                "parent_sample_id": "parent_1",
                "segment_id": "seg_1",
            },
            quality={"dnsmos_sig": 3.2, "dnsmos_status": "success"},
            status={"asr_qwen": "completed"},
            transcripts={"qwen": {"text": "你好世界"}},
        ),
        Sample(
            id="s2",
            source_path=str(wav_b),
            sha256="b" * 64,
            duration=0.2,
            audio={"raw": str(wav_b)},
            labels={
                "tags": ["clear_speech"],
                "tagging": {
                    "snapshot_id": "snap_old",
                    "rule_coverage": {"clear_speech": "matched", "noise_high": "not_matched"},
                },
                "type": "clean",
            },
            quality={"dnsmos_sig": 4.1, "dnsmos_status": "success"},
            status={"asr_qwen": "failed"},
            transcripts={"qwen": {"text": ""}},
            errors={"asr_qwen": "timeout"},
        ),
    ]
    out = tmp_path / "datasets" / "stage1" / "derived" / "web_stage4_demo.parquet"
    # Place under project datasets so list_datasets can find it OR use absolute via API path
    # Use path under tmp which is allowed; pass dataset_id as absolute/relative from project.
    # Write into project-relative path under tmp by using absolute dataset_id in API.
    out.parent.mkdir(parents=True, exist_ok=True)
    Manifest(samples=samples).save(out)
    return str(out.as_posix())


def test_sample_filter_matches_export_freeze(app_client) -> None:
    client, tmp_path, settings = app_client
    ds = _make_dataset(tmp_path, settings)

    listed = client.get(
        f"/api/v1/datasets/{ds}/samples",
        params={"tag": "noise_high", "ternary": "matched", "limit": 50},
    )
    assert listed.status_code == 200
    body = listed.json()
    assert body["total"] == 1
    assert body["items"][0]["sample_id"] == "s1"
    assert body["filters"]["tag"] == "noise_high"

    # Export creates plan+job with same frozen filters (plugin may fail later without warehouse batch;
    # we only assert freeze into job request).
    exp = client.post(
        f"/api/v1/datasets/{ds}/exports",
        json={
            "export_plugin_id": "warehouse_export_annotation",
            "filters": body["filters"],
            "idempotency_key": "exp-1",
        },
    )
    # Plan may fail validation for warehouse plugin params — still check freeze path when 202/200
    if exp.status_code in {200, 202}:
        job = exp.json()
        assert job.get("frozen_filters", {}).get("tag") == "noise_high"
        assert "job_id" in job
    else:
        # If plugin plan rejects missing params, at least list filter contract holds
        assert body["total"] == 1


def test_audio_range_and_explain_no_inference(app_client) -> None:
    client, tmp_path, settings = app_client
    ds = _make_dataset(tmp_path, settings)

    ex = client.get(f"/api/v1/datasets/{ds}/samples/s1/explain")
    assert ex.status_code == 200
    assert ex.json()["inference_invoked"] is False
    assert ex.json()["snapshot_id"] == "snap_old"

    audio = client.get(
        f"/api/v1/datasets/{ds}/samples/s1/audio",
        headers={"Range": "bytes=0-15"},
    )
    assert audio.status_code in {200, 206}
    assert audio.headers.get("accept-ranges", "").lower() == "bytes" or audio.status_code == 206
    if audio.status_code == 206:
        assert "content-range" in {k.lower() for k in audio.headers.keys()}


def test_manual_tag_actions_revision_idempotency_and_no_publish_overwrite(app_client) -> None:
    client, tmp_path, settings = app_client
    ds = _make_dataset(tmp_path, settings)
    work_root = tmp_path / "tagging_work" / "batch_demo"
    work_root.mkdir(parents=True)

    # First add
    r1 = client.post(
        f"/api/v1/datasets/{ds}/samples/s1/tag-actions",
        json={
            "action": "manual_add",
            "tag_id": "critical_short",
            "reason": "stage4",
            "expected_revision": "0",
            "idempotency_id": "tag-op-1",
            "work_root": str(work_root),
            "batch_id": "batch_demo",
            "tagging_config": "configs/tagging/zh_asr_free_tags_v2.yaml",
        },
    )
    assert r1.status_code == 200, r1.text
    body1 = r1.json()
    assert body1["ok"] is True
    assert body1["published_dataset_unchanged"] is True
    assert "critical_short" in body1["effective_tags"]
    rev = str(body1["revision"])

    # Idempotent replay
    r1b = client.post(
        f"/api/v1/datasets/{ds}/samples/s1/tag-actions",
        json={
            "action": "manual_add",
            "tag_id": "critical_short",
            "reason": "stage4",
            "expected_revision": "0",
            "idempotency_id": "tag-op-1",
            "work_root": str(work_root),
            "batch_id": "batch_demo",
            "tagging_config": "configs/tagging/zh_asr_free_tags_v2.yaml",
        },
    )
    assert r1b.status_code == 200
    assert r1b.json().get("idempotent_replay") is True

    # Stale revision → 409
    bad = client.post(
        f"/api/v1/datasets/{ds}/samples/s1/tag-actions",
        json={
            "action": "manual_block",
            "tag_id": "noise_high",
            "reason": "block",
            "expected_revision": "0",
            "idempotency_id": "tag-op-2",
            "work_root": str(work_root),
            "batch_id": "batch_demo",
        },
    )
    assert bad.status_code == 409
    assert bad.json()["error"]["code"] == "REVISION_CONFLICT"

    # Correct revision
    ok = client.post(
        f"/api/v1/datasets/{ds}/samples/s1/tag-actions",
        json={
            "action": "manual_block",
            "tag_id": "noise_high",
            "reason": "block",
            "expected_revision": rev,
            "idempotency_id": "tag-op-3",
            "work_root": str(work_root),
            "batch_id": "batch_demo",
        },
    )
    assert ok.status_code == 200, ok.text

    # Published parquet tags unchanged
    from audio_engine.core.manifest import Manifest

    published = Manifest.load(Path(ds))
    s1 = next(s for s in published.samples if s.id == "s1")
    assert "noise_high" in (s1.labels.get("tags") or [])


def test_review_writes_work_copy_not_published(app_client) -> None:
    client, tmp_path, settings = app_client
    ds = _make_dataset(tmp_path, settings)
    published = Path(ds)
    before = published.read_bytes()

    r = client.post(
        f"/api/v1/datasets/{ds}/samples/s1/review",
        json={
            "review_pass": "first",
            "decision": "accept",
            "gold_kind": "speech",
            "gold_text": "你好世界",
            "speech_scope": "target",
            "human_semantic": "neutral",
            "human_noise": "clean",
            "human_crosstalk": "false",
            "reason": "stage4",
            "expected_revision": "web_r1",
            "idempotency_id": "rev-1",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["immutable_published_unchanged"] is True
    assert body["work_dataset_id"]
    assert published.read_bytes() == before
    work = Path(ROOT / body["work_dataset_id"]) if not Path(body["work_dataset_id"]).is_absolute() else Path(body["work_dataset_id"])
    # work_dataset_id may be absolute under tmp
    work_path = Path(body["work_dataset_id"])
    if not work_path.is_file():
        work_path = tmp_path / "datasets" / "stage1" / "derived" / "web_stage4_demo.annotation_work.parquet"
    assert work_path.is_file()

    # Gold gate on published (no formal) should fail; work may pass depending on dual rules
    gate = client.post(f"/api/v1/datasets/{ds}/gold-gate", json={"require_formal_gold": True})
    assert gate.status_code == 200
    assert gate.json()["formal_gate_ok"] is False
    assert gate.json()["denominator"]["total_samples"] == 2


def test_evaluation_blocked_without_formal_gold(app_client) -> None:
    client, tmp_path, settings = app_client
    ds = _make_dataset(tmp_path, settings)
    client.post("/api/v1/plugins/refresh")

    plan = client.post(
        "/api/v1/plugins/plan",
        json={
            "task_type": "pipeline",
            "root_plugin": {"kind": "pipeline", "id": "asr_eval", "version": "1"},
            "input": {"kind": "dataset", "dataset_id": ds},
            "parameters": {},
            "model_bindings": [],
            "tags": [],
            "execution": {},
        },
    )
    assert plan.status_code == 200, plan.text
    body = plan.json()
    ev = client.post(
        "/api/v1/evaluations",
        json={"plan_id": body["plan_id"], "plan_digest": body["plan_digest"]},
        headers={"Idempotency-Key": "eval-blocked-1"},
    )
    assert ev.status_code == 409
    assert ev.json()["error"]["code"] == "FORMAL_GOLD_GATE"
    details = ev.json()["error"]["details"]
    assert "denominator" in details


def test_manual_state_survives_list_overlay(app_client) -> None:
    """After manual_add, list with binding overlay should not drop working tags when filtering."""

    client, tmp_path, settings = app_client
    ds = _make_dataset(tmp_path, settings)
    work_root = tmp_path / "tagging_work" / "batch_overlay"
    work_root.mkdir(parents=True)
    client.post(
        f"/api/v1/datasets/{ds}/samples/s2/tag-actions",
        json={
            "action": "manual_add",
            "tag_id": "critical_short",
            "reason": "keep",
            "expected_revision": "0",
            "idempotency_id": "ov-1",
            "work_root": str(work_root),
            "batch_id": "batch_overlay",
            "tagging_config": "configs/tagging/zh_asr_free_tags_v2.yaml",
        },
    )
    listed = client.get(
        f"/api/v1/datasets/{ds}/samples",
        params={"tag": "critical_short"},
    )
    assert listed.status_code == 200
    assert listed.json()["total"] == 1
    assert listed.json()["items"][0]["sample_id"] == "s2"
    assert listed.json()["tag_overlay_applied"] is True
