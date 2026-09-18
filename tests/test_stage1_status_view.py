"""Tests for stage-1 unified status view, aggregation, stall, and wait gates."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from audio_engine.core.manifest import Manifest
from audio_engine.core.sample import Sample
from audio_engine.core.stage1.job import Stage1JobRequest, create_job, save_job_state
from audio_engine.core.stage1.status_view import (
    IMPLEMENTATION_GAPS,
    aggregate_events,
    build_status_view,
    detect_stall,
    format_status_text,
    wait_exit_code,
)


def _filled_runtime(tmp_path: Path) -> Path:
    engine = tmp_path / "engine.py"
    engine.write_text("x", encoding="utf-8")
    vllm = tmp_path / "vllm"
    vllm.write_text("x", encoding="utf-8")
    glm_bin = tmp_path / "glm_env" / "bin"
    glm_bin.mkdir(parents=True)
    (glm_bin / "python").write_text("x", encoding="utf-8")
    (glm_bin / "vllm").write_text("x", encoding="utf-8")
    for name in ("qwen", "glm", "sensevoice"):
        d = tmp_path / "models" / name
        d.mkdir(parents=True)
        (d / "config.json").write_text("{}", encoding="utf-8")
    template = tmp_path / "t.jinja"
    template.write_text("{{x}}", encoding="utf-8")
    dnsmos = tmp_path / "d.onnx"
    dnsmos.write_bytes(b"o")
    data = {
        "authorized_gpus": [4, 5],
        "engine_python": str(engine),
        "dnsmos": {"onnx_path": str(dnsmos)},
        "families": {
            "qwen": {
                "model_path": str(tmp_path / "models" / "qwen"),
                "chat_template": str(template),
                "vllm_bin": str(vllm),
                "served_model_name": "qwen3-asr",
                "port": 5555,
                "pipeline": "pipelines/qwen_asr_batch.yaml",
            },
            "glm": {
                "model_path": str(tmp_path / "models" / "glm"),
                "env_root": str(tmp_path / "glm_env"),
                "served_model_name": "glm-asr",
                "port": 5570,
                "pipeline": "pipelines/glm_asr_batch.yaml",
                "env": {"VLLM_USE_FLASHINFER_SAMPLER": "0"},
                "unset_env": ["VLLM_ATTENTION_BACKEND"],
            },
            "sensevoice": {
                "model_path": str(tmp_path / "models" / "sensevoice"),
                "pipeline": "pipelines/sensevoice_asr_batch.yaml",
            },
        },
    }
    path = tmp_path / "server.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _job(tmp_path: Path):
    runtime = _filled_runtime(tmp_path)
    source = tmp_path / "wavs"
    source.mkdir()
    req = Stage1JobRequest(
        batch="status-demo",
        source=str(source),
        source_kind="directory",
        models={
            "qwen": str(tmp_path / "models" / "qwen"),
            "glm": str(tmp_path / "models" / "glm"),
            "sensevoice": str(tmp_path / "models" / "sensevoice"),
        },
        runtime_config=str(runtime),
        gpus=["4", "5"],
        jobs_root=str(tmp_path / "jobs"),
    )
    return create_job(req)


def test_gaps_declare_step3_step4_code_ready_server_pending():
    assert IMPLEMENTATION_GAPS["step3_resume_retry"] is True
    assert IMPLEMENTATION_GAPS["step3_failed_only_requeue"] is True
    assert IMPLEMENTATION_GAPS["step4_dual_gpu_scheduler"] is True
    assert IMPLEMENTATION_GAPS["step4_gpu_lease"] is True
    assert IMPLEMENTATION_GAPS["server_dual_gpu_benchmark"] is False
    assert IMPLEMENTATION_GAPS["server_e2e_accepted"] is False


def test_progress_capped_before_reconcile(tmp_path: Path):
    root, state = _job(tmp_path)
    for name, stage in state.stages.items():
        if name != "reconcile":
            stage.status = "succeeded"
    state.status = "running"
    state.reconcile = {}
    save_job_state(root, state)
    view = build_status_view(root)
    assert view.batch_succeeded is False
    assert view.progress_pct is not None and view.progress_pct < 100
    assert "对账通过前" in format_status_text(view) or view.progress_pct <= 99


def test_wait_exit_requires_reconcile_ok(tmp_path: Path):
    root, state = _job(tmp_path)
    state.status = "succeeded"
    state.reconcile = {"ok": False, "errors": ["missing"]}
    save_job_state(root, state)
    view = build_status_view(root)
    assert view.batch_succeeded is False
    assert wait_exit_code(view) == 1

    state.reconcile = {"ok": True}
    save_job_state(root, state)
    view = build_status_view(root)
    assert view.batch_succeeded is True
    assert wait_exit_code(view) == 0
    assert view.progress_pct == 100.0


def test_wait_exit_needs_attention(tmp_path: Path):
    root, state = _job(tmp_path)
    state.status = "needs_attention"
    state.reconcile = {"ok": False, "errors": ["缺路"]}
    state.needs_attention = [{"kind": "missing_asr_routes"}]
    save_job_state(root, state)
    view = build_status_view(root)
    assert wait_exit_code(view) == 3


def test_aggregate_events_dedup(tmp_path: Path):
    events = tmp_path / "events.jsonl"
    lines = [
        {"at": "t1", "event": "stage", "stage": "asr_qwen_1", "status": "failed", "error": "boom 1"},
        {"at": "t2", "event": "stage", "stage": "asr_qwen_1", "status": "failed", "error": "boom 2"},
        {"at": "t3", "event": "stage", "stage": "asr_qwen_1", "status": "failed", "error": "boom 9"},
    ]
    events.write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in lines) + "\n",
        encoding="utf-8",
    )
    agg = aggregate_events(events)
    assert len(agg) == 1
    assert agg[0].count == 3


def test_stall_detection(tmp_path: Path, monkeypatch):
    root, state = _job(tmp_path)
    state.status = "running"
    state.pid = 1
    state.updated_at = "2000-01-01T00:00:00+00:00"
    for stage in state.stages.values():
        stage.started_at = "2000-01-01T00:00:00+00:00"
        stage.finished_at = None
    # Bypass save_job_state which refreshes updated_at to now.
    (root / "state.json").write_text(
        json.dumps(state.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (root / "events.jsonl").write_text("", encoding="utf-8")
    import os

    old = 946684800  # 2000-01-01
    os.utime(root / "events.jsonl", (old, old))
    monkeypatch.setattr(
        "audio_engine.core.stage1.status_view.pid_is_alive",
        lambda pid: True,
    )
    # Reload state from disk for detect_stall caller consistency
    from audio_engine.core.stage1.job import load_job_state

    loaded = load_job_state(root)
    stall = detect_stall(loaded, root, stall_timeout_s=1.0)
    assert stall["stalled"] is True
    assert stall["worker_alive"] is True


def test_family_coverage_and_json_gaps(tmp_path: Path):
    root, state = _job(tmp_path)
    batch = state.batch
    import audio_engine.core.stage1.status_view as sv

    cleaned_dir = tmp_path / "cleaned"
    asr_dir = tmp_path / "asr"
    cleaned_dir.mkdir(parents=True)
    asr_dir.mkdir(parents=True)
    Manifest(
        [Sample(id="s1", source_path="a.wav", sha256="h1", labels={"original_audio_sha256": "h1"})]
    ).save(cleaned_dir / f"cleaned_{batch}.parquet")
    original_cleaned = sv.STAGE1_CLEANED_DIR
    original_asr = sv.STAGE1_ASR_DIR
    sv.STAGE1_CLEANED_DIR = cleaned_dir
    sv.STAGE1_ASR_DIR = asr_dir
    try:
        Manifest(
            [Sample(id="s1", source_path="a.wav", sha256="h1", labels={"original_audio_sha256": "h1"})]
        ).save(asr_dir / f"qwen_1_asr_{batch}.parquet")
        state.stages["asr_qwen_1"].status = "succeeded"
        state.stages["register_qwen_1"].status = "succeeded"
        save_job_state(root, state)
        view = build_status_view(root)
        assert view.family_coverage["qwen"]["success_runs"] == 1
        assert view.family_coverage["qwen"]["expected_runs"] == 2
        assert view.gaps["implementation"]["step3_resume_retry"] is True
        assert view.gaps["implementation"]["step4_dual_gpu_scheduler"] is True
        assert view.retry_pending is not None
        text = format_status_text(view)
        assert "step3_resume_retry=True" in text
        assert "step4_dual_gpu_scheduler=True" in text
        assert "server_e2e_accepted=False" in text
    finally:
        sv.STAGE1_CLEANED_DIR = original_cleaned
        sv.STAGE1_ASR_DIR = original_asr
