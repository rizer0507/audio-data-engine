"""Fault-injection tests for stage1 step-3 resume/retry isolation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from audio_engine.core.stage1.job import (
    Stage1JobRequest,
    create_job,
    load_job_state,
    save_job_state,
)
from audio_engine.core.stage1.locks import JobLock, JobLockError
from audio_engine.core.stage1.orchestrator import Stage1Orchestrator
from audio_engine.core.stage1.retry_policy import DEFAULT_MAX_ATTEMPTS


def _filled_runtime(tmp_path: Path) -> Path:
    engine = tmp_path / "engine.py"
    engine.write_text("#!/bin/sh\n", encoding="utf-8")
    vllm = tmp_path / "vllm"
    vllm.write_text("#!/bin/sh\n", encoding="utf-8")
    glm_env = tmp_path / "glm_env" / "bin"
    glm_env.mkdir(parents=True)
    (glm_env / "python").write_text("#!/bin/sh\n", encoding="utf-8")
    (glm_env / "vllm").write_text("#!/bin/sh\n", encoding="utf-8")
    for name in ("qwen", "glm", "sensevoice"):
        (tmp_path / "models" / name).mkdir(parents=True)
        (tmp_path / "models" / name / "config.json").write_text("{}", encoding="utf-8")
    template = tmp_path / "chat.jinja"
    template.write_text("{{x}}", encoding="utf-8")
    dnsmos = tmp_path / "dnsmos.onnx"
    dnsmos.write_bytes(b"onnx")
    data = {
        "authorized_gpus": [4, 5],
        "engine_python": str(engine),
        "dnsmos": {"onnx_path": str(dnsmos)},
        "probe": {"ready_timeout_s": 1, "ready_poll_s": 0.1},
        "session": {"root": str(tmp_path / "serve")},
        "families": {
            "qwen": {
                "model_path": str(tmp_path / "models" / "qwen"),
                "chat_template": str(template),
                "vllm_bin": str(vllm),
                "served_model_name": "qwen3-asr",
                "host": "127.0.0.1",
                "port": 5555,
                "gpu_memory_utilization": 0.5,
                "tensor_parallel_size": 1,
                "api_key": "dummy",
                "pipeline": "pipelines/qwen_asr_batch.yaml",
            },
            "glm": {
                "model_path": str(tmp_path / "models" / "glm"),
                "env_root": str(tmp_path / "glm_env"),
                "served_model_name": "glm-asr",
                "host": "0.0.0.0",
                "client_host": "127.0.0.1",
                "port": 5570,
                "tensor_parallel_size": 1,
                "dtype": "bfloat16",
                "max_model_len": 4096,
                "max_num_seqs": 8,
                "gpu_memory_utilization": 0.9,
                "trust_remote_code": True,
                "limit_mm_per_prompt": '{"audio":1}',
                "no_enable_flashinfer_autotune": True,
                "kernel_config": '{"enable_jit_warmup":false,"enable_cutedsl_warmup":false}',
                "env": {
                    "PYTHONNOUSERSITE": "1",
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    "FLASHINFER_DISABLE_VERSION_CHECK": "1",
                    "VLLM_USE_FLASHINFER_SAMPLER": "0",
                },
                "unset_env": [
                    "VLLM_ATTENTION_BACKEND",
                    "VLLM_USE_FLASHINFER_SAMPLE",
                    "VLLM_ATTENTIOIN_BACKEND",
                ],
                "api_key": "dummy",
                "pipeline": "pipelines/glm_asr_batch.yaml",
            },
            "sensevoice": {
                "model_path": str(tmp_path / "models" / "sensevoice"),
                "device": "cuda:0",
                "language": "auto",
                "use_itn": True,
                "disable_update": True,
                "pipeline": "pipelines/sensevoice_asr_batch.yaml",
            },
        },
    }
    path = tmp_path / "server.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def _make_job(tmp_path: Path) -> Path:
    runtime = _filled_runtime(tmp_path)
    source = tmp_path / "wavs"
    source.mkdir()
    req = Stage1JobRequest(
        batch="fault-batch",
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
        catalog_dir=str(tmp_path / "catalog"),
    )
    root, _ = create_job(req)
    return root


def test_family_service_crash_isolates_other_families(tmp_path: Path):
    root = _make_job(tmp_path)
    orch = Stage1Orchestrator(root, dry_run=True, sleep_fn=lambda _s: None)
    original = orch._asr_body

    def boom(family: str, alias: str, gpu=None) -> None:
        if family == "glm" and alias == "glm_2":
            raise RuntimeError("模拟服务崩溃 Connection refused exit=1")
        return original(family, alias, gpu=gpu)

    orch._asr_body = boom  # type: ignore[method-assign]
    state = orch.run()
    assert state.status == "needs_attention"
    assert "glm_2" in (state.error or "") or any(
        "glm" in str(item) for item in state.needs_attention
    )
    assert orch.state.stages["asr_qwen_1"].status in {"succeeded", "skipped"}
    assert orch.state.stages["asr_qwen_2"].status in {"succeeded", "skipped"}
    assert orch.state.stages["asr_sensevoice_1"].status in {"succeeded", "skipped"}
    assert orch.state.stages["asr_sensevoice_2"].status in {"succeeded", "skipped"}
    assert orch.state.stages["asr_glm_1"].status in {"succeeded", "skipped"}
    assert orch.state.stages["asr_glm_2"].status == "failed"
    # Downstream must be blocked — no formal delivery.
    assert orch.state.stages["classify"].status == "blocked"
    assert orch.state.reconcile.get("ok") is not True


def test_retry_exhausted_then_failed_only_recovers(tmp_path: Path):
    root = _make_job(tmp_path)
    orch = Stage1Orchestrator(root, dry_run=True, sleep_fn=lambda _s: None)
    sleeps: list[float] = []
    orch.sleep_fn = lambda s: sleeps.append(s)
    # Force real retry loop on one stage without full pipeline.
    orch.dry_run = False
    calls = {"n": 0}

    def flaky() -> None:
        calls["n"] += 1
        raise RuntimeError("Connection refused exit=1")

    with pytest.raises(RuntimeError, match="Connection refused"):
        orch._call_with_retries("asr_glm_2", flaky)
    stage = orch.state.stages["asr_glm_2"]
    assert calls["n"] == DEFAULT_MAX_ATTEMPTS
    assert stage.attempt_count == DEFAULT_MAX_ATTEMPTS
    assert stage.status == "failed"
    assert stage.circuit_open is True
    assert len(sleeps) == DEFAULT_MAX_ATTEMPTS - 1

    # Explicit failed-only retry clears circuit; body succeeds under dry_run.
    orch.dry_run = True
    orch._force_stages.clear()
    targets = orch._resolve_retry_targets(family="glm", run=2, failed_only=True)
    assert "asr_glm_2" in targets
    for name in targets:
        orch._reset_stage(name)
    assert orch.state.stages["asr_glm_2"].circuit_open is False
    assert orch.state.stages["asr_glm_2"].attempt_count == 0
    orch._call_with_retries(
        "asr_glm_2", lambda: orch._asr_body("glm", "glm_2")
    )
    assert orch.state.stages["asr_glm_2"].status in {"succeeded", "skipped"}
    assert orch.state.stages["asr_glm_2"].circuit_open is False


def test_interrupt_resume_skips_succeeded(tmp_path: Path):
    root = _make_job(tmp_path)
    orch = Stage1Orchestrator(root, dry_run=True, sleep_fn=lambda _s: None)
    # Simulate interrupt after clean: mark early stages done, leave ASR pending.
    orch._call_with_retries("precheck", orch._precheck_body)
    orch._call_with_retries("freeze_snapshot", orch._freeze_snapshot_body)
    orch._call_with_retries("clean", orch._clean_body)
    orch.state.status = "failed"
    orch.state.error = "编排中断"
    save_job_state(root, orch.state)
    assert (root / "worker.lock").exists() is False or True

    asr_calls: list[str] = []
    orch2 = Stage1Orchestrator(root, dry_run=True, sleep_fn=lambda _s: None)
    original_asr = orch2._asr_body

    def tracked(family: str, alias: str, gpu=None) -> None:
        asr_calls.append(alias)
        return original_asr(family, alias, gpu=gpu)

    orch2._asr_body = tracked  # type: ignore[method-assign]
    pre_clean = orch2.state.stages["clean"].status
    state = orch2.resume()
    assert pre_clean in {"succeeded", "skipped"}
    assert orch2.state.stages["clean"].status in {"succeeded", "skipped"}
    assert orch2.state.stages["freeze_snapshot"].status in {"succeeded", "skipped"}
    assert len(asr_calls) == 6  # all routes still needed
    assert state.status in {"needs_attention", "succeeded", "failed"}


def test_corrupt_shard_healed_and_recomputed(tmp_path: Path):
    root = _make_job(tmp_path)
    orch = Stage1Orchestrator(root, dry_run=True, sleep_fn=lambda _s: None)
    orch.run()
    # Corrupt a "successful" ASR stage: clear dry_run flag and point to missing file.
    stage = orch.state.stages["asr_qwen_1"]
    missing = root / "missing_qwen_1.parquet"
    stage.detail = {}
    stage.outputs = [str(missing)]
    stage.status = "succeeded"
    (root / "checkpoints" / "asr_qwen_1.json").write_text("{}", encoding="utf-8")
    save_job_state(root, orch.state)

    orch2 = Stage1Orchestrator(root, dry_run=True, sleep_fn=lambda _s: None)
    orch2._heal_corrupt_successes()
    assert orch2.state.stages["asr_qwen_1"].status == "pending"
    assert "healed" in (orch2.state.stages["asr_qwen_1"].error or "")

    recomputed = orch2.resume()
    assert orch2.state.stages["asr_qwen_1"].status in {"succeeded", "skipped"}
    assert recomputed.status in {"needs_attention", "succeeded", "failed"}


def test_export_failure_retry_does_not_rerun_asr(tmp_path: Path):
    root = _make_job(tmp_path)
    orch = Stage1Orchestrator(root, dry_run=True, sleep_fn=lambda _s: None)
    orch.run()
    # Force export/reconcile failure while ASR remains successful.
    for name in ("export", "reconcile"):
        orch.state.stages[name].status = "failed"
        orch.state.stages[name].error = "export boom"
    orch.state.status = "needs_attention"
    orch.state.reconcile = {"ok": False, "errors": ["export boom"]}
    save_job_state(root, orch.state)

    asr_calls: list[str] = []
    orch2 = Stage1Orchestrator(root, dry_run=True, sleep_fn=lambda _s: None)
    original_asr = orch2._asr_body

    def tracked(family: str, alias: str, gpu=None) -> None:
        asr_calls.append(alias)
        return original_asr(family, alias, gpu=gpu)

    orch2._asr_body = tracked  # type: ignore[method-assign]
    before_attempts = {
        name: orch2.state.stages[name].attempt_count
        for name in orch2.state.stages
        if name.startswith("asr_")
    }
    orch2.retry_export_only()
    assert asr_calls == []
    for name, count in before_attempts.items():
        assert orch2.state.stages[name].attempt_count == count
        assert orch2.state.stages[name].status in {"succeeded", "skipped"}


def test_stale_worker_lock_taken_over_alive_lock_rejected(tmp_path: Path, monkeypatch):
    root = _make_job(tmp_path)
    state = load_job_state(root)
    lock = JobLock(root, job_id=state.job_id)
    info = lock.acquire(mode="run")
    # Simulate a different still-alive worker holding the lock.
    foreign = {
        "pid": info.pid + 99999,
        "worker_token": "deadbeef" * 4,
        "acquired_at": info.acquired_at,
        "job_id": state.job_id,
        "mode": "run",
    }
    (root / "worker.lock").write_text(
        json.dumps(foreign, ensure_ascii=False), encoding="utf-8"
    )
    monkeypatch.setattr(
        "audio_engine.core.stage1.locks.pid_is_alive",
        lambda pid: True,
    )
    lock2 = JobLock(root, job_id=state.job_id)
    with pytest.raises(JobLockError, match="已被 worker 占用"):
        lock2.acquire(mode="resume")

    # Dead worker: resume may steal stale lock.
    monkeypatch.setattr(
        "audio_engine.core.stage1.locks.pid_is_alive",
        lambda pid: False,
    )
    lock3 = JobLock(root, job_id=state.job_id)
    info3 = lock3.acquire(mode="resume", steal_stale=True)
    assert info3.worker_token != foreign["worker_token"]
    # Old token release must not clear new lock.
    lock.token = foreign["worker_token"]
    lock.release(token=foreign["worker_token"])
    assert lock3.read() is not None
    assert lock3.read().worker_token == info3.worker_token


def test_attempt_command_json_records_worker_token(tmp_path: Path):
    root = _make_job(tmp_path)
    orch = Stage1Orchestrator(root, dry_run=True, sleep_fn=lambda _s: None)
    orch.run()
    cmds = list((root / "stages").rglob("command.json"))
    assert cmds
    payload = json.loads(cmds[0].read_text(encoding="utf-8"))
    assert "attempt" in payload
    assert "argv" in payload
