"""Tests for stage1 dual-GPU lease scheduler (step 4)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from audio_engine.core.stage1.gpu_inventory import (
    GpuDevice,
    GpuProcess,
    bind_authorized_gpus,
    can_admit_family,
    families_compatible_on_same_gpu,
    family_vram_budget_mib,
)
from audio_engine.core.stage1.gpu_lease import GpuLeaseError, GpuLeaseStore
from audio_engine.core.stage1.gpu_scheduler import (
    DualGpuScheduler,
    build_dual_gpu_plan,
    build_serial_baseline_plan,
)
from audio_engine.core.stage1.job import Stage1JobRequest, create_job
from audio_engine.core.stage1.orchestrator import Stage1Orchestrator
from audio_engine.core.stage1.status_view import IMPLEMENTATION_GAPS


def _device(
    index: str,
    *,
    used: float = 100.0,
    total: float = 81920.0,
    util: float = 0.0,
    procs: list[GpuProcess] | None = None,
) -> GpuDevice:
    return GpuDevice(
        index=index,
        uuid=f"GPU-uuid-{index}",
        memory_used_mib=used,
        memory_total_mib=total,
        utilization_gpu=util,
        utilization_memory=0.0,
        processes=tuple(procs or ()),
        collect_ok=True,
        note="util ignored",
    )


def test_util_zero_does_not_imply_idle():
    # Nearly full VRAM but util=0 — must reject heavy admission.
    device = _device("4", used=78000.0, total=81920.0, util=0.0)
    ok, reason = can_admit_family(
        device, "glm", gpu_memory_utilization=0.90, allow_unknown=False
    )
    assert ok is False
    assert "vram" in reason or "insufficient" in reason


def test_qwen_glm_not_coresident():
    assert families_compatible_on_same_gpu("qwen", "glm") is False
    assert families_compatible_on_same_gpu("qwen", "qwen") is True


def test_uuid_binding_detects_drift():
    snap = [_device("4"), _device("5")]
    binding = bind_authorized_gpus(
        request_gpus=["4", "5"],
        authorized_gpus=(4, 5),
        authorized_uuids=("GPU-uuid-4", "WRONG"),
        snapshot=snap,
    )
    assert any("漂移" in e for e in binding.errors)
    assert binding.index_to_uuid["4"] == "GPU-uuid-4"


def test_foreign_process_blocks_heavy():
    device = _device(
        "4",
        used=500.0,
        procs=[GpuProcess(pid=4242, used_memory_mib=1000, name="other")],
    )
    ok, reason = can_admit_family(device, "qwen", owned_pids=set(), allow_unknown=False)
    assert ok is False
    assert "foreign_process" in reason


def test_lease_rejects_other_job(tmp_path: Path, monkeypatch):
    store = GpuLeaseStore(tmp_path / "leases")
    monkeypatch.setattr(
        "audio_engine.core.stage1.gpu_lease.pid_is_alive", lambda pid: True
    )
    store.acquire(
        "4",
        job_id="job-a",
        worker_token="token-a",
        family="qwen",
        gpu_uuid="GPU-uuid-4",
    )
    with pytest.raises(GpuLeaseError, match="其他任务"):
        store.acquire(
            "4",
            job_id="job-b",
            worker_token="token-b",
            family="glm",
        )


def test_stale_foreign_lease_not_stolen(tmp_path: Path, monkeypatch):
    store = GpuLeaseStore(tmp_path / "leases")
    monkeypatch.setattr(
        "audio_engine.core.stage1.gpu_lease.pid_is_alive", lambda pid: False
    )
    store.acquire(
        "4",
        job_id="job-a",
        worker_token="token-a",
        family="qwen",
    )
    with pytest.raises(GpuLeaseError, match="不能抢占"):
        store.acquire(
            "4",
            job_id="job-b",
            worker_token="token-b",
            family="glm",
            steal_stale_same_job=True,
        )


def test_old_token_cannot_release_new_lease(tmp_path: Path, monkeypatch):
    store = GpuLeaseStore(tmp_path / "leases")
    monkeypatch.setattr(
        "audio_engine.core.stage1.gpu_lease.pid_is_alive", lambda pid: False
    )
    first = store.acquire(
        "5", job_id="job-a", worker_token="old", family="qwen"
    )
    # Same job resume with new token after stale.
    monkeypatch.setattr(
        "audio_engine.core.stage1.gpu_lease.pid_is_alive",
        lambda pid: False,
    )
    second = store.acquire(
        "5", job_id="job-a", worker_token="new", family="qwen"
    )
    assert second.worker_token == "new"
    store.release("5", worker_token="old", job_id="job-a")
    assert store.read("5") is not None
    store.release("5", worker_token="new", job_id="job-a")
    assert store.read("5") is None
    assert first.gpu_key == "5"


def test_scheduler_claims_two_gpus_in_parallel(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "audio_engine.core.stage1.gpu_lease.pid_is_alive", lambda pid: True
    )
    devices = {"4": _device("4"), "5": _device("5")}

    def snap(ids: list[str]) -> list[GpuDevice]:
        return [devices[str(i)] for i in ids if str(i) in devices]

    binding = bind_authorized_gpus(
        request_gpus=["4", "5"],
        authorized_gpus=(4, 5),
        authorized_uuids=("GPU-uuid-4", "GPU-uuid-5"),
        snapshot=snap(["4", "5"]),
    )
    sched = DualGpuScheduler(
        job_id="job-x",
        worker_token="tok",
        gpus=["4", "5"],
        binding=binding,
        lease_store=GpuLeaseStore(tmp_path / "leases"),
        snapshot_fn=snap,
        allow_unknown_inventory=False,
    )
    sched.enqueue_default_families()
    d1 = sched.try_claim()
    d2 = sched.try_claim()
    assert d1 is not None and d2 is not None
    assert {d1.gpu_key, d2.gpu_key} == {"4", "5"}
    assert {d1.family, d2.family} == {"qwen", "glm"}
    # Third claim waits for free card (sensevoice tail).
    assert sched.try_claim() is None
    sched.release_gpu(d1.gpu_key)
    d3 = sched.try_claim()
    assert d3 is not None
    assert d3.family == "sensevoice"
    assert d3.claim_wait_s >= 0.0


def test_single_gpu_fault_other_continues(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "audio_engine.core.stage1.gpu_lease.pid_is_alive", lambda pid: True
    )
    # GPU 4 blocked by foreign process; GPU 5 free.
    devices = {
        "4": _device(
            "4",
            procs=[GpuProcess(pid=9, used_memory_mib=1000, name="foreign")],
        ),
        "5": _device("5"),
    }

    def snap(ids: list[str]) -> list[GpuDevice]:
        return [devices[str(i)] for i in ids if str(i) in devices]

    binding = bind_authorized_gpus(
        request_gpus=["4", "5"],
        authorized_gpus=(4, 5),
        authorized_uuids=("GPU-uuid-4", "GPU-uuid-5"),
        snapshot=snap(["4", "5"]),
    )
    sched = DualGpuScheduler(
        job_id="job-y",
        worker_token="tok",
        gpus=["4", "5"],
        binding=binding,
        lease_store=GpuLeaseStore(tmp_path / "leases"),
        snapshot_fn=snap,
        allow_unknown_inventory=False,
    )
    sched.enqueue_default_families(remaining={"qwen", "glm"})
    claimed = []
    for _ in range(5):
        d = sched.try_claim()
        if d:
            claimed.append(d)
            # Keep lease held to simulate running.
    assert len(claimed) == 1
    assert claimed[0].gpu_key == "5"


def test_benchmark_plans_are_documented_not_fabricated():
    serial = build_serial_baseline_plan(["qwen", "glm", "sensevoice"], "4")
    dual = build_dual_gpu_plan(["qwen", "glm", "sensevoice"], ["4", "5"])
    assert len(serial) == 3
    assert dual[0]["mode"] == "dual_parallel"
    assert any(item.get("mode") == "tail_claim" for item in dual)
    # No invented timing numbers in the plan.
    assert all("elapsed_s" not in item for item in serial + dual)


def test_gaps_declare_step4_code_ready_benchmark_pending():
    assert IMPLEMENTATION_GAPS["step4_dual_gpu_scheduler"] is True
    assert IMPLEMENTATION_GAPS["step4_gpu_lease"] is True
    assert IMPLEMENTATION_GAPS["server_dual_gpu_benchmark"] is False
    assert IMPLEMENTATION_GAPS["server_e2e_accepted"] is False


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
        "authorized_gpu_uuids": ["GPU-uuid-4", "GPU-uuid-5"],
        "engine_python": str(engine),
        "dnsmos": {"onnx_path": str(dnsmos)},
        "scheduler": {
            "dual_gpu": True,
            "lease_root": str(tmp_path / "gpu_leases"),
            "poll_interval_s": 0.01,
        },
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
                "env": {"PYTHONNOUSERSITE": "1"},
                "unset_env": [],
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


def test_sensevoice_constrained_to_leased_gpu(tmp_path: Path):
    runtime = _filled_runtime(tmp_path)
    source = tmp_path / "wavs"
    source.mkdir()
    req = Stage1JobRequest(
        batch="sched-batch",
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
    orch = Stage1Orchestrator(root, dry_run=True, sleep_fn=lambda _s: None)
    orch._family_gpu["sensevoice"] = "5"
    orch._call_with_retries(
        "serve_start_sensevoice",
        lambda: orch._serve_start_body("sensevoice", gpu="5"),
    )
    session = orch.sessions["sensevoice"]
    assert session.client_env["CUDA_VISIBLE_DEVICES"] == "5"
    pipe = root / "pipelines" / "sensevoice_asr_batch.yaml"
    assert pipe.is_file()
    raw = yaml.safe_load(pipe.read_text(encoding="utf-8"))
    assert raw["sharding"]["gpus"] == [5]


def test_family_vram_budget_respects_util():
    qwen = family_vram_budget_mib("qwen", total_mib=80000, gpu_memory_utilization=0.5)
    glm = family_vram_budget_mib("glm", total_mib=80000, gpu_memory_utilization=0.9)
    assert glm > qwen
