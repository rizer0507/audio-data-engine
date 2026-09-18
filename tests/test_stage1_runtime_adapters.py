"""Unit tests for stage-1 runtime config and family launch adapters."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

from audio_engine.core.stage1.adapters import get_adapter
from audio_engine.core.stage1.adapters.base import dump_plan
from audio_engine.core.stage1.identity import assert_served_model, models_url
from audio_engine.core.stage1.process import (
    ServiceSession,
    build_cuda_env,
    terminate_owned_process,
)
from audio_engine.core.stage1.runtime_config import MissingConfigError, load_runtime_config


def _write_runtime(tmp_path: Path, **overrides) -> Path:
    root = tmp_path
    qwen_vllm = root / "bin" / "vllm"
    qwen_vllm.parent.mkdir(parents=True)
    qwen_vllm.write_text("#!/bin/sh\n", encoding="utf-8")
    glm_root = root / "glm_env"
    (glm_root / "bin").mkdir(parents=True)
    (glm_root / "bin" / "python").write_text("#!/bin/sh\n", encoding="utf-8")
    (glm_root / "bin" / "vllm").write_text("#!/bin/sh\n", encoding="utf-8")
    model_qwen = root / "models" / "qwen"
    model_qwen.mkdir(parents=True)
    template = root / "qwen3_asr_language.jinja"
    template.write_text("{{ messages }}", encoding="utf-8")
    model_glm = root / "models" / "glm"
    model_glm.mkdir(parents=True)
    model_sv = root / "models" / "sensevoice"
    model_sv.mkdir(parents=True)
    engine = root / "engine" / "python"
    engine.parent.mkdir(parents=True)
    engine.write_text("#!/bin/sh\n", encoding="utf-8")
    dnsmos = root / "dnsmos.onnx"
    dnsmos.write_bytes(b"onnx")

    data = {
        "authorized_gpus": [4, 5],
        "engine_python": str(engine),
        "dnsmos": {"onnx_path": str(dnsmos)},
        "probe": {"audio_path": None, "ready_timeout_s": 5, "ready_poll_s": 0.1},
        "session": {"root": str(root / "sessions")},
        "families": {
            "qwen": {
                "enabled": True,
                "model_path": str(model_qwen),
                "chat_template": str(template),
                "served_model_name": "qwen3-asr",
                "vllm_bin": str(qwen_vllm),
                "host": "127.0.0.1",
                "port": 5555,
                "gpu_memory_utilization": 0.5,
                "tensor_parallel_size": 1,
                "api_key": "dummy",
            },
            "glm": {
                "enabled": True,
                "model_path": str(model_glm),
                "env_root": str(glm_root),
                "served_model_name": "glm-asr",
                "host": "0.0.0.0",
                "client_host": "127.0.0.1",
                "port": 5570,
                "tensor_parallel_size": 1,
                "dtype": "bfloat16",
                "max_model_len": 4096,
                "max_num_seqs": 8,
                "gpu_memory_utilization": 0.90,
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
            },
            "sensevoice": {
                "enabled": True,
                "model_path": str(model_sv),
                "device": "cuda:0",
                "language": "auto",
                "use_itn": True,
                "disable_update": True,
            },
        },
    }
    for key, value in overrides.items():
        if key == "families":
            data["families"].update(value)
        else:
            data[key] = value
    path = root / "server.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def test_repo_server_yaml_reports_required_gaps():
    path = Path("configs/stage1/server.yaml")
    cfg = load_runtime_config(path)
    missing = cfg.missing_fields(deploy=True)
    assert any("authorized_gpus" in item for item in missing)
    assert any("engine_python" in item for item in missing)
    assert any("qwen.vllm_bin" in item for item in missing)
    assert any("dnsmos.onnx_path" in item for item in missing)
    # 已知基线路径应已填入，不应因空 model_path 报缺
    assert not any(item.endswith("families.qwen.model_path") for item in missing)
    assert not any("families.glm.env_root" in item for item in missing)


def test_check_config_rejects_blank_authorized_gpus(tmp_path: Path):
    path = _write_runtime(tmp_path, authorized_gpus=None)
    cfg = load_runtime_config(path)
    with pytest.raises(MissingConfigError) as exc:
        cfg.require_ready(families=["qwen"])
    assert any("authorized_gpus" in item for item in exc.value.missing)


def test_reject_unauthorized_gpu(tmp_path: Path):
    cfg = load_runtime_config(_write_runtime(tmp_path))
    with pytest.raises(ValueError, match="不在 authorized_gpus"):
        get_adapter("qwen").plan(cfg, gpu=0)


def test_qwen_plan_preserves_chat_template_and_baseline_flags(tmp_path: Path):
    cfg = load_runtime_config(_write_runtime(tmp_path))
    plan = get_adapter("qwen").plan(cfg, gpu=4, port=5555)
    assert plan.env["CUDA_VISIBLE_DEVICES"] == "4"
    assert "--chat-template" in plan.argv
    assert str(cfg.qwen.chat_template) in plan.argv
    assert plan.argv[plan.argv.index("--served-model-name") + 1] == "qwen3-asr"
    assert plan.argv[plan.argv.index("--gpu-memory-utilization") + 1] == "0.50"
    assert plan.client_env["QWEN_ASR_API_BASE"] == "http://127.0.0.1:5555"
    assert plan.client_env["QWEN_ASR_API_KEY"] == "dummy"
    dumped = dump_plan(plan)
    assert "chat-template" in dumped["argv_shell"]


def test_glm_plan_preserves_independent_env_and_flashinfer(tmp_path: Path):
    cfg = load_runtime_config(_write_runtime(tmp_path))
    plan = get_adapter("glm").plan(cfg, gpu=5, port=5570)
    assert plan.argv[0].endswith("python")
    assert plan.argv[1].endswith("vllm")
    assert plan.argv[2] == "serve"
    assert "--chat-template" not in plan.argv
    assert "--no-enable-flashinfer-autotune" in plan.argv
    assert plan.argv[plan.argv.index("--kernel-config") + 1] == (
        '{"enable_jit_warmup":false,"enable_cutedsl_warmup":false}'
    )
    assert plan.argv[plan.argv.index("--limit-mm-per-prompt") + 1] == '{"audio":1}'
    assert plan.argv[plan.argv.index("--gpu-memory-utilization") + 1] == "0.90"
    assert plan.env["VLLM_USE_FLASHINFER_SAMPLER"] == "0"
    assert plan.env["PYTHONNOUSERSITE"] == "1"
    assert plan.client_env["GLM_ASR_API_BASE"] == "http://127.0.0.1:5570"
    assert plan.client_env["GLM_ASR_MODEL"] == "glm-asr"


def test_sensevoice_plan_is_local_and_sets_cuda(tmp_path: Path):
    cfg = load_runtime_config(_write_runtime(tmp_path))
    plan = get_adapter("sensevoice").plan(cfg, gpu=4)
    assert plan.kind == "local"
    assert plan.port is None
    assert plan.api_base is None
    assert plan.client_env["SENSEVOICE_MODEL_PATH"] == str(cfg.sensevoice.model_path)
    assert plan.client_env["CUDA_VISIBLE_DEVICES"] == "4"


def test_build_cuda_env_unsets_glm_typos(monkeypatch):
    monkeypatch.setenv("VLLM_ATTENTION_BACKEND", "X")
    monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLE", "1")
    monkeypatch.setenv("VLLM_ATTENTIOIN_BACKEND", "typo")
    env = build_cuda_env(
        4,
        overlays={"VLLM_USE_FLASHINFER_SAMPLER": "0"},
        unset=(
            "VLLM_ATTENTION_BACKEND",
            "VLLM_USE_FLASHINFER_SAMPLE",
            "VLLM_ATTENTIOIN_BACKEND",
        ),
    )
    assert env["CUDA_VISIBLE_DEVICES"] == "4"
    assert env["VLLM_USE_FLASHINFER_SAMPLER"] == "0"
    assert "VLLM_ATTENTION_BACKEND" not in env
    assert "VLLM_USE_FLASHINFER_SAMPLE" not in env
    assert "VLLM_ATTENTIOIN_BACKEND" not in env


def test_models_url_and_identity(monkeypatch):
    assert models_url("http://127.0.0.1:5555") == "http://127.0.0.1:5555/v1/models"

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps({"data": [{"id": "qwen3-asr"}]}).encode()

    monkeypatch.setattr(
        "audio_engine.core.stage1.identity.urllib.request.urlopen",
        lambda *args, **kwargs: Response(),
    )
    identity = assert_served_model("http://127.0.0.1:5555", "qwen3-asr")
    assert identity.model_ids == ("qwen3-asr",)


def test_identity_mismatch(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps({"data": [{"id": "other"}]}).encode()

    monkeypatch.setattr(
        "audio_engine.core.stage1.identity.urllib.request.urlopen",
        lambda *args, **kwargs: Response(),
    )
    with pytest.raises(RuntimeError, match="模型身份不匹配"):
        assert_served_model("http://127.0.0.1:5555", "qwen3-asr")


def test_stop_skips_unowned_session(tmp_path: Path, monkeypatch):
    cfg = load_runtime_config(_write_runtime(tmp_path))
    session = ServiceSession(
        family="qwen",
        gpu=4,
        port=5555,
        api_base="http://127.0.0.1:5555",
        served_model_name="qwen3-asr",
        owned=False,
        client_env={},
        pid=999999,
        attached=True,
    )
    called = {"n": 0}

    def boom(**kwargs):
        called["n"] += 1
        raise AssertionError("must not terminate unowned")

    monkeypatch.setattr(
        "audio_engine.core.stage1.adapters.qwen.terminate_owned_process",
        boom,
    )
    get_adapter("qwen").stop(session)
    assert called["n"] == 0


def test_dry_run_start_writes_session(tmp_path: Path):
    cfg = load_runtime_config(_write_runtime(tmp_path))
    session_dir = tmp_path / "sess-qwen"
    session = get_adapter("qwen").start(
        cfg,
        gpu=4,
        session_dir=session_dir,
        dry_run=True,
    )
    assert session.owned is False
    assert (session_dir / "session.json").is_file()
    loaded = ServiceSession.load(session_dir / "session.json")
    assert loaded.argv[0].endswith("vllm")
    assert "--chat-template" in loaded.argv


def test_glm_dry_run_and_sensevoice_start(tmp_path: Path):
    cfg = load_runtime_config(_write_runtime(tmp_path))
    glm_dir = tmp_path / "sess-glm"
    glm = get_adapter("glm").start(cfg, gpu=5, session_dir=glm_dir, dry_run=True)
    assert glm.client_env["GLM_ASR_MODEL"] == "glm-asr"
    assert "--no-enable-flashinfer-autotune" in glm.argv

    sv_dir = tmp_path / "sess-sv"
    sv = get_adapter("sensevoice").start(cfg, gpu=4, session_dir=sv_dir, dry_run=True)
    assert sv.kind == "local"
    assert sv.client_env["CUDA_VISIBLE_DEVICES"] == "4"


def test_terminate_owned_missing_pid_is_noop():
    terminate_owned_process(pid=None)
    terminate_owned_process(pid=-1)


def test_qwen_probe_uses_vllm_helper(tmp_path: Path, monkeypatch):
    cfg = load_runtime_config(_write_runtime(tmp_path))
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")

    monkeypatch.setattr(
        "audio_engine.core.stage1.adapters.qwen.assert_served_model",
        lambda *args, **kwargs: None,
    )

    def fake_probe(**kwargs):
        from audio_engine.core.stage1.probe import ProbeResult

        return ProbeResult(
            ok=True,
            family="qwen",
            audio_path=str(audio),
            text="你好",
            detail={},
        )

    monkeypatch.setattr(
        "audio_engine.core.stage1.adapters.qwen.probe_vllm_transcription",
        fake_probe,
    )
    result = get_adapter("qwen").probe(
        cfg,
        api_base="http://127.0.0.1:5555",
        audio_path=audio,
    )
    assert result.ok and result.text == "你好"


def test_sensevoice_probe_constrained_and_released(tmp_path: Path, monkeypatch):
    cfg = load_runtime_config(_write_runtime(tmp_path))
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")
    released = {"n": 0}

    class FakeModel:
        def generate(self, **kwargs):
            assert os.environ.get("CUDA_VISIBLE_DEVICES") == "5"
            return [{"text": "<|zh|><|NEUTRAL|><|Speech|>探测文本"}]

    monkeypatch.setattr(
        "audio_engine.operators.asr.sensevoice._load_sensevoice_model",
        lambda settings: FakeModel(),
    )

    def fake_release():
        released["n"] += 1
        return 1

    monkeypatch.setattr(
        "audio_engine.operators.asr.sensevoice.release_cached_models",
        fake_release,
    )
    result = get_adapter("sensevoice").probe(cfg, audio_path=audio, gpu=5)
    assert result.ok
    assert "探测文本" in result.text
    assert released["n"] >= 1
