"""Regressions for isolated, bounded DNSMOS sessions and model hash reuse."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from audio_engine.core.quality import dnsmos_p835 as core
from audio_engine.operators.quality.dnsmos import DnsmosOperator


def test_onnx_session_has_explicit_thread_limits_and_no_spin(tmp_path, monkeypatch):
    calls = []

    class Options:
        def __init__(self):
            self.entries = {}

        def add_session_config_entry(self, key, value):
            self.entries[key] = value

    def create(path, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(get_inputs=lambda: [SimpleNamespace(name="audio")])

    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(
        SessionOptions=Options, InferenceSession=create,
        GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL="all"),
        ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL="sequential"),
    ))
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    core.DnsmosP835Session(model)
    options = calls[0]["sess_options"]
    assert options.intra_op_num_threads == options.inter_op_num_threads == 1
    assert options.execution_mode == "sequential"
    assert options.entries == {"session.intra_op.allow_spinning": "0",
                               "session.inter_op.allow_spinning": "0"}


@pytest.mark.parametrize("config", [
    {"intra_op_num_threads": 0}, {"intra_op_num_threads": -1},
    {"inter_op_num_threads": True}, {"intra_op_num_threads": 1.5},
    {"allow_spinning": "false"}, {"unknown": 1}, [],
])
def test_invalid_runtime_settings_are_rejected(config):
    with pytest.raises(ValueError):
        core.DnsmosRuntimeOptions.from_config(config)


def test_session_cache_distinguishes_runtime_settings(tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    operator = DnsmosOperator()
    cfg = {"model_path": str(model)}
    original = operator._session_cache_key(cfg)
    assert original == operator._session_cache_key({**cfg, "runtime": {
        "intra_op_num_threads": 1, "inter_op_num_threads": 1, "allow_spinning": False,
    }})
    assert original != operator._session_cache_key({**cfg, "runtime": {"intra_op_num_threads": 2}})
    assert original != operator._session_cache_key({**cfg, "runtime": {"allow_spinning": True}})


def test_model_digest_is_reused_and_invalidated_on_edit(tmp_path, monkeypatch):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"old-model")
    original_digest = core.file_digest
    calls = []

    def digest(path):
        calls.append(path)
        return original_digest(path)

    monkeypatch.setattr(core, "file_digest", digest)
    first = core.model_file_digest(model)
    for _ in range(20):
        assert core.model_file_digest(model) == first
    assert len(calls) == 1
    model.write_bytes(b"replacement-model")
    assert core.model_file_digest(model) != first
    assert len(calls) == 2


def test_operator_forwards_thread_budget(tmp_path, monkeypatch):
    from audio_engine.operators.quality import dnsmos

    calls = []
    monkeypatch.setattr(dnsmos, "DnsmosP835Session", lambda *args, **kwargs: calls.append(kwargs))
    DnsmosOperator()._create_session({"model_path": str(tmp_path / "model.onnx"),
                                    "runtime": {"intra_op_num_threads": 2}})
    assert calls[0]["runtime_options"].intra_op_num_threads == 2
    assert calls[0]["runtime_options"].allow_spinning is False


def test_thread_local_sessions_stay_separate(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    operator = DnsmosOperator()
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    cfg = {"model_path": str(model), "session_mode": "thread_local"}
    barrier = Barrier(2)
    monkeypatch.setattr(operator, "_create_session", lambda cfg: object())

    def get():
        session = operator._get_session(cfg)
        barrier.wait(timeout=5)
        assert operator._get_session(cfg) is session
        return session

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(get) for _ in range(2)]
        assert futures[0].result() is not futures[1].result()
