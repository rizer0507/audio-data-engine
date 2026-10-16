"""Regressions for isolated, bounded DNSMOS sessions and model hash reuse."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from audio_engine.core.quality import dnsmos_p835 as core
from audio_engine.operators.quality.dnsmos import DnsmosOperator


@pytest.mark.parametrize("batch_dim,expected_batch", [(1, 1), ("N", 4)])
def test_onnx_session_has_explicit_thread_limits_and_no_spin(
    tmp_path, monkeypatch, batch_dim, expected_batch,
):
    calls = []

    class Options:
        def __init__(self):
            self.entries = {}

        def add_session_config_entry(self, key, value):
            self.entries[key] = value

    def create(path, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(get_inputs=lambda: [SimpleNamespace(name="audio", shape=[batch_dim, 144160])])

    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(
        SessionOptions=Options, InferenceSession=create,
        GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL="all"),
        ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL="sequential"),
    ))
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    session = core.DnsmosP835Session(model)
    assert session.window_batch_size == expected_batch
    options = calls[0]["sess_options"]
    assert options.intra_op_num_threads == 2
    assert options.inter_op_num_threads == 1
    assert options.execution_mode == "sequential"
    assert options.entries == {"session.intra_op.allow_spinning": "0",
                               "session.inter_op.allow_spinning": "0"}


@pytest.mark.parametrize("config", [
    {"intra_op_num_threads": 0}, {"intra_op_num_threads": -1},
    {"inter_op_num_threads": True}, {"intra_op_num_threads": 1.5},
    {"allow_spinning": "false"}, {"window_batch_size": 0}, {"unknown": 1}, [],
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
        "intra_op_num_threads": 2, "inter_op_num_threads": 1, "allow_spinning": False,
    }})
    assert original != operator._session_cache_key({**cfg, "runtime": {"intra_op_num_threads": 1}})
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


def test_window_batching_preserves_every_window_and_calibration():
    import numpy as np

    class Session:
        def __init__(self):
            self.inputs = []

        def run(self, _, feed):
            values = next(iter(feed.values()))
            self.inputs.append(values.copy())
            index = values[:, 0] / 16000
            return [np.stack((2 + index * 0.1, 3 + index * 0.05, 2.5 + index * 0.08), axis=1)]

    def scorer(batch_size):
        instance = object.__new__(core.DnsmosP835Session)
        instance._session = Session()
        instance.input_name = "audio"
        instance.model_digest = "test"
        instance.window_batch_size = batch_size
        return instance

    audio = np.arange(20 * 16000, dtype=np.float32)
    reference, batched = scorer(1), scorer(4)
    old, new = reference.score_array(audio, 16000), batched.score_array(audio, 16000)
    assert old.status == new.status == "success"
    assert [x.shape[0] for x in batched._session.inputs] == [4, 4, 3]
    np.testing.assert_array_equal(np.concatenate(reference._session.inputs),
                                  np.concatenate(batched._session.inputs))
    assert (new.sig, new.bak, new.ovrl) == pytest.approx((old.sig, old.bak, old.ovrl), abs=1e-10)


def test_evidence_backend_uses_and_invalidates_runtime_configuration(tmp_path, monkeypatch):
    import json

    import numpy as np
    import soundfile as sf

    from audio_engine.core.evidence.scoring import _DnsmosBackend

    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    config_path = tmp_path / "config.yaml"
    cfg = {"model_path": str(model), "runtime": {"intra_op_num_threads": 2}}
    config_path.write_text(json.dumps(cfg), encoding="utf-8")
    calls = []

    class Session:
        def __init__(self, model, **kwargs):
            calls.append(kwargs["runtime_options"])

        def score_array(self, audio, sr):
            return core.DnsmosScores(sig=3, bak=3, ovrl=3, status="success")

    monkeypatch.setattr(core, "DnsmosP835Session", Session)
    monkeypatch.setattr(sf, "read", lambda *args, **kwargs: (np.zeros(16000), 16000))
    backend = _DnsmosBackend()
    assert backend("unused.wav", str(config_path))["dnsmos_status"] == "success"
    backend("unused.wav", str(config_path))
    assert len(calls) == 1
    cfg["runtime"]["intra_op_num_threads"] = 1
    config_path.write_text(json.dumps(cfg), encoding="utf-8")
    backend("unused.wav", str(config_path))
    assert [options.intra_op_num_threads for options in calls] == [2, 1]
