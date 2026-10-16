"""Regressions for missing local models, startup failures and bounded VAD work."""

from __future__ import annotations

import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from audio_engine.core.fsmn_vad.adapter import load_funasr_vad
from audio_engine.core.fsmn_vad.runtime import validate_runtime_model_dir
from audio_engine.core.operator import OperatorConfig
from audio_engine.core.pipeline import PipelineConfig
from audio_engine.core.sample import Sample
from audio_engine.core.sharded_run import run_sharded_pipeline
from audio_engine.operators.audio import fsmn_vad_segment as vad_module


def _runtime(tmp_path: Path) -> Path:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    for name in ("config.yaml", "configuration.json", "model.pt", "am.mvn"):
        (runtime / name).write_text("fixture", encoding="utf-8")
    return runtime


@pytest.mark.parametrize(
    "missing", [None, "config.yaml", "configuration.json", "model.pt", "am.mvn"],
)
def test_invalid_runtime_never_calls_funasr(tmp_path, monkeypatch, missing):
    def forbidden(**kwargs):
        pytest.fail("An invalid local directory must never reach the model hub")

    monkeypatch.setitem(sys.modules, "funasr", SimpleNamespace(AutoModel=forbidden))
    runtime = tmp_path / "path" / "to" / "fsmn-vad-runtime"
    if missing:
        runtime = _runtime(tmp_path)
        (runtime / missing).unlink()
    with pytest.raises(FileNotFoundError, match="FSMN VAD runtime"):
        load_funasr_vad(runtime)


def test_valid_runtime_loads_local_model(tmp_path, monkeypatch):
    runtime = _runtime(tmp_path)
    calls = []
    monkeypatch.setitem(
        sys.modules, "funasr", SimpleNamespace(AutoModel=lambda **kwargs: calls.append(kwargs))
    )
    load_funasr_vad(runtime)
    assert calls[0]["model"] == str(runtime.resolve())
    assert calls[0]["disable_update"] is True
    assert calls[0]["ncpu"] == 1
    assert validate_runtime_model_dir(runtime) == runtime.resolve()


def test_funasr_thread_budget_can_be_overridden(tmp_path, monkeypatch):
    runtime = _runtime(tmp_path)
    calls = []
    monkeypatch.setitem(
        sys.modules, "funasr", SimpleNamespace(AutoModel=lambda **kwargs: calls.append(kwargs))
    )
    load_funasr_vad(runtime, ncpu=2)
    assert calls[0]["ncpu"] == 2


@pytest.mark.parametrize("ncpu", [0, -1, True, 1.5, "1"])
def test_invalid_thread_budget_never_loads_model(tmp_path, monkeypatch, ncpu):
    monkeypatch.setitem(sys.modules, "funasr", SimpleNamespace(
        AutoModel=lambda **kwargs: pytest.fail("invalid budget must not load a model")
    ))
    with pytest.raises(ValueError, match="ncpu"):
        load_funasr_vad(tmp_path, ncpu=ncpu)


def test_operator_passes_thread_budget_to_model(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(vad_module, "load_funasr_vad", lambda *a, **k: calls.append((a, k)))
    vad_module.FsmnVadSegmentOperator()._build_vad(
        {"model_runtime_dir": str(tmp_path), "ncpu": 2}, mock=False, fingerprint={},
    )
    assert calls == [((str(tmp_path),), {"ncpu": 2})]


def test_operator_rejects_invalid_runtime_before_dependency_imports(tmp_path, monkeypatch):
    monkeypatch.delenv("AUDIO_ENGINE_VAD_MOCK", raising=False)
    monkeypatch.setattr(
        vad_module, "dependency_versions", lambda: pytest.fail("preflight must run first")
    )
    config = OperatorConfig(
        params={"model_runtime_dir": str(tmp_path / "missing"), "source_workers": 2},
        output_dir=tmp_path / "out", run_dir=tmp_path / "run",
    )
    with pytest.raises(FileNotFoundError, match="FSMN VAD runtime"):
        vad_module.FsmnVadSegmentOperator().run([Sample(id="a", source_path="a.wav")], config)
    assert not (tmp_path / "run").exists()


def test_model_initialization_failure_does_not_retry_sources(tmp_path, monkeypatch):
    operator = vad_module.FsmnVadSegmentOperator()
    calls = []

    def build(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("model initialization failed")
        return object()

    monkeypatch.setattr(operator, "_build_vad", build)
    monkeypatch.setattr(operator, "_build_fingerprint", lambda *a, **k: {"mock": True})
    monkeypatch.setattr(
        operator, "_process_one_source", lambda *a, **k: pytest.fail("no sources before ready")
    )
    config = OperatorConfig(
        params={"mock": True, "source_workers": 2},
        output_dir=tmp_path / "out", run_dir=tmp_path / "run",
    )
    samples = [Sample(id=str(i), source_path="unused.wav") for i in range(100)]
    with pytest.raises(RuntimeError, match="model initialization failed"):
        operator.run(samples, config)
    assert len(calls) == 2


def test_pool_bounds_submissions_and_reuses_isolated_models(tmp_path, monkeypatch):
    operator = vad_module.FsmnVadSegmentOperator()
    models = []
    seen = {}
    lock = threading.Lock()
    barrier = threading.Barrier(2)
    submissions_since_wait = 0
    wait_calls = 0

    def build(*args, **kwargs):
        model = object()
        models.append(model)
        return model

    def process(sample, *, vad, **kwargs):
        tid = threading.get_ident()
        with lock:
            first = tid not in seen
            seen.setdefault(tid, []).append(vad)
        if first:
            barrier.wait(timeout=5)
        return []

    class TrackingExecutor(ThreadPoolExecutor):
        def submit(self, *args, **kwargs):
            nonlocal submissions_since_wait
            submissions_since_wait += 1
            return super().submit(*args, **kwargs)

    real_wait = vad_module.wait

    def bounded_wait(futures, **kwargs):
        nonlocal submissions_since_wait, wait_calls
        assert submissions_since_wait <= 4
        assert len(futures) <= 4
        submissions_since_wait = 0
        wait_calls += 1
        return real_wait(futures, **kwargs)

    monkeypatch.setattr(operator, "_build_vad", build)
    monkeypatch.setattr(operator, "_build_fingerprint", lambda *a, **k: {"mock": True})
    monkeypatch.setattr(operator, "_process_one_source", process)
    monkeypatch.setattr(vad_module, "ThreadPoolExecutor", TrackingExecutor)
    monkeypatch.setattr(vad_module, "wait", bounded_wait)
    config = OperatorConfig(
        params={"mock": True, "source_workers": 2, "max_in_flight_sources": 4},
        output_dir=tmp_path / "out", run_dir=tmp_path / "run",
    )
    samples = [Sample(id=str(i), source_path="unused.wav") for i in range(100)]
    assert operator.run(samples, config) == []
    assert len(models) == len(seen) == 2
    assert sum(map(len, seen.values())) == len(samples)
    assert all(all(model is used[0] for model in used) for used in seen.values())
    assert len({id(used[0]) for used in seen.values()}) == 2
    assert wait_calls > 1


def test_parent_preflight_rejects_bad_model_before_splitting(tmp_path, monkeypatch):
    import audio_engine.core.sharded_run as sharded

    monkeypatch.delenv("AUDIO_ENGINE_VAD_MOCK", raising=False)
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(
        "name: bad_vad\ninput:\n  manifest: input.parquet\noutput:\n  manifest: out.parquet\n"
        "sharding:\n  shards: 8\n  parallel_shards: 8\n"
        "pipeline:\n  - name: vad\n    operator: audio.fsmn_vad_segment\n"
        f"    params:\n      model_runtime_dir: {tmp_path.as_posix()}/missing\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sharded, "prepare_full_manifest", lambda *a, **k: pytest.fail("no split before preflight")
    )
    monkeypatch.setattr(
        sharded, "spawn_shard_processes",
        lambda *a, **k: pytest.fail("no children before preflight"),
    )
    config = PipelineConfig.from_yaml(config_path)
    with pytest.raises(FileNotFoundError, match="FSMN VAD runtime"):
        run_sharded_pipeline(config, run_root=tmp_path / "run")
    assert not (tmp_path / "run").exists()
