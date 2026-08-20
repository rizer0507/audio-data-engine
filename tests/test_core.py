from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import audio_engine.operators  # noqa: F401
from audio_engine.core.manifest import Manifest
from audio_engine.core.operator import OperatorConfig
from audio_engine.core.registry import OperatorRegistry
from audio_engine.core.sample import Sample


@pytest.fixture
def sample_wav(tmp_path: Path) -> Path:
    sr = 16000
    t = np.linspace(0, 1, sr, endpoint=False)
    data = 0.3 * np.sin(2 * np.pi * 440 * t)
    path = tmp_path / "test001.wav"
    sf.write(str(path), data.astype(np.float32), sr)
    return path


def test_ingest(sample_wav: Path):
    manifest = Manifest.ingest(sample_wav.parent)
    assert len(manifest) == 1
    assert manifest.samples[0].sample_rate == 16000
    assert manifest.samples[0].duration == pytest.approx(1.0, abs=0.01)


def test_manifest_roundtrip(sample_wav: Path, tmp_path: Path):
    manifest = Manifest.ingest(sample_wav.parent)
    out = tmp_path / "test.parquet"
    manifest.save(out)
    loaded = Manifest.load(out)
    assert len(loaded) == 1
    assert loaded.samples[0].id == manifest.samples[0].id


def test_resample_operator(sample_wav: Path):
    sample = Sample(
        id="test001",
        source_path=str(sample_wav),
        sha256="abc",
        audio={"raw": str(sample_wav)},
        sample_rate=16000,
        duration=1.0,
    )
    op = OperatorRegistry.get("audio.resample")
    config = OperatorConfig(
        params={"sample_rate": 8000, "input_audio_key": "raw", "output_audio_key": "resampled_8k"},
        output_dir=sample_wav.parent / "derived",
        cache_dir=sample_wav.parent / "cache",
    )
    result = op.process(sample, config)
    assert "resampled_8k" in result.sample.audio
    assert result.sample.labels.get("resampled") is True


def test_pcm_and_resample_passthrough(sample_wav: Path):
    """Already-wav / already-16k should not convert or rewrite."""
    sample = Sample(
        id="test001",
        source_path=str(sample_wav),
        sha256="abc",
        audio={"raw": str(sample_wav)},
        sample_rate=16000,
        duration=1.0,
    )
    pcm = OperatorRegistry.get("audio.pcm_to_wav")
    pcm_cfg = OperatorConfig(
        params={
            "sample_rate": 8000,
            "input_audio_key": "raw",
            "output_audio_key": "pcm_wav",
        },
        output_dir=sample_wav.parent / "derived",
        cache_dir=sample_wav.parent / "cache_pcm",
    )
    pcm_result = pcm.process(sample, pcm_cfg)
    assert pcm_result.sample.labels.get("pcm_converted") is False
    assert pcm_result.sample.audio["pcm_wav"] == str(sample_wav.resolve())

    rs = OperatorRegistry.get("audio.resample")
    rs_cfg = OperatorConfig(
        params={
            "sample_rate": 16000,
            "input_audio_key": "pcm_wav",
            "output_audio_key": "resampled_16k",
        },
        output_dir=sample_wav.parent / "derived",
        cache_dir=sample_wav.parent / "cache_rs",
    )
    rs_result = rs.process(pcm_result.sample, rs_cfg)
    assert rs_result.sample.labels.get("resampled") is False
    assert rs_result.sample.audio["resampled_16k"] == str(sample_wav.resolve())
    assert not (sample_wav.parent / "derived" / "resample_16k").exists()


def test_cache_hit(sample_wav: Path):
    sample = Sample(
        id="test001",
        source_path=str(sample_wav),
        sha256="deadbeef",
        audio={"raw": str(sample_wav)},
    )
    op = OperatorRegistry.get("quality.snr")
    config = OperatorConfig(
        params={"input_audio_key": "raw"},
        output_dir=sample_wav.parent / "derived",
        cache_dir=sample_wav.parent / "cache",
    )
    r1 = op.process(sample, config)
    r2 = op.process(sample, config)
    assert r1.cache_hit is False
    assert r2.cache_hit is True


def test_operator_registry():
    names = OperatorRegistry.list_operators()
    assert "audio.resample" in names
    assert "asr.qwen" in names
    assert "asr.sensevoice" in names
    assert "ingest.scan" in names


def test_ingest_pipeline(sample_wav: Path, tmp_path: Path):
    """Ingest runs through the unified PipelineRunner -> Operator path."""
    from audio_engine.core.pipeline import PipelineConfig, PipelineRunner, PipelineStep

    cfg = PipelineConfig(
        name="test_ingest",
        input_manifest="",
        source_dir=str(sample_wav.parent),
        steps=[PipelineStep(name="ingest", operator="ingest.scan")],
        output_dir=tmp_path / "derived",
        cache_dir=tmp_path / "cache",
        runs_dir=tmp_path / "runs",
    )
    runner = PipelineRunner(cfg)
    result = runner.run()

    assert len(result) == 1
    sample = result.samples[0]
    assert sample.sample_rate == 16000
    assert sample.is_completed("ingest.scan")
    assert sample.lineage[0].operator == "ingest.scan"
    assert runner.metrics.processed == 1


def test_filter_manifest(sample_wav: Path):
    manifest = Manifest.ingest(sample_wav.parent)
    manifest.samples[0].labels["badcase"] = "noise"
    filtered = manifest.filter("label_badcase == 'noise'")
    assert len(filtered) == 1

    empty = manifest.filter("label_badcase == 'silence'")
    assert len(empty) == 0


def test_probe_and_select(sample_wav: Path, tmp_path: Path):
    from audio_engine.core.pipeline import PipelineConfig, PipelineRunner, PipelineStep

    # corrupt wav should fail probe → broken → filtered out（输入已是 VAD 后数据，不测 speech_ratio）
    broken = tmp_path / "broken.wav"
    broken.write_bytes(b"not a real wav file")

    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    import shutil

    shutil.copy2(sample_wav, audio_dir / "good.wav")
    shutil.copy2(broken, audio_dir / "broken.wav")

    cfg = PipelineConfig(
        name="test_cleaning",
        input_manifest="",
        source_dir=str(audio_dir),
        steps=[
            PipelineStep(name="ingest", operator="ingest.scan"),
            PipelineStep(
                name="resample",
                operator="audio.resample",
                params={
                    "sample_rate": 16000,
                    "input_audio_key": "raw",
                    "output_audio_key": "resampled_16k",
                },
            ),
            PipelineStep(
                name="probe",
                operator="quality.probe",
                params={"input_audio_key": "resampled_16k"},
            ),
            PipelineStep(
                name="audio_pass",
                operator="quality.filter",
                params={
                    "expr": "label_broken != True and duration > 0",
                    "label_key": "audio_pass",
                },
            ),
            PipelineStep(
                name="select_pass",
                operator="quality.select",
                params={"expr": "label_audio_pass == True"},
            ),
        ],
        output_dir=tmp_path / "derived",
        cache_dir=tmp_path / "cache",
        runs_dir=tmp_path / "runs",
    )
    result = PipelineRunner(cfg).run()
    assert len(result) == 1
    assert result.samples[0].id == "good"
    assert result.samples[0].labels.get("audio_pass") is True


def test_resolve_source(tmp_path: Path):
    from audio_engine.core.source import resolve_source_input

    resources = tmp_path / "resources"
    resources.mkdir()
    manifest = resources / "manifest.yaml"
    manifest.write_text(
        "sources:\n"
        "  - source_id: source_A\n"
        "    source_name: A\n"
        "    ingested_at: '2026-08-20T10:00:00+08:00'\n"
        "    origin: test\n"
        "    path: D:/Data/batch_A\n",
        encoding="utf-8",
    )
    # no sample index → fall back to source_dir
    resolved = resolve_source_input("source_A", resources_manifest=manifest)
    assert resolved["source_dir"] == "D:/Data/batch_A"

    samples_dir = tmp_path / "resources" / "sources" / "source_A"
    samples_dir.mkdir(parents=True)
    (samples_dir / "samples.jsonl").write_text(
        '{"id":"x","source_path":"a.wav","sha256":"1"}\n',
        encoding="utf-8",
    )
    # monkeypatch cwd-relative paths used by resolver: create under real relative path
    # Prefer testing lookup with absolute by temporarily chdir
    import os

    old = os.getcwd()
    os.chdir(tmp_path)
    try:
        resolved2 = resolve_source_input("source_A", resources_manifest=manifest)
        assert resolved2["manifest"].endswith("samples.jsonl")
    finally:
        os.chdir(old)


def _write_wavs(directory: Path, count: int, sr: int = 8000) -> Path:
    """Write `count` distinct wavs that resample must actually convert (sr != 16k)."""
    directory.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        t = np.linspace(0, 0.2, int(sr * 0.2), endpoint=False)
        data = (0.2 + 0.01 * i) * np.sin(2 * np.pi * (200 + 20 * i) * t)
        sf.write(str(directory / f"s{i:03d}.wav"), data.astype(np.float32), sr)
    return directory


def _cleaning_steps():
    from audio_engine.core.pipeline import PipelineStep

    return [
        PipelineStep(name="ingest", operator="ingest.scan"),
        PipelineStep(
            name="resample",
            operator="audio.resample",
            params={
                "sample_rate": 16000,
                "input_audio_key": "raw",
                "output_audio_key": "resampled_16k",
            },
        ),
        PipelineStep(
            name="probe",
            operator="quality.probe",
            params={"input_audio_key": "resampled_16k"},
        ),
        PipelineStep(
            name="audio_pass",
            operator="quality.filter",
            params={"expr": "label_broken != True and duration > 0", "label_key": "audio_pass"},
        ),
        PipelineStep(
            name="select_pass",
            operator="quality.select",
            params={"expr": "label_audio_pass == True"},
        ),
    ]


def test_concurrent_matches_sequential(tmp_path: Path):
    """Thread pool output must equal the sequential output, order included."""
    from audio_engine.core.pipeline import ExecutionConfig, PipelineConfig, PipelineRunner

    audio_dir = _write_wavs(tmp_path / "audio", 12)
    derived = tmp_path / "derived"

    def run(execution: ExecutionConfig, tag: str):
        cfg = PipelineConfig(
            name=f"cleaning_{tag}",
            input_manifest="",
            source_dir=str(audio_dir),
            steps=_cleaning_steps(),
            output_dir=derived,
            cache_dir=tmp_path / f"cache_{tag}",
            runs_dir=tmp_path / "runs",
            execution=execution,
        )
        runner = PipelineRunner(cfg)
        return runner.run(), runner.metrics.to_dict()

    serial, serial_metrics = run(ExecutionConfig(executor="sequential", workers=1), "serial")
    parallel, parallel_metrics = run(ExecutionConfig(executor="thread", workers=4), "parallel")

    assert len(parallel) == 12
    assert [s.id for s in parallel] == [s.id for s in serial]
    assert [s.to_flat_dict() for s in parallel] == [s.to_flat_dict() for s in serial]
    assert parallel_metrics["by_step"] == serial_metrics["by_step"]
    assert parallel_metrics["failed"] == 0


def test_process_pool_matches_sequential(tmp_path: Path):
    """Process pool must match sequential output (picklable worker on Windows spawn)."""
    from audio_engine.core.pipeline import ExecutionConfig, PipelineConfig, PipelineRunner

    audio_dir = _write_wavs(tmp_path / "audio", 8)
    derived = tmp_path / "derived"

    def run(execution: ExecutionConfig, tag: str):
        cfg = PipelineConfig(
            name=f"cleaning_{tag}",
            input_manifest="",
            source_dir=str(audio_dir),
            steps=_cleaning_steps(),
            output_dir=derived,
            cache_dir=tmp_path / f"cache_{tag}",
            runs_dir=tmp_path / "runs",
            execution=execution,
        )
        return PipelineRunner(cfg).run()

    serial = run(ExecutionConfig(executor="sequential", workers=1, checkpoint_every=0), "serial")
    processed = run(ExecutionConfig(executor="process", workers=2, checkpoint_every=0), "process")

    assert len(processed) == 8
    assert [s.id for s in processed] == [s.id for s in serial]
    assert [s.to_flat_dict() for s in processed] == [s.to_flat_dict() for s in serial]


def test_process_pool_isolates_failures(tmp_path: Path):
    """A broken file must fail only its own sample under the process pool."""
    from audio_engine.core.pipeline import ExecutionConfig, PipelineConfig, PipelineRunner

    audio_dir = _write_wavs(tmp_path / "audio", 6)
    (audio_dir / "broken.wav").write_bytes(b"not a wav")

    cfg = PipelineConfig(
        name="cleaning_proc_failures",
        input_manifest="",
        source_dir=str(audio_dir),
        steps=_cleaning_steps(),
        output_dir=tmp_path / "derived",
        cache_dir=tmp_path / "cache",
        runs_dir=tmp_path / "runs",
        execution=ExecutionConfig(executor="process", workers=2, checkpoint_every=0),
    )
    runner = PipelineRunner(cfg)
    result = runner.run()
    metrics = runner.metrics.to_dict()

    assert len(result) == 6
    assert metrics["by_step"]["resample"]["failed"] == 1
    assert "broken" not in [s.id for s in result]


def test_concurrent_isolates_failures(tmp_path: Path):
    """A broken file must fail only its own sample, not the whole pool."""
    from audio_engine.core.pipeline import ExecutionConfig, PipelineConfig, PipelineRunner

    audio_dir = _write_wavs(tmp_path / "audio", 6)
    (audio_dir / "broken.wav").write_bytes(b"not a wav")

    cfg = PipelineConfig(
        name="cleaning_failures",
        input_manifest="",
        source_dir=str(audio_dir),
        steps=_cleaning_steps(),
        output_dir=tmp_path / "derived",
        cache_dir=tmp_path / "cache",
        runs_dir=tmp_path / "runs",
        execution=ExecutionConfig(executor="thread", workers=4),
    )
    runner = PipelineRunner(cfg)
    result = runner.run()
    metrics = runner.metrics.to_dict()

    assert len(result) == 6  # broken one dropped by select
    assert metrics["by_step"]["resample"]["failed"] == 1
    assert metrics["by_step"]["resample"]["processed"] == 6
    assert "broken" not in [s.id for s in result]


def test_derived_paths_are_unique_per_content(tmp_path: Path):
    """Same sample id + different content must not overwrite each other."""
    from audio_engine.core.manifest import file_sha256

    derived = tmp_path / "derived"
    outputs = []
    for idx, seconds in enumerate((0.2, 0.4)):
        src_dir = tmp_path / f"src{idx}"
        src_dir.mkdir()
        path = src_dir / "same_name.wav"
        sr = 8000
        t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
        sf.write(str(path), (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32), sr)

        sample = Sample(
            id="same_name",
            source_path=str(path),
            sha256=file_sha256(path),
            audio={"raw": str(path)},
            sample_rate=sr,
        )
        op = OperatorRegistry.get("audio.resample")
        config = OperatorConfig(
            params={
                "sample_rate": 16000,
                "input_audio_key": "raw",
                "output_audio_key": "resampled_16k",
            },
            output_dir=derived,
            cache_dir=tmp_path / "cache",
        )
        outputs.append(Path(op.process(sample, config).sample.audio["resampled_16k"]))

    assert outputs[0] != outputs[1]
    assert all(p.exists() for p in outputs)
    assert sf.info(str(outputs[0])).duration != sf.info(str(outputs[1])).duration


def test_atomic_writes_leave_no_temp_files(tmp_path: Path):
    from audio_engine.core.artifacts import atomic_path
    from audio_engine.core.pipeline import ExecutionConfig, PipelineConfig, PipelineRunner

    audio_dir = _write_wavs(tmp_path / "audio", 8)
    cfg = PipelineConfig(
        name="cleaning_atomic",
        input_manifest="",
        source_dir=str(audio_dir),
        steps=_cleaning_steps(),
        output_dir=tmp_path / "derived",
        cache_dir=tmp_path / "cache",
        runs_dir=tmp_path / "runs",
        execution=ExecutionConfig(executor="thread", workers=4),
    )
    PipelineRunner(cfg).run()

    assert not list((tmp_path / "cache").rglob("*.tmp*"))
    assert not list((tmp_path / "derived").rglob("*.tmp*"))
    for cache_file in (tmp_path / "cache").rglob("*.json"):
        json.loads(cache_file.read_text(encoding="utf-8"))

    # a failed write publishes nothing and cleans up its temp file
    target = tmp_path / "derived" / "aborted.wav"
    with pytest.raises(RuntimeError), atomic_path(target) as tmp:
        tmp.write_bytes(b"partial")
        raise RuntimeError("boom")
    assert not target.exists()
    assert not list(target.parent.glob("*.tmp*"))


def test_filter_is_batched_and_row_wise(tmp_path: Path):
    """Batch filter must label each sample exactly like the old per-sample query."""
    from audio_engine.core.pipeline import PipelineConfig, PipelineRunner, PipelineStep

    samples = [
        Sample(id="keep", source_path="a.wav", sha256="a", duration=1.0, labels={"broken": False}),
        Sample(id="zero", source_path="b.wav", sha256="b", duration=0.0, labels={"broken": False}),
        Sample(id="bad", source_path="c.wav", sha256="c", duration=1.0, labels={"broken": True}),
    ]
    op = OperatorRegistry.get("quality.filter")
    config = OperatorConfig(
        params={
            "expr": "label_broken != True and duration > 0",
            "label_key": "audio_pass",
            "chunk_size": 2,  # force multiple chunks
        },
        output_dir=tmp_path / "derived",
        cache_dir=tmp_path / "cache",
    )
    result = op.run(samples, config)
    assert [s.labels["audio_pass"] for s in result] == [True, False, False]
    assert all(s.is_completed("quality.filter") for s in result)
    assert result[0].lineage[-1].operator == "quality.filter"

    # unchanged inputs: the operator works on copies
    assert "audio_pass" not in samples[0].labels

    cfg = PipelineConfig(name="filter_only", input_manifest="", steps=[])
    step = PipelineStep(name="f", operator="quality.filter")
    # a manifest operator ignores concurrency settings
    assert isinstance(PipelineRunner(cfg)._run_step(step, []), list)


def test_manifest_split_is_stable_and_merge_deduplicates():
    samples = [
        Sample(id=f"s{i}", source_path=f"s{i}.wav", sha256=f"{i:064x}") for i in range(50)
    ]
    manifest = Manifest(samples)

    shards = manifest.split(4)
    assert sum(len(s) for s in shards) == 50
    # same content always lands in the same shard, regardless of input order
    reordered = Manifest(list(reversed(samples))).split(4)
    for original, again in zip(shards, reordered):
        assert {s.id for s in original} == {s.id for s in again}

    merged, report = Manifest.merge(shards)
    assert report["duplicates"] == 0
    assert len(merged) == 50
    assert [s.sha256 for s in merged.samples] == sorted(s.sha256 for s in samples)

    with_dupes, report = Manifest.merge([shards[0], shards[0]])
    assert report["duplicates"] == len(shards[0])
    assert len(with_dupes) == len(shards[0])

    kept, report = Manifest.merge([shards[0], shards[0]], keep_duplicates=True)
    assert len(kept) == 2 * len(shards[0])

    with pytest.raises(ValueError):
        manifest.split(0)

    with pytest.raises(ValueError):
        manifest.split(2, strategy="round-robin")


def test_duration_balanced_split_evens_load_and_is_deterministic():
    # Varied lengths: LPT should keep total seconds close across shards.
    samples = [
        Sample(id=f"s{i}", source_path=f"s{i}.wav", sha256=f"{i:064x}", duration=float(10 - i))
        for i in range(10)  # 10,9,...,1
    ]
    manifest = Manifest(samples)
    shards = manifest.split(2, strategy="duration-balanced")
    loads = [sum(s.duration or 0.0 for s in part.samples) for part in shards]
    assert sum(loads) == pytest.approx(55.0)
    assert abs(loads[0] - loads[1]) <= 1.0

    # Input order must not change the assignment.
    again = Manifest(list(reversed(samples))).split(2, strategy="duration-balanced")
    for original, other in zip(shards, again):
        assert {s.id for s in original} == {s.id for s in other}

    # Missing duration falls back to weight 1.0 and still covers every sample.
    bare = Manifest([Sample(id=f"n{i}", source_path=f"n{i}.wav", sha256=f"{i:064x}") for i in range(7)])
    parts = bare.split(3, strategy="duration-balanced")
    assert sum(len(p) for p in parts) == 7
    assert all(len(p) > 0 for p in parts)


def _count_processed(operator_name: str, calls: list[str]):
    """Wrap an operator so we can assert resume does not recompute finished samples."""
    from audio_engine.core.operator import BaseOperator

    original = BaseOperator.process

    def spy(self, sample, config):
        if self.full_name == operator_name:
            calls.append(sample.id)
        return original(self, sample, config)

    return original, spy


def test_checkpoint_resume_skips_completed_steps(tmp_path: Path, monkeypatch):
    """A crashed run resumes from committed parts and re-runs only what is left."""
    from audio_engine.core.operator import BaseOperator
    from audio_engine.core.pipeline import ExecutionConfig, PipelineConfig, PipelineRunner

    audio_dir = _write_wavs(tmp_path / "audio", 10)
    boom_at = {"count": 0}

    def make_cfg(**kwargs):
        return PipelineConfig(
            name="ckpt",
            input_manifest="",
            source_dir=str(audio_dir),
            steps=_cleaning_steps(),
            output_dir=tmp_path / "derived",
            cache_dir=tmp_path / "cache",
            runs_dir=tmp_path / "runs",
            execution=ExecutionConfig(executor="sequential", workers=1, checkpoint_every=4),
            **kwargs,
        )

    # crash in the middle of the resample step, after 2 checkpoint parts are committed
    original = BaseOperator.process

    def flaky(self, sample, config):
        if self.full_name == "audio.resample":
            boom_at["count"] += 1
            if boom_at["count"] > 8:
                raise KeyboardInterrupt("simulated crash")
        return original(self, sample, config)

    monkeypatch.setattr(BaseOperator, "process", flaky)
    runner = PipelineRunner(make_cfg())
    run_dir = runner.run_dir
    with pytest.raises(KeyboardInterrupt):
        runner.run()

    resample_ckpt = run_dir / "checkpoints" / "01_resample"
    state = json.loads((resample_ckpt / "_state.json").read_text(encoding="utf-8"))
    assert state["complete"] is False
    assert len(state["parts"]) == 2  # 8 samples committed in parts of 4

    # resume: ingest is restored, resample only redoes the remaining 2 samples
    monkeypatch.setattr(BaseOperator, "process", original)
    calls: list[str] = []
    _, spy = _count_processed("audio.resample", calls)
    monkeypatch.setattr(BaseOperator, "process", spy)

    resumed = PipelineRunner(make_cfg(resume=str(run_dir)))
    assert resumed.run_dir == run_dir
    result = resumed.run()

    assert len(result) == 10
    assert len(calls) == 2
    assert json.loads((resample_ckpt / "_state.json").read_text(encoding="utf-8"))["complete"]

    # a full rerun of the finished run recomputes nothing
    calls.clear()
    again = PipelineRunner(make_cfg(resume=str(run_dir))).run()
    assert len(again) == 10
    assert calls == []
    assert [s.id for s in again.samples] == [s.id for s in result.samples]


def test_checkpoint_ignores_uncommitted_and_changed_config(tmp_path: Path):
    from audio_engine.core.checkpoint import StepCheckpoint

    directory = tmp_path / "00_step"
    directory.mkdir(parents=True)
    fingerprint = {"pipeline": "abc", "operator": "audio.resample", "version": "1.1.0"}

    checkpoint = StepCheckpoint(directory, fingerprint)
    samples = [Sample(id="a", source_path="a.wav", sha256="a")]
    checkpoint.append(samples, count_in=1, counts={"processed": 1})
    checkpoint.finish(1)

    reloaded = StepCheckpoint(directory, fingerprint).load()
    assert reloaded.complete is True
    assert reloaded.consumed == 1
    assert [s.id for s in reloaded.read_samples()] == ["a"]
    assert reloaded.restored_counts()["processed"] == 1

    # a part file written but never recorded in _state.json is ignored
    Manifest([Sample(id="ghost", source_path="g.wav", sha256="g")]).save(
        directory / "part-000001.parquet"
    )
    assert len(StepCheckpoint(directory, fingerprint).load().parts) == 1

    # fingerprint change (params, operator version or input) invalidates everything
    changed = StepCheckpoint(directory, {**fingerprint, "version": "2.0.0"}).load()
    assert changed.parts == []
    assert changed.complete is False
    assert changed.consumed == 0

    # a recorded part whose file vanished truncates the usable prefix
    (directory / "part-000000.parquet").unlink()
    assert StepCheckpoint(directory, fingerprint).load().parts == []


def test_checkpoint_can_be_disabled(tmp_path: Path):
    from audio_engine.core.pipeline import ExecutionConfig, PipelineConfig, PipelineRunner

    audio_dir = _write_wavs(tmp_path / "audio", 4)
    cfg = PipelineConfig(
        name="no_ckpt",
        input_manifest="",
        source_dir=str(audio_dir),
        steps=_cleaning_steps(),
        output_dir=tmp_path / "derived",
        cache_dir=tmp_path / "cache",
        runs_dir=tmp_path / "runs",
        execution=ExecutionConfig(checkpoint_every=0),
    )
    runner = PipelineRunner(cfg)
    assert len(runner.run()) == 4
    assert not (runner.run_dir / "checkpoints").exists()

    with pytest.raises(ValueError):
        ExecutionConfig(checkpoint_every=-1)


def test_execution_config_precedence_and_validation():
    from audio_engine.core.pipeline import ExecutionConfig, PipelineConfig, PipelineStep

    step = PipelineStep(name="resample", operator="audio.resample", execution={"workers": 2})
    cfg = PipelineConfig(
        name="prec",
        input_manifest="",
        steps=[step],
        execution=ExecutionConfig(workers=8),
    )
    assert cfg.step_execution(step).workers == 2

    cfg.execution_override = {"workers": 1}
    assert cfg.step_execution(step).workers == 1
    assert cfg.step_execution(step).concurrent is False

    assert ExecutionConfig(workers=3).in_flight == 12
    assert ExecutionConfig(workers=3, max_in_flight=5).in_flight == 5

    assert ExecutionConfig(executor="process", workers=2).concurrent is True
    with pytest.raises(ValueError):
        ExecutionConfig(executor="ray")
    with pytest.raises(ValueError):
        ExecutionConfig(workers=0)
    with pytest.raises(ValueError):
        ExecutionConfig().merged({"threads": 4})
