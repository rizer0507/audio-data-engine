"""Engine-level concurrency, block persistence, run lifecycle and rule isolation."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest
from test_tagging_concurrency import CONFIG, samples

from audio_engine.core.sample import Sample
from audio_engine.core.tagging.archive import ArchivedResults, ResultArchive
from audio_engine.core.tagging.config import (
    LexiconEntry,
    LexiconSpec,
    RuleDef,
    TagDef,
    load_tagging_config,
)
from audio_engine.core.tagging.engine import MultiTagEngine
from audio_engine.core.tagging.evidence.acoustic import AcousticEvidenceView
from audio_engine.core.tagging.execution import TaggingExecution, evaluate_samples
from audio_engine.core.tagging.monitoring import TaggingMonitor
from audio_engine.core.tagging.service import TaggingRunRequest, TaggingService
from audio_engine.core.tagging.state import SampleTagState, TagStateStore


@pytest.fixture
def cfg():
    return load_tagging_config(CONFIG)


def request(tmp_path, values, **kwargs):
    return TaggingRunRequest(
        batch_id="runtime",
        config_path=CONFIG,
        samples=values,
        work_root=tmp_path / "batch",
        publish=False,
        **kwargs,
    )


def acoustic_config(cfg, conditions):
    return replace(
        cfg,
        evidence_mode="compute_acoustic",
        tags={str(i): TagDef(str(i), str(i)) for i in range(len(conditions))},
        rules=[RuleDef(str(i), "1", str(i), condition) for i, condition in enumerate(conditions)],
    )


def test_engines_overlap_with_one_sample_worker(cfg, tmp_path, monkeypatch):
    import audio_engine.core.tagging.scheduling as module

    barrier = threading.Barrier(2)

    class Adapter:
        def __init__(self):
            self.owner = threading.get_ident()

        def compute(self, sample, *, deps, params):
            assert self.owner == threading.get_ident()
            barrier.wait(5)  # VAD and DNSMOS must run together even with workers=1.
            if "vad" in deps:
                return AcousticEvidenceView(vad_status="success", vad_speech_state="present")
            return AcousticEvidenceView(dnsmos_status="success", dnsmos_sig=1.0)

    monkeypatch.setattr(module, "RealAcousticAdapter", Adapter)
    config = acoustic_config(
        cfg, [{"all": [{"capability": "vad_speech_present"}, {"capability": "dnsmos_sig_low"}]}]
    )
    result = TaggingService(config).run(
        request(tmp_path, samples(1), workers=1, engine_workers={"vad": 1, "dnsmos": 1})
    )
    assert result.results[0].effective_tags == ["0"]
    for dep in ("vad", "dnsmos"):
        assert result.metrics["engines"][dep]["completed"] == 1
        assert result.metrics["engines"][dep]["in_flight"] == 0


def test_engine_limits_and_one_call_for_multiple_tags(cfg, tmp_path, monkeypatch):
    import audio_engine.core.tagging.scheduling as module

    active, peak, calls = 0, 0, []
    lock = threading.Lock()
    barrier = threading.Barrier(2)

    class Adapter:
        def compute(self, sample, *, deps, params):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
                calls.append((sample.id, set(deps)))
                position = len(calls)
            if position <= 2:
                barrier.wait(5)
            time.sleep(0.005)
            with lock:
                active -= 1
            return AcousticEvidenceView(dnsmos_status="success", dnsmos_sig=1.0)

    monkeypatch.setattr(module, "RealAcousticAdapter", Adapter)
    config = acoustic_config(cfg, [{"capability": "dnsmos_sig_low"}] * 3)
    service = TaggingService(config)
    result = service.run(
        request(
            tmp_path,
            samples(12),
            workers=4,
            batch_size=1,
            engine_workers={"dnsmos": 2},
            engine_max_in_flight={"dnsmos": 3},
        )
    )
    assert peak == 2
    assert len(calls) == 12  # Three tags consume the same model fact.
    metrics = result.metrics["engines"]["dnsmos"]
    assert metrics["peak_running"] == 2
    assert metrics["peak_in_flight"] <= 3
    assert metrics["completed"] == 12
    assert all(r.effective_tags == ["0", "1", "2"] for r in result.results)
    known = samples(2)
    for sample in known:
        sample.quality = {"dnsmos_status": "success", "dnsmos_sig": 1.0}
    reused = service.run(request(tmp_path / "reused", known, workers=2))
    assert len(calls) == 12
    assert reused.metrics["engines"]["dnsmos"]["reused"] == 2


def test_engine_retry_is_bounded_and_logged(cfg, tmp_path, monkeypatch):
    import audio_engine.core.tagging.scheduling as module

    class Adapter:
        def __init__(self):
            self.calls = 0

        def compute(self, sample, *, deps, params):
            self.calls += 1
            return AcousticEvidenceView(
                dnsmos_status="failed" if self.calls == 1 else "success", dnsmos_sig=1.0
            )

    monkeypatch.setattr(module, "RealAcousticAdapter", Adapter)
    result = TaggingService(acoustic_config(cfg, [{"capability": "dnsmos_sig_low"}])).run(
        request(tmp_path, samples(1), workers=1, engine_retries=1)
    )
    assert result.metrics["engines"]["dnsmos"]["attempts"] == 2
    assert result.metrics["engines"]["dnsmos"]["retries"] == 1
    assert result.metrics["engines"]["dnsmos"]["failed"] == 0
    assert "engine_retry" in (Path(result.run_dir) / "tagging.log.jsonl").read_text(
        encoding="utf-8"
    )


def test_block_dispatch_reduces_futures_and_bounds_input(cfg, monkeypatch):
    original = MultiTagEngine.evaluate_sample
    barrier = threading.Barrier(2)
    produced = 0

    def evaluate(self, sample, **kwargs):
        if sample.id in {"s0", "s3"}:
            barrier.wait(5)
        return original(self, sample, **kwargs)

    monkeypatch.setattr(MultiTagEngine, "evaluate_sample", evaluate)

    def jobs():
        nonlocal produced
        for i, sample in enumerate(samples(12)):
            produced += 1
            yield i, sample, SampleTagState(sample.id)

    stream = evaluate_samples(
        iter(jobs()),
        config=cfg,
        execution=TaggingExecution(2, "thread", 6, 3),
        adapter=None,
        snapshot_id="test",
        rule_ids=None,
        preview=False,
    )
    first = next(stream)
    assert produced == 6
    assert len([first, *stream]) == 12


def test_partial_block_failure_is_checkpointed_and_visible(cfg, tmp_path, monkeypatch):
    original = MultiTagEngine.evaluate_sample

    def evaluate(self, sample, **kwargs):
        if sample.id == "s1":
            raise ValueError("block failure")
        return original(self, sample, **kwargs)

    monkeypatch.setattr(MultiTagEngine, "evaluate_sample", evaluate)
    with pytest.raises(RuntimeError, match="sample_id=s1"):
        TaggingService(cfg).run(
            request(tmp_path, samples(6), workers=2, batch_size=3, max_in_flight=6)
        )
    work = next((tmp_path / "batch" / "work").iterdir())
    metrics = json.loads((work / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["status"] == "failed"
    assert metrics["engines"]["text"]["in_flight"] == 0
    assert (
        "s0"
        in json.loads((work / "progress.json").read_text(encoding="utf-8"))["completed_sample_ids"]
    )
    assert "block failure" in (work / "tagging.log.jsonl").read_text(encoding="utf-8")


def test_pipeline_context_auto_resume_and_input_drift(cfg, tmp_path, monkeypatch):
    service = TaggingService(cfg)
    parent = tmp_path / "runs" / "pipeline_run"
    first = service.run(
        request(
            tmp_path,
            samples(3),
            workers=1,
            pipeline_run_dir=parent,
            step_name="multi_tag",
            retain_results=False,
        )
    )
    assert isinstance(first.results, ArchivedResults)
    directory = Path(first.run_dir)
    assert directory.is_relative_to(parent)
    meta = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    assert meta["pipeline_run_id"] == "pipeline_run"
    assert meta["status"] == "completed"
    assert Path(meta["checkpoint_path"]).exists()
    records = [
        json.loads(line)["record"]
        for line in (directory / "tagging.log.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert all(record["extra"]["pipeline_run_id"] == "pipeline_run" for record in records)
    assert first.results[0].as_dict() == next(iter(first.results)).as_dict()
    original = MultiTagEngine.evaluate_sample
    calls = []

    def evaluate(self, sample, **kwargs):
        calls.append(sample.id)
        return original(self, sample, **kwargs)

    monkeypatch.setattr(MultiTagEngine, "evaluate_sample", evaluate)
    resumed = service.run(
        request(
            tmp_path,
            samples(3),
            workers=1,
            pipeline_run_dir=parent,
            pipeline_resume=True,
            retain_results=False,
        )
    )
    assert resumed.run_id == first.run_id
    assert calls == []
    changed = samples(3)
    changed[0].transcripts["qwen_1"]["raw_text"] = "changed raw evidence"
    restarted = service.run(
        request(
            tmp_path,
            changed,
            workers=1,
            pipeline_run_dir=parent,
            pipeline_resume=True,
            retain_results=False,
        )
    )
    assert restarted.run_id != first.run_id
    assert calls == ["s0", "s1", "s2"]


def test_lazy_results_overlay_current_manual_tags(cfg, tmp_path):
    service = TaggingService(cfg)
    first = service.run(request(tmp_path, samples(1), workers=1, retain_results=False))
    assert "tag_1" in first.results[0].effective_tags
    service.remove_tag(
        batch_id="runtime",
        sample_id="s0",
        tag_id="tag_1",
        audio_sha256="a" * 64,
        actor="test",
        reason="manual",
        work_root=tmp_path / "batch",
    )
    resumed = service.run(
        request(tmp_path, samples(1), workers=1, retain_results=False, resume_run_id=first.run_id)
    )
    assert "tag_1" not in resumed.results[0].effective_tags
    assert next(iter(resumed.results)).effective_tags == resumed.samples[0].labels["tags"]


def test_archive_transaction_markers_and_colliding_legacy_names(tmp_path):
    archive = ResultArchive(tmp_path)
    try:
        archive.add("a/b", {"sample_id": "a/b", "text": "甲"})
        archive.add("a_b", {"sample_id": "a_b", "text": "乙"})
        archive.flush()
        assert archive.completed_ids() == set()
        assert archive.load("a/b")["text"] == "甲"
        assert archive.load("a_b")["text"] == "乙"
        archive.commit(["a/b"])
        assert archive.completed_ids() == {"a/b"}
        archive.invalidate(["a/b"])
        assert archive.completed_ids() == set()
        assert len(list((tmp_path / "sample_results" / "blocks").glob("*.jsonl"))) == 1
    finally:
        archive.close()


def test_compact_progress_and_lazy_resume(cfg, tmp_path, monkeypatch):
    config = replace(
        cfg,
        tags={"all": TagDef("all", "all")},
        rules=[RuleDef("all", "1", "all", {"capability": "true"})],
    )
    values = [Sample(id=f"x{i}", source_path=f"x{i}.wav") for i in range(4100)]
    service = TaggingService(config)
    first = service.run(
        request(tmp_path, values, workers=1, retain_results=False, checkpoint_every=512)
    )
    progress = json.loads((Path(first.work_dir) / "progress.json").read_text(encoding="utf-8"))
    assert progress["completed_sample_ids"] == []
    assert progress["completed_count"] == 4100
    assert progress["completed_store"] == "sample_results.sqlite3"
    assert len(list((Path(first.work_dir) / "sample_results" / "blocks").glob("*.jsonl"))) == 9

    def fail(*args, **kwargs):
        raise AssertionError("resume must not evaluate committed samples")

    monkeypatch.setattr(MultiTagEngine, "evaluate_sample", fail)
    resumed = service.run(
        request(tmp_path, values, workers=1, retain_results=False, resume_run_id=first.run_id)
    )
    assert len(resumed.results) == 4100
    assert resumed.metrics["progress"]["restored"] == 4100


def test_indexed_state_import_and_optimistic_conflict(tmp_path):
    from audio_engine.core.tagging.state import ConcurrentModificationError

    store = TagStateStore(tmp_path / "tag_state.json")
    old = SampleTagState("legacy")
    store.put(old)
    store.enable_indexed()
    try:
        state = store.get("legacy")
        assert state.version == old.version
        other = TagStateStore(store.path)
        try:
            other.put(other.get("legacy"))
            with pytest.raises(ConcurrentModificationError):
                store.put_many([state])
        finally:
            other.close()
    finally:
        store.close()


def test_distinct_rule_lexicons_stay_isolated(cfg, tmp_path):
    import re

    def lexicon(name, text):
        return LexiconSpec(
            name, "1", name, name, entries=[LexiconEntry(name, text, compiled=re.compile(text))]
        )

    config = replace(
        cfg,
        tags={name: TagDef(name, name) for name in ("yes", "no")},
        lexicons={"yes": lexicon("yes", "好"), "no": lexicon("no", "坏")},
        rules=[
            RuleDef(
                name,
                "1",
                name,
                {
                    "capability": "lexicon_match",
                    "lexicon": name,
                    "text": {"source": "family_representative", "family": "sensevoice"},
                },
            )
            for name in ("yes", "no")
        ],
    )
    result = TaggingService(config).run(request(tmp_path, samples(6), workers=2, batch_size=2))
    assert all(r.effective_tags == ["yes"] for r in result.results)


def test_live_heartbeat_reports_running_engine(tmp_path):
    monitor = TaggingMonitor(tmp_path, {"tagging_run_id": "test"}, interval=0.02)
    monitor.update(total=1)
    monitor.engine("dnsmos", submitted=1, running=1)
    monitor.start()
    try:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            snapshot = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
            if snapshot["updated_at"] > snapshot["started_at"] + 0.015:
                break
            time.sleep(0.01)
        assert snapshot["engines"]["dnsmos"]["running"] == 1
        assert snapshot["status"] == "running"
    finally:
        monitor.finish_engine("dnsmos", 0.02)
        monitor.finish("completed")


def test_run_lease_prevents_duplicate_resume_and_releases_on_exception(tmp_path):
    from audio_engine.core.tagging.archive import RunLease

    with pytest.raises(ValueError, match="interrupted"), RunLease(tmp_path):
        with pytest.raises(RuntimeError, match="already active"), RunLease(tmp_path):
            pytest.fail("duplicate run entered")
        raise ValueError("interrupted")
    with RunLease(tmp_path):
        pass


def test_progress_cli_reads_pipeline_tagging_metrics(cfg, tmp_path):
    from typer.testing import CliRunner

    from audio_engine.cli.main import app

    parent = tmp_path / "pipeline"
    run = TaggingService(cfg).run(request(tmp_path, samples(2), workers=1, pipeline_run_dir=parent))
    runner = CliRunner()
    for path in (parent, Path(run.work_dir)):
        response = runner.invoke(app, ["pipeline", "progress", str(path)])
        assert response.exit_code == 0, response.output
        assert "done=2/2" in response.output
        assert "status=completed" in response.output
        assert "asr_text" in response.output


def test_chunked_parquet_preserves_dynamic_columns_and_null_first_block(tmp_path):
    import pandas as pd

    from audio_engine.core.manifest import Manifest
    from audio_engine.core.tagging.export import write_tagged_parquet

    values = samples(2)
    values[0].quality = {"dnsmos_sig": None}
    values[1].quality = {"dnsmos_sig": 1.0, "new_metric": 2.0}
    values[1].labels["new_label"] = {"nested": "value"}
    path = tmp_path / "tagged.parquet"
    write_tagged_parquet(path, values, batch_size=1)
    restored = Manifest.load(path).samples
    assert [s.id for s in restored] == [s.id for s in values]
    assert restored[1].quality == values[1].quality
    assert restored[1].labels == values[1].labels
    assert "quality_new_metric" in pd.read_parquet(path).columns


def test_legacy_result_resume_into_block_archive(cfg, tmp_path):
    from audio_engine.core.tagging.persist import save_progress, save_sample_result

    service = TaggingService(cfg)
    ws = service.workspace("runtime", work_root=tmp_path / "batch")
    sample = samples(1)[0]
    result = MultiTagEngine(cfg).evaluate_sample(sample)
    save_sample_result(
        ws, sample.id, {**result.as_dict(), "tagging": {"eval_status": result.eval_status}}
    )
    save_progress(
        ws,
        {
            "completed_sample_ids": [sample.id],
            "definition_fingerprint": cfg.content_fingerprint,
            "config_fingerprint": cfg.content_fingerprint,
        },
    )
    resumed = service.run(
        request(tmp_path, [sample], workers=1, resume_run_id=ws.run_id, retain_results=False)
    )
    assert resumed.metrics["progress"]["restored"] == 1
    assert len(resumed.results[0].rule_results) == len(result.rule_results)
    archive = ResultArchive(ws.work_dir, readonly=True)
    try:
        assert archive.completed_ids() == {sample.id}
    finally:
        archive.close()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_size": 0},
        {"engine_workers": {"dnsmos": 0}},
        {"engine_max_in_flight": {"vad": 0}},
        {"engine_retries": -1},
        {"engine_workers": {"unknown": 2}},
    ],
)
def test_invalid_runtime_limits_rejected(cfg, tmp_path, kwargs):
    with pytest.raises(ValueError):
        TaggingService(acoustic_config(cfg, [{"capability": "dnsmos_sig_low"}])).run(
            request(tmp_path, samples(1), **kwargs)
        )
