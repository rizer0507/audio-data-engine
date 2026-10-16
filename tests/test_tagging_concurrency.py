"""Real worker overlap, backpressure, stats, and recoverable concurrent failures."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

from audio_engine.core.sample import Sample
from audio_engine.core.tagging.config import RuleDef, TagDef, load_tagging_config
from audio_engine.core.tagging.engine import MultiTagEngine
from audio_engine.core.tagging.evidence.acoustic import AcousticEvidenceView
from audio_engine.core.tagging.execution import TaggingExecution, evaluate_samples
from audio_engine.core.tagging.service import TaggingRunRequest, TaggingService
from audio_engine.core.tagging.state import (
    ConcurrentModificationError,
    SampleTagState,
    TagStateStore,
)

CONFIG = Path(__file__).resolve().parents[1] / "configs/tagging/zh_asr_free_tags_v2.yaml"


@pytest.fixture
def cfg():
    return load_tagging_config(CONFIG)


def samples(n):
    return [
        Sample(
            id=f"s{i}",
            source_path=f"s{i}.wav",
            sha256="a" * 64,
            transcripts={
                key: {"text": text, "raw_text": text, "status": "success"}
                for key, text in (
                    ("glm_1", "好"),
                    ("glm_2", "好"),
                    ("sensevoice_1", "好"),
                    ("sensevoice_2", "好"),
                    ("qwen_1", "要"),
                    ("qwen_2", "要"),
                )
            },
        )
        for i in range(n)
    ]


def request(tmp_path, values, **kwargs):
    return TaggingRunRequest(
        batch_id="concurrency",
        config_path=CONFIG,
        samples=values,
        work_root=tmp_path,
        publish=False,
        **kwargs,
    )


def test_worker_overlap_backpressure_and_completion_order(cfg, monkeypatch):
    barrier = threading.Barrier(2)
    original = MultiTagEngine.evaluate_sample
    produced = 0

    def evaluate(self, sample, **kwargs):
        if sample.id in {"s0", "s1"}:
            barrier.wait(timeout=5)  # Would fail if the implementation were sequential.
        if sample.id == "s0":
            time.sleep(0.05)
        return original(self, sample, **kwargs)

    monkeypatch.setattr(MultiTagEngine, "evaluate_sample", evaluate)

    def jobs():
        nonlocal produced
        for index, sample in enumerate(samples(6)):
            produced += 1
            yield index, sample, SampleTagState(sample.id)

    stream = evaluate_samples(
        iter(jobs()),
        config=cfg,
        execution=TaggingExecution(2, "thread", 2),
        adapter=None,
        snapshot_id="test",
        rule_ids=None,
        preview=False,
    )
    first = next(stream)
    assert first[1].id == "s1"  # Slow first input does not block later completions.
    assert produced == 2
    rest = list(stream)
    assert sorted(x[0] for x in [first, *rest]) == list(range(6))


@pytest.mark.parametrize("executor", ["thread", "process"])
def test_service_matches_serial_and_keeps_input_order(cfg, tmp_path, executor):
    service = TaggingService(cfg)
    serial = service.run(request(tmp_path / "serial", samples(5), workers=1))
    parallel = service.run(
        request(
            tmp_path / executor,
            samples(5),
            workers=2,
            executor=executor,
            max_in_flight=2,
            checkpoint_every=2,
        )
    )
    assert [s.id for s in parallel.samples] == [s.id for s in serial.samples]
    assert [r.effective_tags for r in parallel.results] == [
        r.effective_tags for r in serial.results
    ]
    assert [[rr.as_dict() for rr in r.rule_results] for r in parallel.results] == [
        [rr.as_dict() for rr in r.rule_results] for r in serial.results
    ]
    assert parallel.stats == serial.stats
    entries = [
        json.loads(line)
        for line in (Path(parallel.work_dir) / "tagging.log.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert all(e["record"]["extra"]["tagging_run_id"] == parallel.run_id for e in entries)
    assert any("workers=2" in e["record"]["message"] for e in entries)
    checkpoints = [e for e in entries if "tagging checkpoint" in e["record"]["message"]]
    assert len(checkpoints) == 3  # 5 samples / 2, instead of 5 whole-state rewrites.


def test_failure_flushes_completed_work_and_resume_skips_it(cfg, tmp_path, monkeypatch):
    import audio_engine.core.tagging.service as module

    committed = threading.Event()
    original_evaluate = MultiTagEngine.evaluate_sample
    original_save = module.save_sample_result

    def save(ws, sample_id, payload):
        result = original_save(ws, sample_id, payload)
        if sample_id == "s1":
            committed.set()
        return result

    def evaluate(self, sample, **kwargs):
        if sample.id == "s0":
            assert committed.wait(5)
            raise ValueError("injected failure")
        return original_evaluate(self, sample, **kwargs)

    monkeypatch.setattr(module, "save_sample_result", save)
    monkeypatch.setattr(MultiTagEngine, "evaluate_sample", evaluate)
    service = TaggingService(cfg)
    with pytest.raises(RuntimeError, match="sample_id=s0"):
        service.run(
            request(tmp_path, samples(2), workers=2, executor="thread", checkpoint_every=100)
        )
    work = next((tmp_path / "work").iterdir())
    progress = json.loads((work / "progress.json").read_text(encoding="utf-8"))
    assert progress["completed_sample_ids"] == ["s1"]
    assert "sample_id=s0" in (work / "tagging.log.jsonl").read_text(encoding="utf-8")
    assert not (tmp_path / "published" / "CURRENT.json").exists()

    evaluated = []

    def record(self, sample, **kwargs):
        evaluated.append(sample.id)
        return original_evaluate(self, sample, **kwargs)

    monkeypatch.setattr(MultiTagEngine, "evaluate_sample", record)
    resumed = service.run(
        request(tmp_path, samples(2), workers=2, executor="thread", resume_run_id=work.name)
    )
    assert evaluated == ["s0"]
    assert [s.id for s in resumed.samples] == ["s0", "s1"]
    assert "restored_sample:s1" in resumed.messages


def test_default_acoustic_sessions_are_isolated_and_stats_exact(cfg, tmp_path, monkeypatch):
    import audio_engine.core.tagging.scheduling as module

    barrier = threading.Barrier(2)
    instances = []

    class Adapter:
        def __init__(self):
            self.owner = threading.get_ident()
            instances.append(self)

        def compute(self, sample, *, deps, params):
            assert self.owner == threading.get_ident()
            barrier.wait(timeout=5)
            return AcousticEvidenceView(energy_status="success", strict_all_channels_zero=True)

    monkeypatch.setattr(module, "RealAcousticAdapter", Adapter)
    acoustic_cfg = replace(
        cfg,
        evidence_mode="compute_acoustic",
        tags={"zero": TagDef("zero", "zero")},
        rules=[RuleDef("zero", "1", "zero", {"capability": "strict_all_channels_zero"})],
    )
    result = TaggingService(acoustic_cfg).run(
        request(tmp_path, samples(4), workers=2, max_in_flight=2)
    )
    assert len(instances) == 2
    assert result.stats["actual_calls"]["energy"] == 4
    assert result.stats["failed"]["energy"] == 0
    assert all(r.effective_tags == ["zero"] for r in result.results)
    assert all(s.quality["energy_status"] == "success" for s in result.samples)


def test_bulk_state_commit_refuses_manual_edit_conflict(tmp_path):
    store = TagStateStore(tmp_path / "state.json")
    prior = store.get("s0")
    store.put(store.get("s0"))
    with pytest.raises(ConcurrentModificationError, match="version conflict"):
        store.put_many([prior])
    assert store.get("s0").version == 1


def test_failed_force_recompute_invalidates_old_completion_markers(cfg, tmp_path, monkeypatch):
    service = TaggingService(cfg)
    original = MultiTagEngine.evaluate_sample
    first = service.run(request(tmp_path, samples(2), workers=1))

    def fail(self, sample, **kwargs):
        raise ValueError("recompute interrupted")

    monkeypatch.setattr(MultiTagEngine, "evaluate_sample", fail)
    with pytest.raises(ValueError, match="recompute interrupted"):
        service.run(
            request(
                tmp_path, samples(2), workers=1, force_recompute=True, resume_run_id=first.run_id
            )
        )
    progress = json.loads((Path(first.work_dir) / "progress.json").read_text(encoding="utf-8"))
    assert progress["completed_sample_ids"] == []
    evaluated = []

    def record(self, sample, **kwargs):
        evaluated.append(sample.id)
        return original(self, sample, **kwargs)

    monkeypatch.setattr(MultiTagEngine, "evaluate_sample", record)
    resumed = service.run(request(tmp_path, samples(2), workers=1, resume_run_id=first.run_id))
    assert evaluated == ["s0", "s1"]
    assert not resumed.messages


@pytest.mark.parametrize(
    "kwargs",
    [{"workers": 0}, {"max_in_flight": 0}, {"executor": "invalid"}, {"checkpoint_every": 0}],
)
def test_invalid_execution_rejected(cfg, tmp_path, kwargs):
    with pytest.raises(ValueError):
        TaggingService(cfg).run(request(tmp_path, samples(2), **kwargs))


def test_duplicate_sample_ids_rejected(cfg, tmp_path):
    values = samples(1) * 2
    with pytest.raises(ValueError, match="unique sample IDs"):
        TaggingService(cfg).run(request(tmp_path, values))


def test_auto_executor_avoids_small_batch_spawn_cost():
    assert TaggingExecution.resolve(8, "auto", None, compute=False, total=64).executor == "thread"
    assert (
        TaggingExecution.resolve(8, "auto", None, compute=False, total=30000).executor == "process"
    )
    assert TaggingExecution.resolve(8, "auto", None, compute=True, total=30000).executor == "thread"
    assert (
        TaggingExecution.resolve(1, "auto", None, compute=False, total=30000).executor
        == "sequential"
    )


@pytest.mark.parametrize("zero_limits", [False, True])
def test_pipeline_cli_overrides_reach_tag_service(tmp_path, monkeypatch, zero_limits):
    from audio_engine.core.pipeline import PipelineConfig, PipelineRunner, PipelineStep
    from audio_engine.operators.quality.multi_tag import MultiTagV1Operator

    received = []

    def run(self, values, config):
        received.append(config.params)
        return values

    monkeypatch.setattr(MultiTagV1Operator, "run", run)
    monkeypatch.setattr(PipelineRunner, "_open_checkpoint", lambda *args: None)
    step = PipelineStep(
        name="tags",
        operator="quality.multi_tag_v1",
        params={"workers": 8, "executor": "auto", "max_in_flight": 16, "checkpoint_every": 100},
    )
    runner = PipelineRunner(
        PipelineConfig(
            name="tags",
            input_manifest="",
            steps=[step],
            runs_dir=tmp_path,
            execution_override={
                "workers": 3,
                "executor": "thread",
                "max_in_flight": 0 if zero_limits else 5,
                "checkpoint_every": 0 if zero_limits else 7,
            },
        )
    )
    runner._run_step(step, samples(2))
    assert received[0]["workers"] == 3
    assert received[0]["executor"] == "thread"
    assert received[0]["max_in_flight"] == (None if zero_limits else 5)
    assert received[0]["checkpoint_every"] == (100 if zero_limits else 7)
