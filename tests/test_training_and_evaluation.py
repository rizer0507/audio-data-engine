from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import audio_engine.operators  # noqa: F401
from audio_engine.core.catalog import ArtifactCatalog, DatasetRelease
from audio_engine.core.operator import OperatorConfig
from audio_engine.core.registry import OperatorRegistry
from audio_engine.core.sample import Sample
from audio_engine.core.training import run_training_job


def _release(catalog: ArtifactCatalog, tmp_path: Path) -> DatasetRelease:
    ids = {}
    for name in ("source", "train", "dev", "test"):
        path = tmp_path / f"{name}.parquet"
        path.write_bytes(name.encode())
        ids[name] = catalog.register_file(path, kind="manifest").artifact_id
    return catalog.put_release(
        DatasetRelease(
            release_id="ds_v1",
            source_artifact_id=ids["source"],
            outputs={key: ids[key] for key in ("train", "dev", "test")},
            policy_version="p1",
            normalization_version="n1",
            gold_revision="g1",
            split_seed=1,
            group_key="speaker_id",
        )
    )


def test_training_adapter_registers_model_and_is_idempotent(tmp_path: Path):
    catalog = ArtifactCatalog(tmp_path / "catalog")
    _release(catalog, tmp_path)
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text("epochs: 1\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.bin"
    script = tmp_path / "trainer.py"
    script.write_text(
        "import os, pathlib; pathlib.Path(os.environ['AUDIO_DATA_CHECKPOINT']).write_text('ok')",
        encoding="utf-8",
    )
    kwargs = {
        "catalog": catalog,
        "jobs_dir": tmp_path / "jobs",
        "release_id": "ds_v1",
        "recipe": recipe,
        "command": [sys.executable, str(script)],
        "checkpoint": checkpoint,
        "model_id": "model_v1",
        "base_model": "base",
    }
    first_job, first_model = run_training_job(**kwargs)
    second_job, second_model = run_training_job(**kwargs)
    assert first_job.status == "succeeded"
    assert second_job.job_id == first_job.job_id
    assert second_model == first_model == catalog.get_model("model_v1")
    assert (tmp_path / "jobs" / first_job.job_id / "input.json").exists()


def _evaluated_sample(sample_id: str, old_errors: int, new_errors: int, bucket: str) -> Sample:
    return Sample(
        id=sample_id,
        source_path=f"{sample_id}.wav",
        labels={"classification_bucket": bucket},
        quality={
            "old_substitutions": old_errors,
            "old_deletions": 0,
            "old_insertions": 0,
            "old_reference_length": 10,
            "new_substitutions": new_errors,
            "new_deletions": 0,
            "new_insertions": 0,
            "new_reference_length": 10,
        },
    )


def test_evaluation_report_uses_corpus_cer_and_gates(tmp_path: Path):
    samples = [
        _evaluated_sample("a", 2, 1, "normal"),
        _evaluated_sample("b", 4, 2, "hardcase"),
    ]
    operator = OperatorRegistry.get("quality.evaluation_report")
    operator.run(
        samples,
        OperatorConfig(
            run_dir=tmp_path / "run",
            params={
                "baseline_prefix": "old",
                "candidate_prefix": "new",
                "gates": [
                    {"name": "overall", "max_cer_regression": 0},
                    {"name": "hardcase", "bucket": "hardcase", "max_cer_regression": 0},
                ],
            },
        ),
    )
    report = json.loads((tmp_path / "run/reports/evaluation.json").read_text(encoding="utf-8"))
    assert report["overall"]["old"]["corpus_cer"] == pytest.approx(0.3)
    assert report["overall"]["new"]["corpus_cer"] == pytest.approx(0.15)
    assert len(report["paired_bootstrap"]["ci95"]) == 2
    assert report["passed"] is True
    assert (tmp_path / "run/reports/evaluation.xlsx").exists()


def test_evaluation_gate_failure_still_writes_report(tmp_path: Path):
    operator = OperatorRegistry.get("quality.evaluation_report")
    with pytest.raises(ValueError, match="regression gate failed"):
        operator.run(
            [_evaluated_sample("a", 1, 2, "normal")],
            OperatorConfig(
                run_dir=tmp_path / "run",
                params={
                    "baseline_prefix": "old",
                    "candidate_prefix": "new",
                    "gates": [{"name": "overall", "max_cer_regression": 0}],
                },
            ),
        )
    assert (tmp_path / "run/reports/evaluation.json").exists()


def test_evaluation_report_skips_missing_gold_and_notes_coverage(tmp_path: Path):
    scored = _evaluated_sample("a", 1, 0, "标注一致")
    scored.labels["gold_text"] = "你好"
    scored.transcripts = {
        "old": {"text": "你好"},
        "new": {"text": "你好"},
    }
    missing = Sample(
        id="b",
        source_path="b.wav",
        labels={"type": "面谈审批"},
        transcripts={"old": {"text": "x"}, "new": {"text": "y"}},
        quality={},
    )
    operator = OperatorRegistry.get("quality.evaluation_report")
    operator.run(
        [scored, missing],
        OperatorConfig(
            run_dir=tmp_path / "run",
            params={
                "baseline_prefix": "old",
                "candidate_prefix": "new",
                "bucket_key": "type",
                "allow_missing_gold": True,
                "fail_on_regression": False,
            },
        ),
    )
    report = json.loads((tmp_path / "run/reports/evaluation.json").read_text(encoding="utf-8"))
    assert report["gold_coverage"]["total"] == 2
    assert report["gold_coverage"]["scored"] == 1
    assert report["gold_coverage"]["missing_gold"] == 1
    assert report["gold_coverage"]["missing_gold_ids"] == ["b"]
    xlsx = tmp_path / "run/reports/evaluation.xlsx"
    assert xlsx.exists()
    frame = __import__("pandas").read_excel(xlsx, sheet_name="结果")
    assert list(frame["id"]) == ["a", "b"]
    assert bool(frame.loc[frame["id"] == "b", "eval_scored"].iloc[0]) is False


def test_text_metrics_skip_missing_reference():
    operator = OperatorRegistry.get("quality.text_metrics")
    sample = Sample(
        id="x",
        source_path="x.wav",
        labels={},
        transcripts={"old_model": {"text": "你好"}, "new_model": {"text": "你好"}},
    )
    config = OperatorConfig(
        params={
            "config_path": "configs/metrics/model_eval.yaml",
            "normalization_path": "configs/normalization/zh_asr_v1.yaml",
            "skip_missing_reference": True,
            "eval_hypotheses": ["old_model", "new_model"],
        }
    )
    result = operator.process(sample, config)
    assert result.sample.quality.get("eval_skipped_no_gold") is True
    assert "old_model_cer" not in result.sample.quality


def test_copy_transcripts_only_if_missing():
    operator = OperatorRegistry.get("quality.copy_transcripts")
    sample = Sample(
        id="x",
        source_path="x.wav",
        transcripts={"qwen": {"text": "来自qwen"}, "old_model": {"text": "已有"}},
    )
    updated = operator.process(
        sample,
        OperatorConfig(params={"mapping": {"qwen": "old_model"}, "only_if_missing": True}),
    ).sample
    assert updated.get_transcript_text("old_model") == "已有"
    filled = operator.process(
        Sample(id="y", source_path="y.wav", transcripts={"qwen": {"text": "来自qwen"}}),
        OperatorConfig(params={"mapping": {"qwen": "old_model"}, "only_if_missing": True}),
    ).sample
    assert filled.get_transcript_text("old_model") == "来自qwen"
