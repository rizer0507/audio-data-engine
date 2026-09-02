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
    report = json.loads((tmp_path / "run/reports/evaluation.json").read_text())
    assert report["overall"]["old"]["corpus_cer"] == pytest.approx(0.3)
    assert report["overall"]["new"]["corpus_cer"] == pytest.approx(0.15)
    assert len(report["paired_bootstrap"]["ci95"]) == 2
    assert report["passed"] is True


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
