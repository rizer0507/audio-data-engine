"""Tests for CLI --source-name manifest naming."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from audio_engine.core.manifest import Manifest
from audio_engine.core.sample import Sample
from audio_engine.core.source_naming import (
    apply_source_name_to_single_pipeline,
    cleaned_output_path,
    expand_layout_templates,
    manifest_path,
    pipeline_run_name,
    resolve_existing_manifest,
    validate_source_name,
)


def test_pipeline_run_name():
    assert pipeline_run_name("data_cleaning_source_A", "test_local") == (
        "data_cleaning_test_local"
    )
    assert pipeline_run_name("multi_asr_aggregate", "mt3000") == (
        "multi_asr_aggregate_mt3000"
    )
    assert pipeline_run_name("data_cleaning_source_A", None) == "data_cleaning_source_A"


def test_validate_source_name():
    assert validate_source_name("mt3000") == "mt3000"
    assert validate_source_name("mt-3000") == "mt-3000"
    with pytest.raises(ValueError):
        validate_source_name("../x")
    with pytest.raises(ValueError):
        validate_source_name("")


def test_manifest_paths():
    assert cleaned_output_path("mt3000") == "datasets/manifests/cleaned_mt3000.parquet"
    assert manifest_path("qwen_asr", "mt3000").as_posix() == (
        "datasets/manifests/qwen_asr_mt3000.parquet"
    )


def test_resolve_existing_prefers_parquet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    manifests = tmp_path / "datasets" / "manifests"
    manifests.mkdir(parents=True)
    parquet = manifests / "cleaned_mt3000.parquet"
    jsonl = manifests / "cleaned_mt3000.jsonl"
    Manifest([Sample(id="a", source_path="/tmp/a.wav", duration=1.0)]).save(parquet)
    Manifest([Sample(id="b", source_path="/tmp/b.wav", duration=1.0)]).save(jsonl)

    resolved = resolve_existing_manifest("cleaned_mt3000")
    assert resolved == parquet.resolve()


def test_expand_layout_templates():
    pairs = expand_layout_templates(None, "mt3000")
    assert pairs == [
        ("cleaned_mt3000", "datasets/manifests/qwen_asr_mt3000.parquet"),
        ("qwen_asr_mt3000", "datasets/manifests/multi_asr_aggregate_mt3000.parquet"),
    ]


def test_apply_cleaning_and_qwen_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    wav_dir = tmp_path / "wav"
    wav_dir.mkdir()
    manifests = tmp_path / "datasets" / "manifests"
    manifests.mkdir(parents=True)
    Manifest([Sample(id="a", source_path="/tmp/a.wav", duration=1.0)]).save(
        manifests / "cleaned_mt3000.parquet"
    )

    class _Step:
        def __init__(self, operator: str, params: dict | None = None):
            self.operator = operator
            self.params = params or {}

    cleaning = apply_source_name_to_single_pipeline(
        pipeline_name="data_cleaning_source_A",
        steps=[_Step("ingest.scan")],
        source_name="mt3000",
        source_dir=wav_dir,
    )
    assert cleaning["output_manifest"].endswith("cleaned_mt3000.parquet")
    assert Path(cleaning["source_dir"]) == wav_dir.resolve()

    qwen = apply_source_name_to_single_pipeline(
        pipeline_name="qwen_asr_batch",
        steps=[_Step("asr.qwen_batch")],
        source_name="mt3000",
    )
    assert qwen["input_manifest"].endswith("cleaned_mt3000.parquet")
    assert qwen["output_manifest"].endswith("qwen_asr_mt3000.parquet")

    sensevoice = apply_source_name_to_single_pipeline(
        pipeline_name="sensevoice_asr_batch",
        steps=[_Step("asr.sensevoice_batch")],
        source_name="mt3000",
    )
    assert sensevoice["input_manifest"].endswith("cleaned_mt3000.parquet")
    assert sensevoice["output_manifest"].endswith("sensevoice_asr_mt3000.parquet")

    Manifest([Sample(id="a", source_path="/tmp/a.wav")]).save(
        manifests / "qwen_asr_mt3000.parquet"
    )
    aggregate_step = _Step(
        "quality.aggregate_manifests",
        {"manifests": [{"model": "sensevoice", "path": "placeholder.parquet"}]},
    )
    aggregate = apply_source_name_to_single_pipeline(
        pipeline_name="multi_asr_aggregate",
        steps=[aggregate_step],
        source_name="mt3000",
    )
    assert aggregate["input_manifest"].endswith("qwen_asr_mt3000.parquet")
    assert aggregate_step.params["manifests"][0]["path"].endswith(
        "sensevoice_asr_mt3000.parquet"
    )

    Manifest([Sample(id="a", source_path="/tmp/a.wav")]).save(
        manifests / "multi_asr_aggregate_mt3000.parquet"
    )
    metrics = apply_source_name_to_single_pipeline(
        pipeline_name="asr_metric_pipeline",
        steps=[_Step("quality.text_metrics")],
        source_name="mt3000",
    )
    assert metrics["input_manifest"].endswith("multi_asr_aggregate_mt3000.parquet")
    assert metrics["output_manifest"].endswith("multi_asr_metrics_mt3000.parquet")


def test_run_staged_with_source_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from audio_engine.core.sharded_run import run_staged_pipelines

    monkeypatch.chdir(tmp_path)
    script = tmp_path / "tag.py"
    script.write_text(
        "def process(sample, params, context):\n"
        "    return {'labels': {'staged_ok': True}}\n",
        encoding="utf-8",
    )

    manifests = tmp_path / "datasets" / "manifests"
    manifests.mkdir(parents=True)
    cleaned = manifests / "cleaned_mt3000.parquet"
    Manifest(
        [
            Sample(id="a", source_path="/tmp/a.wav", duration=1.0),
            Sample(id="b", source_path="/tmp/b.wav", duration=2.0),
        ]
    ).save(cleaned)

    def _write_stage(name: str) -> Path:
        path = tmp_path / "pipelines" / f"{name}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        # Placeholder paths; --source-name overrides them.
        path.write_text(
            yaml.dump(
                {
                    "name": name,
                    "input": {"manifest": str(manifests / "placeholder.parquet")},
                    "output": {"manifest": str(manifests / f"{name}_out.parquet")},
                    "runs_dir": str(tmp_path / "runs"),
                    "execution": {"checkpoint_every": 0},
                    "sharding": {"shards": 2, "parallel_shards": 2, "strategy": "hash"},
                    "pipeline": [
                        {
                            "name": "tag",
                            "operator": "script.python",
                            "params": {"path": str(script)},
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        # from_yaml requires existing input when loading; create placeholder.
        Manifest([Sample(id="x", source_path="/tmp/x.wav", duration=1.0)]).save(
            manifests / "placeholder.parquet"
        )
        return path

    _write_stage("stage_qwen")
    _write_stage("stage_sv")
    root = tmp_path / "pipelines" / "orchestrator.yaml"
    root.write_text(
        yaml.dump(
            {
                "name": "multi",
                "source_name_layout": [
                    {"input": "cleaned_{source_name}", "output": "qwen_asr_{source_name}"},
                    {
                        "input": "qwen_asr_{source_name}",
                        "output": "multi_asr_aggregate_{source_name}",
                    },
                ],
                "stages": [
                    "pipelines/stage_qwen.yaml",
                    "pipelines/stage_sv.yaml",
                ],
            }
        ),
        encoding="utf-8",
    )

    from audio_engine.core.sharded_run import load_stage_paths

    result = run_staged_pipelines(
        load_stage_paths(root),
        name="multi",
        runs_dir=tmp_path / "runs",
        source_name="mt3000",
        source_name_layout=[
            {"input": "cleaned_{source_name}", "output": "qwen_asr_{source_name}"},
            {
                "input": "qwen_asr_{source_name}",
                "output": "multi_asr_aggregate_{source_name}",
            },
        ],
    )
    assert result.final_manifest == "datasets/manifests/multi_asr_aggregate_mt3000.parquet"
    assert Path(result.final_manifest).exists()
    assert (manifests / "qwen_asr_mt3000.parquet").exists()
    loaded = Manifest.load(result.final_manifest)
    assert len(loaded) == 2
    assert all(s.labels.get("staged_ok") is True for s in loaded.samples)
