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
    model_asr_kind,
    parse_join_manifest_arg,
    pipeline_run_name,
    resolve_existing_manifest,
    rewrite_join_manifests_for_source,
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
    assert model_asr_kind("sensevoice") == "sensevoice_asr"
    assert model_asr_kind("kimi_asr") == "kimi_asr"


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
        ("cleaned_mt3000", "datasets/manifests/sensevoice_asr_mt3000.parquet"),
    ]


def test_apply_cleaning_qwen_sensevoice_metric(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    wav_dir = tmp_path / "wav"
    wav_dir.mkdir()
    manifests = tmp_path / "datasets" / "manifests"
    manifests.mkdir(parents=True)
    for stem in (
        "cleaned_mt3000",
        "qwen_asr_mt3000",
        "sensevoice_asr_mt3000",
        "multi_asr_aggregate_mt3000",
    ):
        Manifest([Sample(id="a", source_path="/tmp/a.wav", duration=1.0)]).save(
            manifests / f"{stem}.parquet"
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

    sense = apply_source_name_to_single_pipeline(
        pipeline_name="sensevoice_asr_batch",
        steps=[_Step("asr.sensevoice_batch")],
        source_name="mt3000",
    )
    assert sense["input_manifest"].endswith("cleaned_mt3000.parquet")
    assert sense["output_manifest"].endswith("sensevoice_asr_mt3000.parquet")

    kimi = apply_source_name_to_single_pipeline(
        pipeline_name="kimi_asr_batch",
        steps=[_Step("asr.kimi_batch")],
        source_name="mt3000",
    )
    assert kimi["output_manifest"].endswith("kimi_asr_mt3000.parquet")

    kimi_local = apply_source_name_to_single_pipeline(
        pipeline_name="kimi_audio_asr_batch",
        steps=[_Step("asr.kimi_audio_batch")],
        source_name="mt3000",
    )
    assert kimi_local["output_manifest"].endswith("kimi_audio_asr_mt3000.parquet")

    aggregate = apply_source_name_to_single_pipeline(
        pipeline_name="multi_asr_aggregate",
        steps=[
            _Step(
                "quality.aggregate_manifests",
                {
                    "manifests": [
                        {
                            "model": "sensevoice",
                            "path": "datasets/manifests/sensevoice_asr_source_A.parquet",
                        }
                    ]
                },
            )
        ],
        source_name="mt3000",
    )
    assert aggregate["input_manifest"].endswith("qwen_asr_mt3000.parquet")
    assert aggregate["output_manifest"].endswith("multi_asr_aggregate_mt3000.parquet")
    assert aggregate["aggregate_manifests"][0]["model"] == "sensevoice"
    assert aggregate["aggregate_manifests"][0]["path"].endswith(
        "sensevoice_asr_mt3000.parquet"
    )

    metric = apply_source_name_to_single_pipeline(
        pipeline_name="asr_metric_pipeline",
        steps=[_Step("quality.text_metrics")],
        source_name="mt3000",
    )
    assert metric["input_manifest"].endswith("multi_asr_aggregate_mt3000.parquet")
    assert metric["output_manifest"].endswith("multi_asr_metrics_mt3000.parquet")


def test_parse_and_rewrite_join_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    manifests = tmp_path / "datasets" / "manifests"
    manifests.mkdir(parents=True)
    Manifest([Sample(id="a", source_path="/tmp/a.wav", duration=1.0)]).save(
        manifests / "sensevoice_asr_mt3000.parquet"
    )
    Manifest([Sample(id="a", source_path="/tmp/a.wav", duration=1.0)]).save(
        manifests / "kimi_asr_mt3000.parquet"
    )

    bare = parse_join_manifest_arg("sensevoice", "mt3000")
    assert bare["model"] == "sensevoice"
    assert bare["path"].endswith("sensevoice_asr_mt3000.parquet")

    rewritten = rewrite_join_manifests_for_source(
        [{"model": "sensevoice", "path": "ignored.parquet"}], "mt3000"
    )
    assert rewritten[0]["path"].endswith("sensevoice_asr_mt3000.parquet")

    explicit = parse_join_manifest_arg(
        f"kimi={manifests / 'kimi_asr_mt3000.parquet'}", None
    )
    assert explicit["model"] == "kimi"


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
                        "input": "cleaned_{source_name}",
                        "output": "sensevoice_asr_{source_name}",
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
                "input": "cleaned_{source_name}",
                "output": "sensevoice_asr_{source_name}",
            },
        ],
    )
    assert result.final_manifest == "datasets/manifests/sensevoice_asr_mt3000.parquet"
    assert Path(result.final_manifest).exists()
    assert (manifests / "qwen_asr_mt3000.parquet").exists()
    loaded = Manifest.load(result.final_manifest)
    assert len(loaded) == 2
    assert all(s.labels.get("staged_ok") is True for s in loaded.samples)
