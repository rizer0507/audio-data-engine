"""Decoupled evaluation: register eval set → infer by alias → join by id → vs gold."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import audio_engine.operators  # noqa: F401
from audio_engine.cli.main import app
from audio_engine.core.manifest import Manifest
from audio_engine.core.operator import OperatorConfig
from audio_engine.core.registry import OperatorRegistry
from audio_engine.core.sample import Sample
from audio_engine.core.source_naming import apply_eval_name_to_single_pipeline


def _eval_sample(
    sample_id: str,
    *,
    gold: str,
    bucket: str,
    transcripts: dict[str, str] | None = None,
) -> Sample:
    return Sample(
        id=sample_id,
        source_path=f"{sample_id}.wav",
        sha256=(sample_id.encode().hex() + "0" * 64)[:64],
        audio={"resampled_16k": f"{sample_id}.wav"},
        labels={
            "gold_text": gold,
            "label": gold,
            "type": bucket,
            "classification_bucket": bucket,
        },
        transcripts={
            key: {"text": text} for key, text in (transcripts or {}).items()
        },
    )


def test_apply_eval_name_asr_aggregate_metric(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    manifests = tmp_path / "datasets" / "manifests"
    manifests.mkdir(parents=True)
    Manifest([_eval_sample("a", gold="你好", bucket="auto_gold")]).save(
        manifests / "eval_local_test.parquet"
    )
    Manifest(
        [
            Sample(
                id="a",
                source_path="a.wav",
                transcripts={"qwen-sft-epoch10": {"text": "你好"}},
            )
        ]
    ).save(manifests / "qwen-sft-epoch10_asr_eval_local_test.parquet")
    Manifest([_eval_sample("a", gold="你好", bucket="auto_gold")]).save(
        manifests / "eval_aggregate_eval_local_test.parquet"
    )

    class _Step:
        def __init__(self, operator: str, params: dict | None = None):
            self.operator = operator
            self.params = params or {}

    asr = apply_eval_name_to_single_pipeline(
        pipeline_name="qwen_asr_batch",
        steps=[_Step("asr.qwen_batch")],
        eval_name="eval_local_test",
        asr_run="qwen-sft-epoch10",
    )
    assert asr["input_manifest"].endswith("eval_local_test.parquet")
    assert asr["output_manifest"].endswith("qwen-sft-epoch10_asr_eval_local_test.parquet")
    assert asr["asr_run"] == "qwen-sft-epoch10"

    aggregate = apply_eval_name_to_single_pipeline(
        pipeline_name="eval_aggregate",
        steps=[_Step("quality.aggregate_manifests", {"manifests": []})],
        eval_name="eval_local_test",
        join_manifests=[
            {
                "model": "qwen-sft-epoch10",
                "path": "datasets/manifests/qwen-sft-epoch10_asr_eval_local_test.parquet",
            }
        ],
    )
    assert aggregate["input_manifest"].endswith("eval_local_test.parquet")
    assert aggregate["output_manifest"].endswith("eval_aggregate_eval_local_test.parquet")
    assert aggregate["aggregate_manifests"][0]["model"] == "qwen-sft-epoch10"

    metric = apply_eval_name_to_single_pipeline(
        pipeline_name="eval_metric_pipeline",
        steps=[_Step("quality.text_metrics")],
        eval_name="eval_local_test",
    )
    assert metric["input_manifest"].endswith("eval_aggregate_eval_local_test.parquet")
    assert metric["output_manifest"].endswith("eval_metrics_eval_local_test.parquet")


def test_aggregate_left_allows_extra_ids(tmp_path: Path):
    other = tmp_path / "sft.parquet"
    Manifest(
        [
            Sample(
                id="a",
                source_path="a.wav",
                sha256="a" * 64,
                transcripts={"qwen-sft-epoch10": {"text": "你好"}},
            ),
            Sample(
                id="extra",
                source_path="extra.wav",
                sha256="e" * 64,
                transcripts={"qwen-sft-epoch10": {"text": "多余"}},
            ),
        ]
    ).save(other)
    operator = OperatorRegistry.get("quality.aggregate_manifests")
    result = operator.run(
        [Sample(id="a", source_path="a.wav", sha256="a" * 64)],
        OperatorConfig(
            params={
                "id_policy": "left",
                "manifests": [{"model": "qwen-sft-epoch10", "path": str(other)}],
            },
            run_dir=tmp_path / "run",
            step_name="aggregate",
        ),
    )
    assert result[0].get_transcript_text("qwen-sft-epoch10") == "你好"
    report = json.loads(
        (tmp_path / "run/reports/aggregate_alignment.json").read_text(encoding="utf-8")
    )
    assert report["id_policy"] == "left"
    assert report["aligned"] is True
    assert report["manifests"][0]["extra_ids"] == 1


def test_text_metrics_vs_gold_expands(tmp_path: Path):
    config_path = tmp_path / "model_eval.yaml"
    config_path.write_text(
        "vs_gold:\n  purpose: model_evaluation\n  metrics: [cer]\n",
        encoding="utf-8",
    )
    norm = tmp_path / "norm.yaml"
    norm.write_text("name: test\n", encoding="utf-8")
    sample = Sample(
        id="a",
        source_path="a.wav",
        labels={"gold_text": "不需要"},
        transcripts={
            "qwen": {"text": "需要"},
            "qwen-sft-epoch10": {"text": "不需要"},
        },
    )
    result = OperatorRegistry.get("quality.text_metrics").process(
        sample,
        OperatorConfig(
            params={
                "config_path": str(config_path),
                "normalization_path": str(norm),
                "eval_hypotheses": ["qwen", "qwen-sft-epoch10"],
            }
        ),
    )
    quality = result.sample.quality
    assert quality["vs_gold_hypotheses"] == ["qwen", "qwen-sft-epoch10"]
    assert quality["qwen_cer"] == pytest.approx(1 / 3, rel=1e-4)
    assert quality["qwen-sft-epoch10_cer"] == 0


def test_evaluation_report_multi_model_by_type(tmp_path: Path):
    samples = [
        Sample(
            id="a",
            source_path="a.wav",
            labels={"type": "auto_gold", "gold_text": "你好"},
            transcripts={
                "qwen": {"text": "你好"},
                "qwen-sft-epoch10": {"text": "你好"},
            },
            quality={
                "qwen_substitutions": 0,
                "qwen_deletions": 0,
                "qwen_insertions": 0,
                "qwen_reference_length": 2,
                "qwen_cer": 0.0,
                "qwen-sft-epoch10_substitutions": 0,
                "qwen-sft-epoch10_deletions": 0,
                "qwen-sft-epoch10_insertions": 0,
                "qwen-sft-epoch10_reference_length": 2,
                "qwen-sft-epoch10_cer": 0.0,
            },
        ),
        Sample(
            id="b",
            source_path="b.wav",
            labels={"type": "hardcase", "gold_text": "不需要"},
            transcripts={
                "qwen": {"text": "需要"},
                "qwen-sft-epoch10": {"text": "不需要"},
            },
            quality={
                "qwen_substitutions": 0,
                "qwen_deletions": 1,
                "qwen_insertions": 0,
                "qwen_reference_length": 3,
                "qwen_cer": 1 / 3,
                "qwen-sft-epoch10_substitutions": 0,
                "qwen-sft-epoch10_deletions": 0,
                "qwen-sft-epoch10_insertions": 0,
                "qwen-sft-epoch10_reference_length": 3,
                "qwen-sft-epoch10_cer": 0.0,
            },
        ),
    ]
    operator = OperatorRegistry.get("quality.evaluation_report")
    operator.run(
        samples,
        OperatorConfig(
            run_dir=tmp_path / "run",
            params={
                "model_prefixes": ["qwen", "qwen-sft-epoch10"],
                "bucket_key": "type",
                "fail_on_regression": False,
            },
        ),
    )
    report = json.loads((tmp_path / "run/reports/evaluation.json").read_text(encoding="utf-8"))
    assert report["model_prefixes"] == ["qwen", "qwen-sft-epoch10"]
    assert report["overall"]["qwen"]["corpus_cer"] == pytest.approx(0.2)
    assert report["overall"]["qwen-sft-epoch10"]["corpus_cer"] == 0.0
    assert "hardcase" in report["buckets"]
    assert report["passed"] is True
    frame = __import__("pandas").read_excel(
        tmp_path / "run/reports/evaluation.xlsx", sheet_name="统计摘要"
    )
    assert "qwen_text ← label" in set(frame["对比"])
    assert "qwen-sft-epoch10_text ← label" in set(frame["对比"])
    assert "相对base总体字准率" in frame.columns
    assert "相对base平均字准率" in frame.columns
    base_rows = frame[frame["对比"] == "qwen_text ← label"]
    assert base_rows["相对base总体字准率"].isna().all()
    sft = frame[(frame["对比"] == "qwen-sft-epoch10_text ← label") & (frame["type"] == "总计")].iloc[0]
    assert sft["相对base总体字准率"] == pytest.approx(0.2)  # 1.0 - 0.8
    assert sft["相对base平均字准率"] == pytest.approx(1.0 / 6.0, abs=1e-6)


def test_eval_register_and_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "datasets" / "manifests").mkdir(parents=True)
    (tmp_path / "data" / "catalog").mkdir(parents=True)
    (tmp_path / "runs").mkdir()
    source = tmp_path / "summary.parquet"
    Manifest(
        [
            _eval_sample("a", gold="你好", bucket="auto_gold", transcripts={"qwen": "你好"}),
            _eval_sample("b", gold="", bucket="noise", transcripts={"qwen": ""}),
        ]
    ).save(source)

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["eval", "register", str(source), "--name", "eval_local_test"],
    )
    assert result.exit_code == 0, result.output
    dest = tmp_path / "datasets" / "manifests" / "eval_local_test.parquet"
    assert dest.exists()
    check = runner.invoke(app, ["eval", "check", "eval_local_test"])
    assert check.exit_code == 0, check.output
    assert "with_gold" in check.output
