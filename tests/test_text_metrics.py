from pathlib import Path

import pytest

from audio_engine.core.manifest import Manifest
from audio_engine.core.operator import OperatorConfig
from audio_engine.core.registry import OperatorRegistry
from audio_engine.core.sample import Sample
from audio_engine.metrics.align import align_characters
from audio_engine.metrics.cer import calculate_cer
from audio_engine.metrics.normalization import normalize_text
from audio_engine.metrics.runner import MetricConfigError, run_text_metrics


def test_cer_operations_and_empty_reference():
    assert calculate_cer("需要", "需要")["cer"] == 0
    assert calculate_cer("不需要", "需要")["deletions"] == 1
    assert calculate_cer("需要", "啊需要")["insertions"] == 1
    assert calculate_cer("", "")["cer"] == 0
    assert calculate_cer("", "需要") == {
        "cer": 2.0,
        "substitutions": 0,
        "deletions": 0,
        "insertions": 2,
        "reference_length": 0,
    }


def test_align_characters_substitution_and_deletion():
    ops = align_characters("不需要", "需要")
    assert any(item["operation"] == "deletion" and item["reference"] == "不" for item in ops)
    ops = align_characters("贷款", "代款")
    assert any(item["operation"] == "substitution" for item in ops)


def test_normalization_preserves_fillers():
    config = {
        "punctuation": {"remove": True},
        "whitespace": {"remove": True},
        "filler": {"remove": False},
    }
    assert normalize_text("我不需要，贷款。", config) == "我不需要贷款"
    assert normalize_text("嗯 啊", config) == "嗯啊"


def test_runner_validates_fields_and_collisions():
    comparison = {
        "reference": {"field": "gold_text"},
        "hypothesis": {"field": "model_text"},
        "metrics": ["cer"],
        "output": {"prefix": "model"},
    }
    assert (
        run_text_metrics({"gold_text": "需要", "model_text": "需要"}, comparison, {})["model_cer"]
        == 0
    )
    with pytest.raises(MetricConfigError, match="not found"):
        run_text_metrics({"gold_text": "需要"}, comparison, {})
    with pytest.raises(MetricConfigError, match="already exist"):
        run_text_metrics(
            {"gold_text": "需要", "model_text": "需要", "model_cer": 1}, comparison, {}
        )


def test_runner_reads_gold_provenance_fields_from_labels():
    comparison = {
        "purpose": "model_evaluation",
        "reference": {"field": "gold_text"},
        "hypothesis": {"field": "old_model_text"},
        "metrics": ["cer"],
        "output": {"prefix": "old_model"},
    }
    record = {
        "gold_text": "不需要",
        "gold_source": "human",
        "gold_status": "verified",
        "gold_version": "gold_v1",
        "old_model_text": "需要",
    }
    out = run_text_metrics(record, comparison, {})
    assert out["old_model_cer"] == pytest.approx(1 / 3, rel=1e-4)


def test_aggregate_requires_aligned_ids(tmp_path: Path):
    other = tmp_path / "sense.parquet"
    Manifest(
        [Sample(id="a", source_path="a.wav", transcripts={"sensevoice": {"text": "需要"}})]
    ).save(other)
    operator = OperatorRegistry.get("quality.aggregate_manifests")
    result = operator.run(
        [Sample(id="a", source_path="a.wav", transcripts={"qwen": {"text": "需要"}})],
        OperatorConfig(params={"manifests": [{"model": "sensevoice", "path": str(other)}]}),
    )
    assert result[0].get_transcript_text("sensevoice") == "需要"
    assert result[0].transcripts["sensevoice"]["extra"]["source_manifest"] == str(other)
    with pytest.raises(ValueError, match="not aligned"):
        operator.run(
            [Sample(id="b", source_path="b.wav")],
            OperatorConfig(params={"manifests": [{"model": "sensevoice", "path": str(other)}]}),
        )
