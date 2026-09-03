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
            OperatorConfig(
                params={"manifests": [{"model": "sensevoice", "path": str(other)}]},
                run_dir=tmp_path / "run-missing",
                step_name="aggregate",
            ),
        )
    assert (tmp_path / "run-missing/reports/aggregate_alignment.json").exists()


def test_aggregate_rejects_same_id_with_different_audio_hash(tmp_path: Path):
    other = tmp_path / "sense.parquet"
    Manifest(
        [
            Sample(
                id="same",
                source_path="incoming.wav",
                sha256="b" * 64,
                transcripts={"sensevoice": {"text": "需要"}},
            )
        ]
    ).save(other)
    operator = OperatorRegistry.get("quality.aggregate_manifests")
    config = OperatorConfig(
        params={"manifests": [{"model": "sensevoice", "path": str(other)}]},
        run_dir=tmp_path / "run-hash",
        step_name="aggregate",
    )

    with pytest.raises(ValueError, match="audio hashes are not aligned"):
        operator.run(
            [Sample(id="same", source_path="base.wav", sha256="a" * 64)],
            config,
        )

    report = (tmp_path / "run-hash/reports/aggregate_alignment.json").read_text(encoding="utf-8")
    assert '"sha256_mismatches": 1' in report
    assert '"aligned": false' in report


def test_text_metrics_vs_base_expands_and_tracks_max(tmp_path: Path):
    config_path = tmp_path / "gold_agreement.yaml"
    config_path.write_text(
        "vs_base:\n  base: qwen1\n  purpose: gold_generation_agreement\n  metrics: [cer]\n",
        encoding="utf-8",
    )
    norm = tmp_path / "norm.yaml"
    norm.write_text("name: test\npunctuation: {remove: true}\nwhitespace: {remove: true}\n", encoding="utf-8")
    sample = Sample(
        id="a",
        source_path="a.wav",
        transcripts={
            "qwen1": {"text": "不需要"},
            "qwen2": {"text": "不需要"},
            "sensevoice1": {"text": "需要"},
        },
    )
    result = OperatorRegistry.get("quality.text_metrics").process(
        sample,
        OperatorConfig(
            params={
                "config_path": str(config_path),
                "normalization_path": str(norm),
            }
        ),
    )
    quality = result.sample.quality
    assert quality["vs_base_reference"] == "qwen1"
    assert quality["vs_base_hypothesis_count"] == 2
    assert quality["qwen2_vs_qwen1_agreement_cer"] == 0
    assert quality["sensevoice1_vs_qwen1_agreement_cer"] == pytest.approx(1 / 3, rel=1e-4)
    assert quality["max_vs_base_agreement_cer"] == pytest.approx(1 / 3, rel=1e-4)


def test_text_metrics_agreement_base_override(tmp_path: Path):
    config_path = tmp_path / "gold_agreement.yaml"
    config_path.write_text(
        "vs_base:\n  base: ignored\n  purpose: gold_generation_agreement\n  metrics: [cer]\n",
        encoding="utf-8",
    )
    norm = tmp_path / "norm.yaml"
    norm.write_text("name: test\n", encoding="utf-8")
    sample = Sample(
        id="a",
        source_path="a.wav",
        transcripts={"qwen1": {"text": "需要"}, "doubao1": {"text": "需要"}},
    )
    result = OperatorRegistry.get("quality.text_metrics").process(
        sample,
        OperatorConfig(
            params={
                "config_path": str(config_path),
                "normalization_path": str(norm),
                "agreement_base": "qwen1",
            }
        ),
    )
    assert result.sample.quality["vs_base_reference"] == "qwen1"
    assert result.sample.quality["max_vs_base_agreement_cer"] == 0
    assert result.sample.quality["doubao1_vs_qwen1_agreement_cer"] == 0
