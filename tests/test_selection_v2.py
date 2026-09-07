"""Unit tests for selection_v2.0 / consensus_v2 engine."""

from __future__ import annotations

from pathlib import Path

import audio_engine.operators  # noqa: F401
from audio_engine.core.operator import OperatorConfig
from audio_engine.core.registry import OperatorRegistry
from audio_engine.core.sample import Sample
from audio_engine.core.selection_v2 import classify_sample
from audio_engine.core.selection_v2.config import SelectionV2Config as Cfg
from audio_engine.core.eval_ready import inspect_eval_manifest, is_formal_gold_sample
from audio_engine.core.manifest import Manifest


def _cfg(**kwargs) -> Cfg:
    root = Path(__file__).resolve().parents[1]
    params = {
        "rule_version": "selection_v2.0",
        "semantic_lexicon_path": str(root / "configs/selection/semantic_lexicon_zh_v2.yaml"),
        "similarity": {
            "strict_threshold": 0.95,
            "consensus_threshold": 0.90,
            "dominant_cluster_ratio": 0.75,
        },
        "short_utterance": {"max_audio_sec": 2.0, "max_text_chars": 6},
        "silence": {"max_speech_ratio": 0.05},
        "risk_gate": {
            "semantic_inversion_auto_accept": False,
            "vad_miss_auto_empty": False,
        },
        "model_families": {
            "qwen": ["qwen", "qwen1", "qwen2"],
            "sensevoice": ["sensevoice", "sensevoice1", "sensevoice2"],
        },
        "primary_family": "qwen",
        "secondary_family": "sensevoice",
    }
    params.update(kwargs)
    return Cfg.from_params(params)


def test_invalid_audio_exclude():
    sample = Sample(
        id="bad",
        source_path="bad.wav",
        duration=0.0,
        labels={"broken": True},
        transcripts={"qwen1": {"text": "你好"}, "sensevoice1": {"text": "你好"}},
    )
    result = classify_sample(sample, _cfg())
    assert result.type == "invalid_audio"
    assert result.decision == "exclude"
    assert result.label_tier == "none"


def test_all_empty_without_speech_evidence_is_vad_miss():
    sample = Sample(
        id="e1",
        source_path="e.wav",
        duration=1.5,
        transcripts={"qwen1": {"text": ""}, "sensevoice1": {"text": ""}},
    )
    result = classify_sample(sample, _cfg())
    assert result.type == "possible_vad_miss"
    assert result.decision == "model_review"


def test_all_empty_with_weak_speech_is_true_silence():
    sample = Sample(
        id="e2",
        source_path="e.wav",
        duration=1.5,
        labels={"speech_ratio": 0.01, "has_speech": False},
        transcripts={"qwen1": {"text": ""}, "sensevoice1": {"text": ""}},
    )
    result = classify_sample(sample, _cfg())
    assert result.type == "true_silence"
    assert result.decision == "auto_empty"


def test_semantic_inversion_never_auto_accept():
    sample = Sample(
        id="s1",
        source_path="s.wav",
        duration=3.0,
        transcripts={
            "qwen1": {"text": "不需要"},
            "qwen2": {"text": "不需要"},
            "sensevoice1": {"text": "需要"},
            "sensevoice2": {"text": "需要"},
            "doubao1": {"text": "不需要"},
        },
    )
    cfg = _cfg(
        model_families={
            "qwen": ["qwen", "qwen1", "qwen2"],
            "sensevoice": ["sensevoice", "sensevoice1", "sensevoice2"],
            "doubao": ["doubao", "doubao1"],
        }
    )
    result = classify_sample(sample, cfg)
    assert result.type == "semantic_inversion"
    assert result.decision == "model_review"
    assert result.semantic_risk is True


def test_critical_token_conflict_before_similarity():
    sample = Sample(
        id="c1",
        source_path="c.wav",
        duration=3.0,
        transcripts={
            "qwen1": {"text": "我现在不需要贷款"},
            "sensevoice1": {"text": "我现在需要贷款"},
        },
    )
    result = classify_sample(sample, _cfg())
    assert result.type in {"semantic_inversion", "critical_token_conflict"}
    assert result.decision == "model_review"


def test_short_utterance_blocks_pseudo_high():
    sample = Sample(
        id="sh1",
        source_path="sh.wav",
        duration=1.0,
        transcripts={
            "qwen1": {"text": "哈哈"},
            "sensevoice1": {"text": "呵呵"},
        },
    )
    result = classify_sample(sample, _cfg())
    assert result.type == "short_utterance_risk"
    assert result.decision == "model_review"
    assert result.short_utterance is True


def test_pseudo_gold_high_and_fields():
    sample = Sample(
        id="g1",
        source_path="g.wav",
        duration=4.0,
        transcripts={
            "qwen1": {"text": "今天天气不错"},
            "sensevoice1": {"text": "今天天气不错"},
        },
    )
    result = classify_sample(sample, _cfg())
    assert result.type == "pseudo_gold_high"
    assert result.decision == "auto_accept"
    assert result.label_tier == "pseudo_high"
    assert result.label_source == "model_consensus"
    assert result.is_human_verified is False
    assert result.label == "今天天气不错"
    labels = result.to_labels("selection_zh_asr_v2_0")
    assert labels["label_tier"] == "pseudo_high"
    assert labels["gold_text"] == "今天天气不错"


def test_model_missing_subtype_qwen():
    sample = Sample(
        id="m1",
        source_path="m.wav",
        duration=3.0,
        transcripts={
            "qwen1": {"text": ""},
            "qwen2": {"text": ""},
            "sensevoice1": {"text": "你好请问有什么可以帮您"},
        },
    )
    result = classify_sample(sample, _cfg())
    assert result.type == "model_missing"
    assert result.subtype == "qwen_missing"
    assert result.decision == "model_review"


def test_classify_operator_v2_config(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    op = OperatorRegistry.get("quality.classify")
    samples = [
        Sample(
            id="g1",
            source_path="g.wav",
            duration=4.0,
            transcripts={
                "qwen1": {"text": "今天天气不错"},
                "sensevoice1": {"text": "今天天气不错"},
            },
        )
    ]
    result = op.run(
        samples,
        OperatorConfig(
            name="classify",
            operator="quality.classify",
            params={"config_path": str(root / "configs/selection/zh_asr_v2.yaml")},
        ),
    )
    assert result[0].labels["type"] == "pseudo_gold_high"
    assert result[0].labels["rule_version"] == "selection_v2.0"


def test_formal_eval_rejects_pseudo_gold(tmp_path: Path):
    path = tmp_path / "pseudo.parquet"
    Manifest(
        [
            Sample(
                id="a",
                source_path="a.wav",
                sha256="a" * 64,
                audio={"resampled_16k": "a.wav"},
                labels={
                    "gold_text": "你好",
                    "type": "pseudo_gold_high",
                    "classification_bucket": "pseudo_gold_high",
                    "label_tier": "pseudo_high",
                    "label_source": "model_consensus",
                },
            )
        ]
    ).save(path)
    report = inspect_eval_manifest(path, require_formal_gold=True)
    assert report.errors
    assert any("pseudo" in e.lower() for e in report.errors)


def test_formal_eval_accepts_human_and_external(tmp_path: Path):
    path = tmp_path / "formal.parquet"
    Manifest(
        [
            Sample(
                id="h1",
                source_path="h.wav",
                sha256="h" * 64,
                audio={"resampled_16k": "h.wav"},
                labels={
                    "gold_text": "不需要",
                    "type": "human_gold",
                    "annotation_state": "human_accepted",
                    "label_source": "human",
                    "label_tier": "gold",
                    "is_human_verified": True,
                },
            ),
            Sample(
                id="e1",
                source_path="e.wav",
                sha256="e" * 64,
                audio={"resampled_16k": "e.wav"},
                labels={
                    "gold_text": "需要",
                    "type": "auto_gold",
                    "gold_source": "external",
                    "gold_mode": "external",
                    "label_source": "external",
                    "label_tier": "gold",
                },
            ),
        ]
    ).save(path)
    assert is_formal_gold_sample(Manifest.load(path).samples[0])
    report = inspect_eval_manifest(path, require_formal_gold=True)
    assert not report.errors


def test_train_eval_leak_check(tmp_path: Path):
    train = tmp_path / "train.parquet"
    eval_path = tmp_path / "eval.parquet"
    Manifest(
        [
            Sample(
                id="shared",
                source_path="s.wav",
                sha256="s" * 64,
                labels={"duplicate_group_id": "dup1", "gold_text": "a", "annotation_state": "human_accepted"},
                audio={"resampled_16k": "s.wav"},
            )
        ]
    ).save(train)
    Manifest(
        [
            Sample(
                id="shared",
                source_path="s.wav",
                sha256="s" * 64,
                labels={
                    "duplicate_group_id": "dup1",
                    "gold_text": "a",
                    "annotation_state": "human_accepted",
                    "type": "human_gold",
                },
                audio={"resampled_16k": "s.wav"},
            )
        ]
    ).save(eval_path)
    report = inspect_eval_manifest(
        eval_path, require_formal_gold=True, train_manifest=train
    )
    assert any("leakage" in e for e in report.errors)
