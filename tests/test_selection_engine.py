"""Unit tests for consensus selection_v1.1 engine."""

from __future__ import annotations

import re

import audio_engine.operators  # noqa: F401
from audio_engine.core.operator import OperatorConfig
from audio_engine.core.registry import OperatorRegistry
from audio_engine.core.sample import Sample
from audio_engine.core.selection_engine import (
    DEFAULT_NEGATIVE_PHRASES,
    DEFAULT_POSITIVE_PHRASES,
    SelectionConfig,
    _compile_phrase_pattern,
    classify_sample,
    medoid_index,
    pairwise_min_similarity,
    resolve_family,
    semantic_class,
)
from audio_engine.core.transcript_reconcile import character_similarity


def test_resolve_family_prefix_and_alias():
    families = {
        "qwen": ["qwen", "qwen1", "qwen2"],
        "sensevoice": ["sensevoice", "sensevoice1"],
    }
    assert resolve_family("qwen1", families) == "qwen"
    assert resolve_family("sensevoice2", families) == "sensevoice"
    assert resolve_family("kimi", families) == "kimi"


def test_semantic_negative_beats_positive():
    neg = _compile_phrase_pattern(DEFAULT_NEGATIVE_PHRASES)
    pos = _compile_phrase_pattern(DEFAULT_POSITIVE_PHRASES)
    assert semantic_class("不需要", negative_pattern=neg, positive_pattern=pos) == "negative"
    assert semantic_class("需要", negative_pattern=neg, positive_pattern=pos) == "positive"
    assert semantic_class("今天天气", negative_pattern=neg, positive_pattern=pos) == "none"


def test_similarity_and_medoid_deterministic():
    assert character_similarity("不需要", "不需要") == 1.0
    texts = ["我不需要", "我现在不需要", "我不需要", "我不用"]
    # medoid should prefer the repeated / central form
    idx = medoid_index(texts)
    assert texts[idx] == "我不需要"
    assert medoid_index(texts) == medoid_index(list(texts))  # stable
    assert pairwise_min_similarity(["你好", "你好"]) == 1.0


def test_noise_all_empty():
    sample = Sample(
        id="n1",
        source_path="n.wav",
        duration=1.0,
        transcripts={"qwen1": {"text": ""}, "sensevoice1": {"text": ""}},
    )
    result = classify_sample(sample, SelectionConfig())
    assert result.type == "noise"
    assert result.decision == "auto_empty"
    assert result.label == ""


def test_voicemail_requires_two_families():
    cfg = SelectionConfig()
    pattern = re.compile("语音信箱|无法接通")
    only_qwen = Sample(
        id="v1",
        source_path="v.wav",
        duration=2.0,
        transcripts={
            "qwen1": {"text": "您好您拨打的用户暂时无法接通"},
            "qwen2": {"text": "您好您拨打的用户暂时无法接通"},
            "sensevoice1": {"text": ""},
            "sensevoice2": {"text": ""},
        },
    )
    # Single family hit → not voicemail (falls through to hallucination/hardcase)
    solo = classify_sample(only_qwen, cfg, voicemail_pattern=pattern)
    assert solo.type != "voicemail"

    both = Sample(
        id="v2",
        source_path="v.wav",
        duration=2.0,
        transcripts={
            "qwen1": {"text": "您好这里是语音信箱请留言"},
            "sensevoice1": {"text": "您好这里是语音信箱请留言"},
        },
    )
    dual = classify_sample(both, cfg, voicemail_pattern=pattern)
    assert dual.type == "voicemail"
    assert dual.decision == "auto_accept"
    assert dual.label == "您好这里是语音信箱请留言"


def test_semantic_inversion_before_similarity():
    sample = Sample(
        id="si",
        source_path="si.wav",
        duration=2.0,
        transcripts={
            "qwen1": {"text": "我现在确实需要办理"},
            "qwen2": {"text": "我现在确实需要办理"},
            "sensevoice1": {"text": "我现在确实不需要办理"},
            "sensevoice2": {"text": "我现在确实不需要办理"},
        },
    )
    result = classify_sample(sample, SelectionConfig())
    assert result.type == "semantic_inversion"
    assert result.decision == "model_review"
    assert result.semantic_qwen == "positive"
    assert result.semantic_sensevoice == "negative"


def test_semantic_inversion_third_family_majority():
    sample = Sample(
        id="si3",
        source_path="si.wav",
        duration=2.0,
        transcripts={
            "qwen1": {"text": "需要"},
            "sensevoice1": {"text": "不需要"},
            "kimi1": {"text": "不需要"},
        },
    )
    result = classify_sample(sample, SelectionConfig())
    assert result.type == "semantic_inversion"
    assert result.decision == "auto_accept"
    assert result.label == "不需要"


def test_hallucination_empty_majority_single_family():
    sample = Sample(
        id="h1",
        source_path="h.wav",
        duration=2.0,
        transcripts={
            "qwen1": {"text": "不需要"},
            "qwen2": {"text": ""},
            "sensevoice1": {"text": ""},
            "sensevoice2": {"text": ""},
        },
    )
    result = classify_sample(sample, SelectionConfig())
    assert result.type == "hallucination"
    assert result.decision == "model_review"


def test_qwen_missing():
    sample = Sample(
        id="qm",
        source_path="qm.wav",
        duration=2.0,
        transcripts={
            "qwen1": {"text": ""},
            "qwen2": {"text": ""},
            "sensevoice1": {"text": "不需要"},
            "sensevoice2": {"text": "不需要"},
        },
    )
    result = classify_sample(sample, SelectionConfig())
    assert result.type == "qwen_missing"
    assert result.decision == "model_review"


def test_auto_gold_uses_medoid_not_random():
    sample = Sample(
        id="ag",
        source_path="ag.wav",
        duration=2.0,
        transcripts={
            "qwen1": {"text": "我不需要"},
            "qwen2": {"text": "我不需要"},
            "sensevoice1": {"text": "我不需要"},
            "sensevoice2": {"text": "我不需要"},
        },
    )
    result = classify_sample(sample, SelectionConfig())
    assert result.type == "auto_gold"
    assert result.decision == "auto_accept"
    assert result.label == "我不需要"
    assert result.selected_model in {"qwen1", "qwen2", "sensevoice1", "sensevoice2"}


def test_consensus_gold_dominant_cluster():
    sample = Sample(
        id="cg",
        source_path="cg.wav",
        duration=2.0,
        transcripts={
            "qwen1": {"text": "不需要"},
            "qwen2": {"text": "不需要"},
            "sensevoice1": {"text": "不需要"},
            "sensevoice2": {"text": "不需要办理这个"},
        },
    )
    result = classify_sample(sample, SelectionConfig())
    assert result.type == "consensus_gold"
    assert result.decision == "auto_accept"
    assert result.label == "不需要"
    assert result.support_count >= 3
    assert result.support_family_count >= 2


def test_hardcase_no_consensus():
    sample = Sample(
        id="hc",
        source_path="hc.wav",
        duration=2.0,
        transcripts={
            "qwen1": {"text": "我要办理"},
            "qwen2": {"text": "我要了解一下"},
            "sensevoice1": {"text": "我不办理"},
            "sensevoice2": {"text": "我知道了"},
        },
    )
    result = classify_sample(sample, SelectionConfig())
    assert result.type == "hardcase"
    assert result.decision == "model_review"


def test_classify_operator_writes_decision_fields():
    samples = [
        Sample(
            id="agree",
            source_path="a.wav",
            duration=2.0,
            transcripts={
                "qwen1": {"text": "你好世界"},
                "sensevoice1": {"text": "你好世界"},
            },
        )
    ]
    result = OperatorRegistry.get("quality.classify").run(
        samples,
        OperatorConfig(params={"config_path": "configs/selection/zh_asr_v1.yaml"}),
    )
    sample = result[0]
    assert sample.labels["type"] == "auto_gold"
    assert sample.labels["decision"] == "auto_accept"
    assert sample.labels["label"] == "你好世界"
    assert sample.labels["rule_version"] == "selection_v1.1"
    assert sample.labels["selection_policy_version"] == "selection_zh_asr_v1_6"
    assert sample.labels["annotation_state"] == "auto_accepted"
