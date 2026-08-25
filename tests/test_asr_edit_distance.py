from __future__ import annotations

from audio_engine.core.operator import OperatorConfig
from audio_engine.core.registry import OperatorRegistry
from audio_engine.core.sample import Sample
from audio_engine.core.transcript_reconcile import levenshtein_ops
import audio_engine.operators  # noqa: F401


def test_levenshtein_ops_counts_sub_del_ins():
    # baseline "abcd" vs hyp "abXd" → 1 substitution
    ops = levenshtein_ops("abcd", "abXd")
    assert ops["total"] == 1
    assert ops["错字"] == 1
    assert ops["少字"] == 0
    assert ops["多字"] == 0

    # baseline "你好世界" vs hyp "你好" → 2 deletions (少字)
    ops = levenshtein_ops("你好世界", "你好")
    assert ops["少字"] == 2
    assert ops["多字"] == 0
    assert ops["错字"] == 0
    assert ops["total"] == 2

    # baseline "你好" vs hyp "你好啊啊" → 2 insertions (多字)
    ops = levenshtein_ops("你好", "你好啊啊")
    assert ops["多字"] == 2
    assert ops["少字"] == 0
    assert ops["total"] == 2


def test_asr_edit_distance_operator_vs_baseline():
    sample = Sample(
        id="s1",
        source_path="/tmp/a.wav",
        transcripts={
            "qwen": {"text": "需要确认一下"},
            "sensevoice": {"text": "需要确认"},
        },
    )
    operator = OperatorRegistry.get("quality.asr_edit_distance")
    result = operator.process(
        sample,
        OperatorConfig(params={"baseline": "qwen", "compare_models": ["sensevoice"]}),
    )
    quality = result.sample.quality
    assert quality["asr_edit_baseline"] == "qwen"
    assert quality["asr_edit_sensevoice_少字"] == 2
    assert quality["asr_edit_sensevoice_多字"] == 0
    assert quality["asr_edit_sensevoice_total"] == 2
    assert "sensevoice" in quality["asr_edit_json"]
