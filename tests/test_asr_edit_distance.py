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


def test_normalize_transcripts_strips_tags_and_punct():
    sample = Sample(
        id="s1",
        source_path="/tmp/a.wav",
        transcripts={
            "qwen": {"text": "您好，世界！", "model": "qwen"},
            "sensevoice": {
                "text": "<|zh|><|EMO_UNKNOW|>您好，世界",
                "model": "sensevoice",
                "extra": {"emotion": "EMO_UNKNOW"},
            },
        },
    )
    operator = OperatorRegistry.get("quality.normalize_transcripts")
    result = operator.process(
        sample,
        OperatorConfig(params={"models": ["qwen", "sensevoice"], "keep_raw": True}),
    )
    qwen = result.sample.transcripts["qwen"]
    sense = result.sample.transcripts["sensevoice"]
    assert qwen["text"] == "您好世界"
    assert qwen["extra"]["raw_text"] == "您好，世界！"
    assert sense["text"] == "您好世界"
    assert sense["extra"]["raw_text"] == "<|zh|><|EMO_UNKNOW|>您好，世界"
    assert sense["extra"]["emotion"] == "EMO_UNKNOW"


def test_normalize_transcripts_blanks_exact_qwen_phrases(tmp_path):
    operator = OperatorRegistry.get("quality.normalize_transcripts")
    params = {
        "keep_raw": True,
        "blank_exact_hotwords_path": "configs/normalization/blank_exact_qwen_v1.yaml",
    }
    config = OperatorConfig(params=params, cache_dir=tmp_path / "cache", force=True)

    # vocabulary dump as one sentence (the real failure mode)
    blanked = operator.process(
        Sample(
            id="hot",
            source_path="/tmp/a.wav",
            sha256="a" * 64,
            transcripts={
                "qwen": {
                    "text": "没有，暂时不用，不需要谢谢，不可以，啊不用，我不需要，不要，不用，不需要。"
                },
                "sensevoice": {"text": "<|zh|>你好。"},
            },
        ),
        config,
    ).sample
    assert blanked.transcripts["qwen"]["text"] == ""
    assert blanked.transcripts["qwen"]["extra"]["blanked_exact_hotword"] is True
    assert blanked.transcripts["sensevoice"]["text"] == "你好"

    # single short phrase must NOT be blanked under the dump-string rule
    kept_short = operator.process(
        Sample(
            id="short",
            source_path="/tmp/b.wav",
            sha256="b" * 64,
            transcripts={"qwen": {"text": "不需要！"}},
        ),
        config,
    ).sample
    assert kept_short.transcripts["qwen"]["text"] == "不需要"

    kept = operator.process(
        Sample(
            id="more",
            source_path="/tmp/c.wav",
            sha256="c" * 64,
            transcripts={
                "qwen": {"text": "我不需要贷款"},
                "sensevoice": {"text": "我不需要贷款"},
            },
        ),
        config,
    ).sample
    assert kept.transcripts["qwen"]["text"] == "我不需要贷款"

