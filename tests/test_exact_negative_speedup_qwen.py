"""否定词精准命中、语气词删除后的字准，以及转写前的端口检查。"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pyarrow as pa
import pytest

from audio_engine.core.sample import Sample

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "exact_negative_speedup_qwen.py"


@pytest.fixture
def script():
    spec = importlib.util.spec_from_file_location("exact_negative_speedup_qwen", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_best_hit_keeps_longest_exact_route(script):
    hit = script.best_hit({"qwen1": "不用", "glm1": "暂时不用", "sv1": "我暂时不用谢谢"})
    assert hit == ("glm1", "暂时不用")


def test_best_hit_rejects_substring(script):
    assert script.best_hit({"qwen1": "我不用了", "glm1": "不用了谢谢"}) is None


def test_negative_phrases_come_from_yaml(script):
    phrases = script.exact_filter.NEGATIVE_PHRASES
    assert "不需要" in phrases and "嗯不要了" in phrases
    assert script.exact_filter.NEGATIVE_PHRASES_YAML.name == "fenshen_1015_negative_phrases.yaml"


def test_spoken_drop_uses_filler_yaml(script):
    tokens = script.speedup._SPOKEN_DROP_TOKENS
    assert "吧" in tokens and "你" in tokens and "你妈" in tokens
    assert "嗯" not in tokens and "哦" not in tokens
    assert script.speedup.FILLER_TOKENS_YAML.name == "fenshen_1015_filler_tokens.yaml"


def test_filler_only_difference_is_char_acc_one(script):
    mismatch, accuracy = script.char_acc_mismatch("不需要", "啊不需要吧")
    assert mismatch is False
    assert accuracy == 1.0


def test_different_words_are_char_acc_not_one(script):
    mismatch, accuracy = script.char_acc_mismatch("不需要", "是啊")
    assert mismatch is True
    assert accuracy != 1.0


def test_filter_stamps_label_from_any_route(script):
    table = pa.table(
        {
            "id": ["keep", "drop"],
            "labels": [
                json.dumps({"source_audio_id": "parent-1"}),
                json.dumps({"source_audio_id": "parent-2"}),
            ],
            "qwen1_text": ["是啊", "你好"],
            "glm1_text": ["不需要。", "暂时"],
            "sv1_text": ["", ""],
            "transcripts": ["{}", "{}"],
        }
    )
    filtered, hits = script.filter_exact_table(table)
    assert filtered.column("id").to_pylist() == ["keep"]
    assert hits["不需要"] == 1
    labels = json.loads(filtered.column("labels")[0].as_py())
    assert labels["exact_negative_route"] == "glm1"
    assert labels["exact_negative_label"] == "不需要"
    assert labels["source_audio_id"] == "parent-1"


def test_review_row_keeps_paths_and_both_qwen_texts(script):
    sample = Sample(
        id="derived-1",
        source_path="/tmp/speed.wav",
        sha256="speed-sha",
        audio={"aug_speed": "/tmp/speed.wav", "resampled_16k": "/tmp/speed.wav"},
        transcripts={"qwen_1p4": {"text": "是啊"}},
        labels={
            "augmentation": {
                "source": {
                    "sample_id": "seg-1",
                    "sha256": "src-sha",
                    "audio": {"resampled_16k": "/tmp/before.wav"},
                    "transcripts": {"qwen1": {"text": "嗯不需要"}},
                    "labels_extra": {
                        "exact_negative_label": "不需要",
                        "exact_negative_route": "glm1",
                        "exact_negative_phrase": "不需要",
                        "source_audio_id": "parent-1",
                        "normalized_audio_path": "/tmp/parent.wav",
                    },
                }
            }
        },
        status={"asr.qwen_batch": "completed"},
    )
    row = script.review_row(sample, before_key="qwen1", after_key="qwen_1p4")
    assert row["bucket"] == "mismatch"
    assert row["label"] == "不需要"
    assert row["qwen_before"] == "嗯不需要"
    assert row["qwen_after"] == "是啊"
    assert row["audio_before"] == "/tmp/before.wav"
    assert row["audio_after"] == "/tmp/speed.wav"
    assert row["source_audio_id"] == "parent-1"
    assert row["source_path"] == "/tmp/parent.wav"
    assert json.loads(row["audio"])["resampled_16k"] == "/tmp/speed.wav"
    transcripts = json.loads(row["transcripts"])
    assert transcripts["exact_negative_phrase"]["text"] == "不需要"
    assert transcripts["qwen1"]["text"] == "嗯不需要"
    assert transcripts["qwen_1p4"]["text"] == "是啊"
    assert list(transcripts)[0] == "exact_negative_phrase"
    assert row["candidate_text"] == "嗯不需要"
    assert row["char_acc"] != 1.0


def test_preannotation_falls_back_to_longest_contained_phrase(script):
    assert script.longest_contained_phrase("麻将扑克嗯不需要谢谢李诺") == "嗯不需要谢谢"
    assert script.longest_contained_phrase("不去") is None
    assert script.resolve_preannotation_phrase("", "呃不可以", "没有不可以") == "不可以"
    assert script.resolve_preannotation_phrase("不需要", "没有") == "不需要"


def test_unreachable_port_stops_before_transcription(script):
    def fetch(_api_base: str, *, timeout_s: float = 5.0):
        raise RuntimeError("connection refused")

    with pytest.raises(SystemExit, match="端口未联通"):
        script.assert_qwen_reachable("http://127.0.0.1:5555", "qwen3-asr", fetch=fetch)
