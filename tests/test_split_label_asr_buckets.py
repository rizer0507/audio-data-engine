"""三路转写和人工标注取交集后的字面替换与分桶。"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/split_label_asr_buckets.py"


@pytest.fixture
def script():
    spec = importlib.util.spec_from_file_location("split_label_asr_buckets", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_apply_colloquial_replaces_every_occurrence_longer_first(script):
    assert script.apply_colloquial("哎，唉啦嘛那地滴呃昂哈噢喔恩摁啥") == "喂，喂了吗哪的的啊啊啊哦哦嗯嗯什么"
    assert script.apply_colloquial("泥马尼玛啥事儿") == "你妈你妈什么事"
    assert script.apply_colloquial("那那") == "哪哪"
    assert script.apply_colloquial(None) == ""


def test_non_speech_stays_and_text_is_not_rewritten(script, tmp_path: Path):
    transcripts = tmp_path / "three.parquet"
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    pq.write_table(
        pa.table(
            {
                "id": ["speech-agree", "speech-qwen", "noise", "asr-only"],
                "qwen_1_text": ["哎。", "没有", "哎", "喂"],
                "glm_text": ["唉", "喂", "哎", "喂"],
                "sensevoice_text": ["喂", "喂", "哎", "喂"],
                "source_path": ["a.wav", "b.wav", "c.wav", "d.wav"],
            }
        ),
        transcripts,
    )
    pq.write_table(
        pa.table(
            {
                "sample_id": ["speech-agree", "speech-qwen", "noise", "ann-only"],
                "gold_text": ["喂", "喂呀", "哎", "喂"],
                "gold_kind": ["speech", "speech", "non_speech", "speech"],
                "status": ["annotated", "annotated", "empty", "annotated"],
            }
        ),
        export_dir / "annotations_demo.parquet",
    )

    summary = script.split_buckets(transcripts, export_dir, tmp_path / "out")
    assert summary["intersection"] == 3
    assert summary["asr_only"] == 1
    assert summary["annotation_only"] == 1

    classified = pq.read_table(tmp_path / "out" / "classified.parquet")
    rows = {row["id"]: row for row in classified.to_pylist()}
    assert rows["speech-agree"]["bucket"] == "auto_gold"
    assert rows["speech-agree"]["norm_qwen_1_text"] == "喂。"
    assert rows["speech-agree"]["norm_glm_text"] == "喂"
    assert rows["speech-agree"]["qwen_1_text"] == "哎。"
    assert rows["speech-qwen"]["bucket"] == "hardcase"
    assert rows["speech-qwen"]["norm_gold_text"] == "喂呀"
    assert rows["noise"]["bucket"] == "non_speech"
    assert rows["noise"]["norm_qwen_1_text"] == "哎"
    assert rows["noise"]["norm_gold_text"] == "哎"
    assert "asr-only" not in rows
    assert pq.read_table(tmp_path / "out" / "non_speech.parquet").column("id").to_pylist() == ["noise"]
    assert summary["buckets"] == {"auto_gold": 1, "hardcase": 1, "non_speech": 1, "other": 0}


def test_qwen_match_without_three_way_agreement_is_other(script):
    assert script.assign_bucket("speech", "喂", "你好", "喂", "喂") == "other"
    assert script.assign_bucket("non_speech", "喂", "喂", "喂", "喂") == "non_speech"
