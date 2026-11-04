"""Bucket rules and non_speech insertion distance for post-train evaluation."""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

from audio_engine.core.post_train_eval import (
    assign_bucket,
    evaluate,
    load_prediction,
    score_hypothesis,
)


def test_assign_bucket_priority_and_three_way_agreement():
    assert (
        assign_bucket(gold_kind="non_speech", qwen="你好", glm="你好", sensevoice="你好", human="你好")
        == "non_speech"
    )
    assert (
        assign_bucket(gold_kind="speech", qwen="喂。", glm="喂", sensevoice="喂，", human="喂")
        == "auto_gold"
    )
    assert assign_bucket(gold_kind="speech", qwen="没有", glm="喂", sensevoice="喂", human="喂") == "hardcase"
    assert assign_bucket(gold_kind="speech", qwen="喂", glm="你好", sensevoice="喂", human="喂。") == "other"


def test_non_speech_distance_is_insertion_count():
    empty = score_hypothesis("non_speech", "你好", "")
    assert empty.exact == 1
    assert empty.insertions == 0
    extra = score_hypothesis("non_speech", "你好", "哎，你好。")
    assert extra.exact == 0
    assert extra.insertions == 3
    assert extra.edits == extra.insertions


def test_evaluate_writes_source_columns_and_split_metrics(tmp_path: Path):
    transcripts = tmp_path / "three_way.parquet"
    gold = tmp_path / "gold.parquet"
    train_pred = tmp_path / "model_train.parquet"
    dev_pred = tmp_path / "model_dev.parquet"
    pq.write_table(
        _table(
            [
                {"id": "a", "qwen_1_text": "喂。", "glm_text": "喂", "sensevoice_text": "喂", "source_path": "vad/a.wav"},
                {"id": "b", "qwen_1_text": "没有", "glm_text": "喂", "sensevoice_text": "喂", "source_path": "vad/b.wav"},
                {"id": "c", "qwen_1_text": "噪声", "glm_text": "噪声", "sensevoice_text": "噪声", "source_path": "vad/c.wav"},
            ]
        ),
        transcripts,
    )
    pq.write_table(
        _table(
            [
                {"id": "a", "gold_text": "喂", "gold_kind": "speech"},
                {"id": "b", "gold_text": "喂", "gold_kind": "speech"},
                {"id": "c", "gold_text": "", "gold_kind": "non_speech"},
            ]
        ),
        gold,
    )
    pq.write_table(_table([{"id": "a", "new-model_text": "喂"}, {"id": "b", "new-model_text": "喂"}]), train_pred)
    pq.write_table(_table([{"id": "c", "new-model_text": "啊"}]), dev_pred)

    output = tmp_path / "out"
    result = evaluate(
        transcripts,
        [
            load_prediction("train", train_pred, "new-model"),
            load_prediction("dev", dev_pred, "new-model"),
        ],
        output,
        gold,
    )

    classified = pq.read_table(output / "classified.parquet")
    assert "qwen_1_text" in classified.schema.names
    assert "source_path" in classified.schema.names
    buckets = dict(zip(classified.column("id").to_pylist(), classified.column("eval_bucket").to_pylist()))
    assert buckets == {"a": "auto_gold", "b": "hardcase", "c": "non_speech"}
    assert result["summary"][0]["split"] == "train"
    by_key = {(row["split"], row["bucket"]): row for row in result["summary"]}
    assert by_key[("train", "auto_gold")]["accuracy"] == 100.0
    assert by_key[("train", "auto_gold")]["char_acc"] == 100.0
    assert by_key[("train", "hardcase")]["correct"] == 1
    assert by_key[("dev", "non_speech")]["insertions"] == 1
    assert by_key[("dev", "non_speech")]["char_acc"] is None
    assert (output / "evaluation.xlsx").is_file()


def _table(rows: list[dict]):
    import pyarrow as pa

    keys = list(rows[0])
    return pa.table({key: [row.get(key) for row in rows] for key in keys})
