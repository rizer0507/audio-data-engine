from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from audio_engine.core.transcript_reconcile import (
    character_similarity,
    clean_control_tags,
    reconcile_transcripts,
)


def test_clean_and_character_similarity():
    assert clean_control_tags("<|zh|><|EMO_UNKNOW|><|within|>你 好！") == "你 好！"
    assert character_similarity("你好。", "<|zh|>你 好") == 1.0
    assert character_similarity("你好", "你们好") == pytest.approx(0.6667)


def test_reconcile_xlsx_and_fenp(tmp_path: Path):
    source = tmp_path / "qwen.xlsx"
    pd.DataFrame(
        [
            {"id": "a", "qwen_text": "您好，世界！", "备注": "保留"},
            {"id": "b", "qwen_text": "天气很好", "备注": "<|within|>需清洗"},
            {"id": "c", "qwen_text": "缺失结果", "备注": "保留"},
        ]
    ).to_excel(source, index=False)
    fenp = tmp_path / "sensevoice.fenp"
    records = [
        {"id": "a", "transcripts": {"sensevoice": {"text": "<|zh|>您好世界"}}},
        {"id": "b", "text": "<|EMO_UNKNOW|><|within|>天气不好"},
    ]
    fenp.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in records))

    output = tmp_path / "cleaned.xlsx"
    summary = reconcile_transcripts(source, fenp, output, threshold=0.8)
    result = pd.read_excel(output)

    assert summary == {
        "total": 3,
        "consistent": 1,
        "inconsistent": 2,
        "missing_sensevoice": 1,
        "consistent_rate": 0.3333,
        "threshold": 0.8,
        "output": str(output),
    }
    assert result["asr_consistent"].tolist() == [True, False, False]
    assert result.loc[0, "sensevoice_clean_text"] == "您好世界"
    assert result.loc[2, "comparison_reason"] == "未找到SenseVoice结果"
    assert not result.astype(str).apply(lambda column: column.str.contains(r"<\|", regex=True)).any().any()


def test_reconcile_rejects_duplicate_sensevoice_ids(tmp_path: Path):
    source = tmp_path / "qwen.xlsx"
    pd.DataFrame([{"id": "a", "qwen_text": "文本"}]).to_excel(source, index=False)
    result = tmp_path / "result.jsonl"
    result.write_text('{"id":"a","text":"一"}\n{"id":"a","text":"二"}\n')

    with pytest.raises(ValueError, match="ID不唯一"):
        reconcile_transcripts(source, result, tmp_path / "out.xlsx")
