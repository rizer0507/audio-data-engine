"""VAD segments 按 file_id 去掉接洽（身份核实）。"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/filter_segments_drop_identity_verify.py"


@pytest.fixture
def script():
    spec = importlib.util.spec_from_file_location(
        "filter_segments_drop_identity_verify", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    table = pa.table(
        {
            "id": ["seg_keep", "seg_drop_a", "seg_drop_b", "seg_missing"],
            "label_source_audio_id": [
                "keep-other",
                "drop-full",
                "drop-full",
                "keep-missing",
            ],
            "labels": [
                json.dumps({"source_audio_id": "keep-other"}, ensure_ascii=False),
                json.dumps({"source_audio_id": "drop-full"}, ensure_ascii=False),
                json.dumps({"source_audio_id": "drop-full"}, ensure_ascii=False),
                json.dumps({"source_audio_id": "keep-missing"}, ensure_ascii=False),
            ],
        }
    )
    parquet_path = tmp_path / "segments_batchA.parquet"
    pq.write_table(table, parquet_path)
    frame = pd.DataFrame(
        {
            "file_id": ["keep-other", "drop-full", "only-xlsx"],
            "AI话术分类标签": ["开场", "接洽（身份核实）", "接洽（身份核实）"],
        }
    )
    xlsx_path = tmp_path / "labels.xlsx"
    frame.to_excel(xlsx_path, index=False)
    return parquet_path, xlsx_path


def test_dry_run_prints_match_and_writes_nothing(script, tmp_path, capsys):
    parquet_path, xlsx_path = _write_inputs(tmp_path)
    before = parquet_path.read_bytes()
    code = script.main(
        [str(parquet_path), "--xlsx", str(xlsx_path), "--dry-run", "--no-backup"]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "file_id_column='file_id'" in out
    assert "label_column='AI话术分类标签'" in out
    assert "would_drop_rows=2" in out
    assert parquet_path.read_bytes() == before


def test_apply_backups_full_and_rewrites_segments(script, tmp_path):
    parquet_path, xlsx_path = _write_inputs(tmp_path)
    code = script.main([str(parquet_path), "--xlsx", str(xlsx_path), "--force"])
    assert code == 0
    backup = parquet_path.with_name("segments_batchA_vad_all.parquet")
    assert backup.is_file()
    assert pq.read_table(backup).num_rows == 4
    kept = pq.read_table(parquet_path).column("id").to_pylist()
    assert kept == ["seg_keep", "seg_missing"]


def test_zero_hits_aborts_without_allow_flag(script, tmp_path):
    table = pa.table(
        {
            "id": ["seg_only"],
            "label_source_audio_id": ["parent-a"],
        }
    )
    parquet_path = tmp_path / "segments_x.parquet"
    pq.write_table(table, parquet_path)
    frame = pd.DataFrame(
        {"file_id": ["other"], "AI话术分类标签": ["接洽（身份核实）"]}
    )
    xlsx_path = tmp_path / "labels.xlsx"
    frame.to_excel(xlsx_path, index=False)
    with pytest.raises(SystemExit, match="0 命中"):
        script.main([str(parquet_path), "--xlsx", str(xlsx_path), "--no-backup"])


def test_segment_id_is_not_matched_to_file_id(script, tmp_path):
    table = pa.table(
        {
            "id": ["seg_keep"],
            "label_source_audio_id": ["parent-keep"],
        }
    )
    parquet_path = tmp_path / "segments_y.parquet"
    pq.write_table(table, parquet_path)
    frame = pd.DataFrame(
        {"file_id": ["seg_keep"], "AI话术分类标签": ["接洽（身份核实）"]}
    )
    xlsx_path = tmp_path / "labels.xlsx"
    frame.to_excel(xlsx_path, index=False)
    with pytest.raises(SystemExit, match="0 命中"):
        script.main([str(parquet_path), "--xlsx", str(xlsx_path), "--no-backup"])
