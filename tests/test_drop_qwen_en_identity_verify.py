"""按 file_id 从 qwen 单字嗯证据里去掉接洽（身份核实）。"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/drop_qwen_en_identity_verify.py"


@pytest.fixture
def script():
    spec = importlib.util.spec_from_file_location("drop_qwen_en_identity_verify", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    table = pa.table(
        {
            "id": ["seg_keep", "seg_full", "seg_half_a", "seg_half_b", "seg_missing", "seg_space"],
            "label_source_audio_id": [
                "keep-other",
                "drop-full",
                "drop-half",
                "drop-half",
                "keep-missing",
                "  drop-space  ",
            ],
            "qwen_text": ["嗯", "嗯", "嗯", "嗯", "嗯", "嗯"],
        }
    )
    parquet_path = tmp_path / "prepared_evidence_b1_b2_qwen_en.parquet"
    pq.write_table(table, parquet_path)
    frame = pd.DataFrame(
        {
            "file-id": ["keep-other", "drop-full", "drop-half", "only-xlsx", "", "drop-space"],
            "AI话术分类标签": [
                "开场",
                "接洽（身份核实）",
                "接洽(身份核实)",
                "接洽（身份核实）",
                "接洽（身份核实）",
                " 接洽（身份核实） ",
            ],
        }
    )
    xlsx_path = tmp_path / "labels.xlsx"
    frame.to_excel(xlsx_path, index=False)
    return parquet_path, xlsx_path


def test_drop_only_identity_verify_rows(script, tmp_path):
    parquet_path, xlsx_path = _write_inputs(tmp_path)
    out = tmp_path / "out" / "kept.parquet"
    code = script.main([str(parquet_path), "--xlsx", str(xlsx_path), "--output", str(out)])
    assert code == 0
    kept = pq.read_table(out).column("id").to_pylist()
    assert kept == ["seg_keep", "seg_missing"]


def test_same_file_id_with_any_identity_row_is_dropped(script):
    frame = pd.DataFrame(
        {
            "file_id": ["same", "same"],
            "AI话术分类标签": ["开场", "接洽（身份核实）"],
        }
    )
    drop_ids, stats = script.drop_file_ids(frame)
    assert drop_ids == {"same"}
    assert stats["label_rows"] == 1
    table = pa.table(
        {
            "id": ["seg_same", "seg_other"],
            "label_source_audio_id": ["same", "other"],
        }
    )
    filtered, parquet_stats = script.filter_table(table, drop_ids)
    assert filtered.column("id").to_pylist() == ["seg_other"]
    assert parquet_stats["match_source"] == "label_source_audio_id"
    assert parquet_stats["dropped"] == 1
    assert parquet_stats["drop_ids_miss"] == 0


def test_segment_id_is_not_the_match_key(script):
    drop_ids = {"seg_keep"}
    table = pa.table(
        {
            "id": ["seg_keep"],
            "label_source_audio_id": ["parent-other"],
        }
    )
    filtered, stats = script.filter_table(table, drop_ids)
    assert filtered.column("id").to_pylist() == ["seg_keep"]
    assert stats["dropped"] == 0


def test_nested_labels_fill_blank_flat_column(script):
    table = pa.table(
        {
            "id": ["seg_nested", "seg_flat"],
            "label_source_audio_id": [None, "flat-parent"],
            "labels": ['{"source_audio_id": "nested-parent"}', None],
        }
    )
    filtered, stats = script.filter_table(table, {"nested-parent"})
    assert filtered.column("id").to_pylist() == ["seg_flat"]
    assert stats["match_source"] == "label_source_audio_id+labels.source_audio_id"


def test_default_output_name_appends_drop_identity(script, tmp_path):
    path = tmp_path / "prepared_evidence_b1_b2_qwen_en.parquet"
    assert script.output_path_for(path, tmp_path / "dst") == (
        tmp_path / "dst" / "prepared_evidence_b1_b2_qwen_en_drop_identity.parquet"
    )


def test_missing_label_column_exits(script):
    frame = pd.DataFrame({"file_id": ["a"], "其他列": ["x"]})
    with pytest.raises(SystemExit, match="话术分类"):
        script.drop_file_ids(frame)


def test_missing_parent_column_exits(script):
    table = pa.table({"id": ["seg_only"]})
    with pytest.raises(SystemExit, match="label_source_audio_id"):
        script.filter_table(table, {"seg_only"})
