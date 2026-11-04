"""再切段留下的 seg_ 父源，打包时改回父 wav 文件名里的 fenshen 编号。"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/pack_hardcase_review.py"


@pytest.fixture
def script():
    spec = importlib.util.spec_from_file_location("pack_hardcase_review", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolve_source_audio_id_replaces_segment_parent(script):
    parent = (
        "/data/pcm_to_wav/01/01958ec80da367c9_fenshen-18386e1d-917dcf2f.wav"
    )
    assert (
        script.resolve_source_audio_id("seg_9b3df4bfd43a23bdad70ede3318b", parent)
        == "fenshen-18386e1d-917dcf2f"
    )


def test_resolve_source_audio_id_keeps_real_parent(script):
    parent = "/data/pcm_to_wav/01/abc_fenshen-276cc46a-bf3387be.wav"
    assert (
        script.resolve_source_audio_id("fenshen-276cc46a-bf3387be", parent)
        == "fenshen-276cc46a-bf3387be"
    )


def test_row_writes_recovered_parent_and_vad_interval(script):
    child = "/data/vad_segments/se/seg_7c48eb23c029fd45d90e1d1e165c.wav"
    parent = (
        "/data/pcm_to_wav/01/01958ec80da367c9_fenshen-18386e1d-917dcf2f.wav"
    )
    row = pd.Series(
        {
            "id": "seg_7c48eb23c029fd45d90e1d1e165c",
            "label_segment_id": "seg_7c48eb23c029fd45d90e1d1e165c",
            "label_source_audio_id": "seg_9b3df4bfd43a23bdad70ede3318b",
            "sha256": "abc",
            "source_path": child,
            "audio": json.dumps(
                {"resampled_16k": child, "parent_resampled_16k": parent}
            ),
            "transcripts": json.dumps({"qwen1": {"text": "你咋个？"}}),
            "duration": 2.24,
            "label_parent_pcm_sha256": "01958ec80da367c9",
            "label_segment_index": 2,
            "label_vad_start_ms": 4800.0,
            "label_vad_end_ms": 5940.0,
            "label_vad_inferred_end_ms": None,
            "label_vad_end_reason": "vad_endpoint",
        }
    )
    packed = script._row(row, "label-7000")
    assert packed["source_audio_id"] == "fenshen-18386e1d-917dcf2f"
    assert packed["source_path"] == parent
    assert packed["segment_index"] == 2
    assert packed["vad_start_ms"] == 4800
    assert packed["vad_end_ms"] == 5940
    assert packed["vad_end_reason"] == "vad_endpoint"
    assert json.loads(packed["audio"])["resampled_16k"] == child


def test_repair_packed_parents_rewrites_only_segment_parents(script):
    parent = "/data/pcm_to_wav/01/01958ec80da367c9_fenshen-18386e1d-917dcf2f.wav"
    packed = pd.DataFrame(
        {
            "id": ["seg_child", "seg_ok"],
            "source_audio_id": ["seg_oldparent", "fenshen-276cc46a-bf3387be"],
            "source_path": [
                parent,
                "/data/pcm_to_wav/02/abc_fenshen-276cc46a-bf3387be.wav",
            ],
            "review_source": ["label-7000", "label-7000"],
        }
    )
    bucket = pd.DataFrame(
        {
            "id": ["seg_child", "seg_ok"],
            "label_segment_index": [2, 0],
            "label_vad_start_ms": [4800.0, 0.0],
            "label_vad_end_ms": [5940.0, 800.0],
            "label_vad_inferred_end_ms": [None, None],
            "label_vad_end_reason": ["vad_endpoint", "vad_endpoint"],
        }
    )
    repaired = script.repair_packed_parents(packed, bucket)
    assert repaired["source_audio_id"].tolist() == [
        "fenshen-18386e1d-917dcf2f",
        "fenshen-276cc46a-bf3387be",
    ]
    assert repaired["segment_index"].tolist() == [2, 0]
    assert repaired["vad_start_ms"].tolist() == [4800, 0]
    assert "label_segment_index" not in repaired.columns
