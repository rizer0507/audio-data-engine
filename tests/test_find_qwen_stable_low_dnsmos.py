"""Exercise real Parquet joins, status handling and exported audio paths."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/find_qwen_stable_low_dnsmos.py"


@pytest.fixture
def script():
    spec = importlib.util.spec_from_file_location("find_qwen_stable_low_dnsmos", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_parquet(path: Path, rows: list[dict]) -> Path:
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def qwen_row(sample_id: str, text: str = "你好", **entry) -> dict:
    return {
        "id": sample_id,
        "sha256": hashlib.sha256(sample_id.encode()).hexdigest(),
        "source_path": "/server/parent.wav",
        "audio": json.dumps({"resampled_16k": f"/server/segments/{sample_id}.wav"}),
        "transcripts": json.dumps({"qwen": {"text": text, **entry}}, ensure_ascii=False),
    }


def dns_row(sample_id: str, score=2.0, status="success", ovrl=2.0) -> dict:
    return {
        "id": sample_id,
        "quality": json.dumps({"dnsmos_sig": score, "dnsmos_ovrl": ovrl, "dnsmos_status": status}),
    }


def test_join_by_id_filters_status_empty_disagreement_and_bad_scores(script, tmp_path):
    ids = ["hit", "high", "failed", "different", "cleaned_empty", "no_dns",
           "dns_failed", "nan", "zero", "no_status", "no_peer"]
    first = write_parquet(tmp_path / "one.parquet", [qwen_row(i) for i in ids])
    peers = [qwen_row(i) for i in reversed(ids) if i != "no_peer"]
    for row in peers:
        if row["id"] == "hit":
            row.update(qwen_row("hit", "你好。"))
        elif row["id"] == "failed":
            row.update(qwen_row("failed", status="failed"))
        elif row["id"] == "different":
            row.update(qwen_row("different", "不好"))
        elif row["id"] == "cleaned_empty":
            row.update(qwen_row("cleaned_empty", "", extra={"raw_text": "你好"}))
    second = write_parquet(tmp_path / "two.parquet", peers)
    dnsmos = write_parquet(tmp_path / "dns.parquet", [
        dns_row("dns_failed", 1.0, "failed"), dns_row("high", 2.01),
        dns_row("hit", 2.0), dns_row("nan", float("nan")),
        dns_row("zero", 0), dns_row("no_status", 1.0, None),
        dns_row("failed"), dns_row("different"), dns_row("cleaned_empty"),
    ])
    output = tmp_path / "result.txt"
    result = script.find_candidates([first, second], dnsmos, output)
    assert result["matched"] == 1
    assert result["missing_dnsmos"] == 1
    assert result["invalid_dnsmos"] == 4
    assert result["qwen_unaligned"] == 1
    assert output.read_text(encoding="utf-8").splitlines() == ["/server/segments/hit.wav"]
    exported = pq.read_table(result["output_parquet"]).to_pylist()
    assert len(exported) == result["matched"]
    assert exported[0]["sample_id"] == "hit"
    assert exported[0]["wav_path"] == "/server/segments/hit.wav"
    assert exported[0]["dnsmos_sig"] == exported[0]["dnsmos_ovrl"] == 2.0
    transcripts = json.loads(exported[0]["transcripts"])
    assert transcripts["qwen_run_1"]["text"] == "你好"
    assert transcripts["qwen_run_2"]["text"] == "你好。"


def test_merged_native_structs_flat_quality_and_relative_path(script, tmp_path):
    qwen = write_parquet(tmp_path / "merged.parquet", [{
        "id": "x", "source_path": "parent.wav", "sha256": "a" * 64,
        "audio": {"resampled_16k": "segments/x.wav"},
        "transcripts": {"qwen_1": {"text": "需要。"}, "qwen_2": {"text": "需要"}},
    }])
    dnsmos = write_parquet(tmp_path / "dns.parquet", [{
        "id": "x", "quality_dnsmos_sig": 2.0, "quality_dnsmos_ovrl": 2.0,
        "quality_dnsmos_status": "success",
    }])
    output = tmp_path / "paths.txt"
    result = script.find_candidates([qwen], dnsmos, output, path_root=tmp_path)
    assert result["run_keys"] == ["qwen_1", "qwen_2"]
    assert result["matched"] == 1
    assert output.read_text(encoding="utf-8").strip() == str((tmp_path / "segments/x.wav").resolve())


def test_flat_texts_cli_count_and_source_path_option(script, tmp_path, capsys):
    qwen = write_parquet(tmp_path / "merged.parquet", [{
        "id": "x", "source_path": "/server/original.wav", "sha256": "a" * 64,
        "qwen1_text": "谢谢", "qwen2_text": "谢谢！",
    }])
    dnsmos = write_parquet(tmp_path / "dns.parquet", [dns_row("x", 2.9, ovrl=2.9)])
    output = tmp_path / "paths.txt"
    parquet_output = tmp_path / "annotation" / "items.parquet"
    assert script.main([
        "--qwen", str(qwen), "--dnsmos", str(dnsmos), "--sig-threshold", "3",
        "--ovrl-threshold", "3",
        "--output-parquet", str(parquet_output), "--batch", "my_batch",
        "--audio-key", "source_path", "-o", str(output),
    ]) == 0
    assert "命中总个数: 1" in capsys.readouterr().out
    assert output.read_text(encoding="utf-8") == "/server/original.wav\n"
    assert pq.read_table(parquet_output).to_pylist()[0]["batch"] == "my_batch"


def test_requires_both_sig_and_ovrl_low_including_boundaries(script, tmp_path):
    ids = ["both", "sig_only", "ovrl_only", "neither"]
    first = write_parquet(tmp_path / "one.parquet", [qwen_row(i) for i in ids])
    second = write_parquet(tmp_path / "two.parquet", [qwen_row(i) for i in reversed(ids)])
    dnsmos = write_parquet(tmp_path / "dns.parquet", [
        dns_row("both", 2.0, ovrl=2.0), dns_row("sig_only", 1.5, ovrl=2.01),
        dns_row("ovrl_only", 2.01, ovrl=1.5), dns_row("neither", 3.0, ovrl=3.0),
    ])
    output = tmp_path / "out.txt"
    result = script.find_candidates([first, second], dnsmos, output)
    assert result["matched"] == 1
    assert result["sig_low"] == result["ovrl_low"] == 2
    assert output.read_text(encoding="utf-8").splitlines() == ["/server/segments/both.wav"]


@pytest.mark.parametrize("ovrl", [None, float("nan"), float("inf"), 0, 5.1])
def test_invalid_ovrl_does_not_fall_back_to_sig_only(script, tmp_path, ovrl):
    qwen = write_parquet(tmp_path / "merged.parquet", [{
        "id": "x", "source_path": "/server/x.wav",
        "qwen1_text": "你好", "qwen2_text": "你好",
    }])
    dns = write_parquet(tmp_path / "dns.parquet", [dns_row("x", 1.5, ovrl=ovrl)])
    output = tmp_path / "out.txt"
    result = script.find_candidates([qwen], dns, output)
    assert result["matched"] == 0
    assert result["invalid_dnsmos"] == 1
    assert output.read_bytes() == b""
    empty = pq.read_table(output.with_suffix(".parquet"))
    assert empty.num_rows == 0
    assert {"sample_id", "original_audio_sha256", "wav_path", "candidate_text"} <= set(empty.column_names)


def test_missing_ovrl_field_is_skipped(script, tmp_path):
    qwen = write_parquet(tmp_path / "merged.parquet", [{
        "id": "x", "source_path": "/server/x.wav",
        "qwen1_text": "你好", "qwen2_text": "你好",
    }])
    dns = write_parquet(tmp_path / "dns.parquet", [{
        "id": "x", "dnsmos_sig": 1.5, "dnsmos_status": "success",
    }])
    result = script.find_candidates([qwen], dns, tmp_path / "out.txt")
    assert result["matched"] == 0
    assert result["invalid_dnsmos"] == 1


def test_explicit_failure_in_operator_status_is_not_stable(script, tmp_path):
    row = qwen_row("x")
    row["status"] = json.dumps({"asr.qwen": "failed"})
    row["errors"] = json.dumps({"asr.qwen": "timeout"})
    first = write_parquet(tmp_path / "one.parquet", [qwen_row("x")])
    second = write_parquet(tmp_path / "two.parquet", [row])
    dnsmos = write_parquet(tmp_path / "dns.parquet", [dns_row("x")])
    output = tmp_path / "paths.txt"
    result = script.find_candidates([first, second], dnsmos, output)
    assert result["matched"] == 0
    assert output.read_bytes() == b""


@pytest.mark.parametrize("problem", ["duplicate_id", "hash_conflict", "missing_path"])
def test_bad_inputs_do_not_replace_existing_output(script, tmp_path, problem):
    one, two, dns = qwen_row("x"), qwen_row("x"), dns_row("x")
    if problem == "hash_conflict":
        one["sha256"], two["sha256"] = "audio-one", "audio-two"
    if problem == "missing_path":
        one["audio"], one["source_path"] = "{}", None
    rows = [one, one] if problem == "duplicate_id" else [one]
    first = write_parquet(tmp_path / "one.parquet", rows)
    second = write_parquet(tmp_path / "two.parquet", [two])
    dnsmos = write_parquet(tmp_path / "dns.parquet", [dns])
    output = tmp_path / "paths.txt"
    output.write_text("previous\n", encoding="utf-8")
    with pytest.raises(ValueError):
        script.find_candidates([first, second], dnsmos, output)
    assert output.read_text(encoding="utf-8") == "previous\n"


def test_same_file_cannot_be_used_as_two_runs(script, tmp_path):
    qwen = write_parquet(tmp_path / "one.parquet", [qwen_row("x")])
    dns = write_parquet(tmp_path / "dns.parquet", [dns_row("x")])
    with pytest.raises(ValueError, match="输入文件不能重复"):
        script.find_candidates([qwen, qwen], dns, tmp_path / "out.txt")
    with pytest.raises(ValueError, match="两个不同"):
        script.find_candidates([qwen], dns, tmp_path / "out.txt", run_keys=["qwen", "qwen"])


def test_custom_keys_and_pending_are_handled(script, tmp_path):
    qwen = write_parquet(tmp_path / "merged.parquet", [{
        "id": "x", "source_path": "/server/x.wav",
        "transcripts": json.dumps({"pass1": {"text": "好"}, "pass2": {"text": "好", "status": "pending"}}),
    }])
    dns = write_parquet(tmp_path / "dns.parquet", [dns_row("x")])
    with pytest.raises(ValueError, match="run-keys"):
        script.find_candidates([qwen], dns, tmp_path / "out.txt")
    result = script.find_candidates([qwen], dns, tmp_path / "out.txt", run_keys=["pass1", "pass2"])
    assert result["stable_text"] == 0


def test_path_strings_keep_server_absolute_paths(script, tmp_path):
    for path in ("/server/audio/x.wav", "D:\\server\\x.wav"):
        row = {"id": "x", "audio": {"resampled_16k": path}}
        assert script.absolute_audio_path(row, audio_key="resampled_16k", path_root=tmp_path) == path


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), 0.5, 5.1])
def test_invalid_threshold_is_rejected(script, tmp_path, threshold):
    with pytest.raises(ValueError, match="sig-threshold"):
        script.find_candidates([tmp_path / "one.parquet"], tmp_path / "dns.parquet",
                               tmp_path / "out.txt", sig_threshold=threshold)


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), 0.5, 5.1])
def test_invalid_ovrl_threshold_is_rejected(script, tmp_path, threshold):
    with pytest.raises(ValueError, match="ovrl-threshold"):
        script.find_candidates([tmp_path / "one.parquet"], tmp_path / "dns.parquet",
                               tmp_path / "out.txt", ovrl_threshold=threshold)


def test_annotation_preserves_segment_identity_context_and_scores(script, tmp_path):
    row = qwen_row("seg_001")
    child_hash = row["sha256"]
    row["duration"] = 1.1
    row["batch"] = "source_batch"
    row["labels"] = json.dumps({
        "is_vad_segment": True, "segment_id": "seg_001", "source_audio_id": "parent_123",
        "business_record_id": "0000123", "original_audio_sha256": child_hash,
        "segment_pcm_sha256": child_hash, "parent_pcm_sha256": "parent_hash",
        "segment_index": 2, "vad_start_ms": 0, "vad_end_ms": None,
        "vad_inferred_end_ms": 1100, "vad_end_reason": "eof_fallback",
        "clip_start_sample": 0, "clip_end_sample": 17600, "clip_duration_ms": 1100,
        "category": "environment_noise",
    })
    one = write_parquet(tmp_path / "one.parquet", [row])
    two = write_parquet(tmp_path / "two.parquet", [row])
    score = dns_row("seg_001", 1.6, ovrl=1.8)
    quality = json.loads(score["quality"])
    quality["dnsmos_bak"] = 2.3
    score["quality"] = json.dumps(quality)
    dns = write_parquet(tmp_path / "dns.parquet", [score])
    result = script.find_candidates([one, two], dns, tmp_path / "out.txt")
    exported = pq.read_table(result["output_parquet"]).to_pylist()[0]
    assert exported["id"] == exported["sample_id"] == exported["segment_id"] == "seg_001"
    assert exported["sha256"] == exported["original_audio_sha256"] == child_hash
    assert exported["wav_path"] == "/server/segments/seg_001.wav"
    assert exported["source_path"] == "/server/parent.wav"
    assert exported["business_record_id"] == "0000123"
    assert exported["source_audio_id"] == "parent_123"
    assert exported["is_vad_segment"] is True
    assert exported["dnsmos_bak"] == 2.3
    assert exported["batch"] == "source_batch"
    assert exported["source_category"] == "environment_noise"
    assert exported["category"] == "semantic_risk"  # Only the platform template routing.
    labels = json.loads(exported["labels"])
    assert labels["parent_pcm_sha256"] == "parent_hash"
    assert labels["vad_end_ms"] is None and labels["vad_inferred_end_ms"] == 1100
    assert "category" not in labels
    assert json.loads(exported["source_labels"])["category"] == "environment_noise"
    assert exported["candidate_text"] == "你好" and "gold_text" not in exported


def test_vad_export_rejects_parent_audio_fallback(script, tmp_path):
    row = qwen_row("seg_001")
    row["audio"] = "{}"
    row["labels"] = json.dumps({"is_vad_segment": True, "source_audio_id": "parent"})
    one = write_parquet(tmp_path / "one.parquet", [row])
    two = write_parquet(tmp_path / "two.parquet", [row])
    dns = write_parquet(tmp_path / "dns.parquet", [dns_row("seg_001")])
    with pytest.raises(ValueError, match="父录音"):
        script.find_candidates([one, two], dns, tmp_path / "out.txt")
    assert not (tmp_path / "out.txt").exists()
    assert not (tmp_path / "out.parquet").exists()


def test_annotation_cannot_invent_missing_audio_hash(script, tmp_path):
    row = qwen_row("x")
    del row["sha256"]
    one = write_parquet(tmp_path / "one.parquet", [row])
    two = write_parquet(tmp_path / "two.parquet", [row])
    dns = write_parquet(tmp_path / "dns.parquet", [dns_row("x")])
    with pytest.raises(ValueError, match="缺少音频 sha256"):
        script.find_candidates([one, two], dns, tmp_path / "out.txt")


def test_parquet_output_cannot_overwrite_an_input(script, tmp_path):
    one, two, dns = [tmp_path / name for name in ("one.parquet", "two.parquet", "dns.parquet")]
    with pytest.raises(ValueError, match="覆盖输入"):
        script.find_candidates([one, two], dns, tmp_path / "out.txt", output_parquet=one)
    with pytest.raises(ValueError, match="不能相同"):
        script.find_candidates([one, two], dns, tmp_path / "out.parquet")


def test_parquet_write_failure_keeps_both_previous_outputs(script, tmp_path, monkeypatch):
    one = write_parquet(tmp_path / "one.parquet", [qwen_row("x")])
    two = write_parquet(tmp_path / "two.parquet", [qwen_row("x")])
    dns = write_parquet(tmp_path / "dns.parquet", [dns_row("x")])
    txt_output, pq_output = tmp_path / "out.txt", tmp_path / "out.parquet"
    txt_output.write_bytes(b"previous txt")
    pq_output.write_bytes(b"previous parquet")

    def fail_write(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(script.pq, "write_table", fail_write)
    with pytest.raises(OSError, match="disk full"):
        script.find_candidates([one, two], dns, txt_output)
    assert txt_output.read_bytes() == b"previous txt"
    assert pq_output.read_bytes() == b"previous parquet"
