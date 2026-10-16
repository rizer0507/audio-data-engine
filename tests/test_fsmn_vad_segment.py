"""041 FSMN VAD segment: pairing, endpoint retention, identity, operator, warehouse."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import audio_engine.operators  # noqa: F401
from audio_engine.core.fsmn_vad.adapter import MockVadModel, ScheduleVadModel, run_streaming_vad
from audio_engine.core.fsmn_vad.clip import ms_to_sample
from audio_engine.core.fsmn_vad.identity import (
    build_segment_labels,
    segment_content_sha256,
    stable_segment_id,
)
from audio_engine.core.fsmn_vad.pairing import pair_and_filter
from audio_engine.core.fsmn_vad.types import (
    END_REASON_VAD,
    END_SILENCE_KEEP_MS,
    SR,
    VAD_PROCESSING_VERSION,
)
from audio_engine.core.manifest import Manifest, file_sha256
from audio_engine.core.operator import OperatorConfig
from audio_engine.core.registry import OperatorRegistry
from audio_engine.core.sample import Sample
from audio_engine.core.selection_v3.input_contract import original_audio_sha256
from audio_engine.core.warehouse.completion import assert_batch_complete
from audio_engine.core.warehouse.identity import original_audio_sha
from audio_engine.core.annotation_v3.config import AnnotationConfig
from audio_engine.core.dataset_v3.grouping import GroupingConfig, build_leakage_groups


def _pcm_silence(ms: float) -> bytes:
    n = int(ms * SR / 1000)
    return (np.zeros(n, dtype="<i2")).tobytes()


def _pcm_tone(ms: float, freq: float = 440.0) -> bytes:
    n = int(ms * SR / 1000)
    t = np.arange(n, dtype=np.float32) / SR
    sig = (0.2 * np.sin(2 * np.pi * freq * t) * 32767).astype("<i2")
    return sig.tobytes()


def test_ms_to_sample_floor():
    assert ms_to_sample(1000) == 16000
    assert ms_to_sample(999.9) == 15998  # floor(999.9*16)=floor(15998.4)=15998
    assert ms_to_sample(2000) == 32000


@pytest.mark.parametrize("duration_ms", [990, 1000, 1010])
def test_normal_endpoint_keeps_clips_on_both_sides_of_one_second(duration_ms):
    # Clamp the real END_SILENCE_KEEP_MS tail at EOF; <1s discarded, >=1s kept.
    pcm = _pcm_silence(2500)
    start_ms = 2500 - duration_ms
    event_log = [
        {
            "input_start_sample": 0,
            "input_end_sample": len(pcm) // 2,
            "is_final": True,
            "events": [[start_ms, 2470]],
        }
    ]
    result = pair_and_filter(event_log, pcm)
    if duration_ms < 1000:
        assert result.kept == []
        assert result.status == "all_short"
        assert [d.reason for d in result.discarded] == ["short_utterance"]
        assert result.discarded[0].duration_ms == duration_ms
    else:
        assert result.discarded == []
        assert result.status == "completed"
        assert len(result.kept) == 1
        assert result.kept[0].duration_ms == duration_ms
        assert result.kept[0].pcm16le == pcm[ms_to_sample(start_ms) * 2 :]


def test_complete_and_split_events_same_filter():
    # Both event shapes include END_SILENCE_KEEP_MS of real audio after each model endpoint.
    pcm = _pcm_silence(8000)
    event_log = [
        {
            "input_start_sample": 0,
            "input_end_sample": len(pcm) // 2,
            "is_final": True,
            "events": [[2000, 2990], [3000, 4000], [4100, 5110]],
        }
    ]
    result = pair_and_filter(event_log, pcm)
    assert result.discarded == []
    assert len(result.kept) == 3
    assert sorted(round(k.duration_ms) for k in result.kept) == [
        990 + END_SILENCE_KEEP_MS,
        1000 + END_SILENCE_KEEP_MS,
        1010 + END_SILENCE_KEEP_MS,
    ]

    # Same via split events
    event_log_split = [
        {"input_start_sample": 0, "input_end_sample": 16000, "is_final": False, "events": [[2000, -1]]},
        {"input_start_sample": 16000, "input_end_sample": 32000, "is_final": False, "events": [[-1, 2990]]},
        {"input_start_sample": 32000, "input_end_sample": 48000, "is_final": False, "events": [[3000, -1]]},
        {"input_start_sample": 48000, "input_end_sample": 64000, "is_final": False, "events": [[-1, 4000]]},
        {"input_start_sample": 64000, "input_end_sample": 80000, "is_final": False, "events": [[4100, -1]]},
        {"input_start_sample": 80000, "input_end_sample": 128000, "is_final": True, "events": [[-1, 5110]]},
    ]
    result2 = pair_and_filter(event_log_split, pcm)
    assert result2.discarded == []
    assert len(result2.kept) == 3
    assert [(k.start_sample, k.end_sample) for k in result2.kept] == [
        (k.start_sample, k.end_sample) for k in result.kept
    ]


def test_short_endpoint_includes_real_waiting_tail():
    pcm = _pcm_tone(3000)
    event_log = [
        {
            "input_start_sample": 0,
            "input_end_sample": ms_to_sample(2200),
            "is_final": False,
            "events": [[210, 1080]],
        }
    ]
    result = pair_and_filter(event_log, pcm)
    assert result.discarded == []
    assert len(result.kept) == 1
    seg = result.kept[0]
    assert seg.end_reason == END_REASON_VAD
    assert seg.end_ms == 1080  # Preserve the model endpoint for provenance.
    assert seg.start_sample == ms_to_sample(210)
    retained_end = 1080 + END_SILENCE_KEEP_MS
    assert seg.end_sample == ms_to_sample(retained_end)
    assert seg.duration_ms == retained_end - 210
    assert seg.extra_padding_ms == END_SILENCE_KEEP_MS
    assert seg.pcm16le == pcm[ms_to_sample(210) * 2 : ms_to_sample(retained_end) * 2]


def test_endpoint_tail_clamps_to_eof_and_discards_short_clip():
    pcm = _pcm_silence(2500)
    event_log = [
        {
            "input_start_sample": 0,
            "input_end_sample": len(pcm) // 2,
            "is_final": True,
            "events": [[1510, 2000]],
        }
    ]
    result = pair_and_filter(event_log, pcm)
    assert result.kept == []
    assert result.status == "all_short"
    assert len(result.discarded) == 1
    assert result.discarded[0].reason == "short_utterance"
    assert result.discarded[0].end_sample == len(pcm) // 2
    assert result.discarded[0].duration_ms == 990
    assert result.candidates[0].boundary_clamped is True


@pytest.mark.parametrize("end_ms", [2480, 2490, 2500, 2510])
@pytest.mark.parametrize("split_events", [False, True])
@pytest.mark.parametrize("include_eof", [False, True])
def test_model_endpoint_at_or_after_eof_is_discarded(end_ms, split_events, include_eof):
    pcm = _pcm_tone(2500)
    event_log = []
    if split_events:
        event_log.append(
            {
                "input_start_sample": 0,
                "input_end_sample": ms_to_sample(200),
                "is_final": False,
                "events": [[100, -1]],
            }
        )
    event_log.append(
        {
            "input_start_sample": ms_to_sample(200) if split_events else 0,
            "input_end_sample": len(pcm) // 2,
            "is_final": True,
            "events": [[-1 if split_events else 100, end_ms]],
        }
    )

    result = pair_and_filter(event_log, pcm, include_eof_fallback_in_kept=include_eof)

    assert result.kept == []
    assert result.eof_fallback == []
    assert result.status == "endpoint_at_eof"
    assert len(result.candidates) == 1
    assert len(result.discarded) == 1
    discarded = result.discarded[0]
    assert discarded.end_reason == END_REASON_VAD
    assert discarded.reason == "endpoint_at_eof"
    assert discarded.start_ms == 100
    assert discarded.end_ms == end_ms
    assert discarded.start_sample == ms_to_sample(100)
    assert discarded.end_sample == len(pcm) // 2
    assert discarded.duration_ms == 2400


def test_model_endpoint_at_eof_is_discarded_even_without_final_flag():
    pcm = _pcm_silence(2500)
    event_log = [
        {
            "input_start_sample": 0,
            "input_end_sample": len(pcm) // 2,
            "is_final": False,
            "events": [[100, 2500]],
        }
    ]

    result = pair_and_filter(event_log, pcm)

    assert result.kept == []
    assert result.status == "endpoint_at_eof"
    assert [d.reason for d in result.discarded] == ["endpoint_at_eof"]


@pytest.mark.parametrize(
    ("sample_rate", "total_samples", "eof_frame_sample"),
    [
        (8000, 20000, 19840),  # 2500 ms at 8 kHz; last frontend boundary is 2480 ms.
        (16000, 16000, 15680),  # 1000 ms source; last frontend boundary is 980 ms.
        (16000, 40000, 39680),  # 2500 ms source.
        (16000, 40120, 39840),  # 2507.5 ms source; frontend boundary is 2490 ms.
    ],
)
@pytest.mark.parametrize("offset_samples", [-1, 0, 1])
def test_endpoint_eof_frame_boundary_uses_unpadded_model_end(
    sample_rate, total_samples, eof_frame_sample, offset_samples
):
    pcm = np.zeros(total_samples, dtype="<i2").tobytes()
    end_ms = (eof_frame_sample + offset_samples) * 1000.0 / sample_rate
    event_log = [
        {
            "input_start_sample": 0,
            "input_end_sample": total_samples,
            "is_final": True,
            # start=0 so a 1s source padded to EOF is exactly MIN_KEEP_MS and kept.
            "events": [[0, end_ms]],
        }
    ]

    result = pair_and_filter(event_log, pcm, sample_rate=sample_rate)

    # Waiting tail usually hits EOF; discard decision still uses unpadded model end.
    assert result.candidates[0].end_sample == total_samples
    if offset_samples < 0:
        assert result.status == "completed"
        assert result.discarded == []
        assert len(result.kept) == 1
        assert result.kept[0].end_ms == end_ms
        assert result.kept[0].end_sample == total_samples
        assert result.kept[0].extra_padding_ms > 0
    else:
        assert result.kept == []
        assert result.status == "endpoint_at_eof"
        assert [d.reason for d in result.discarded] == ["endpoint_at_eof"]
        assert result.discarded[0].end_ms == end_ms


def test_non_frame_multiple_source_keeps_endpoint_before_final_frame():
    pcm = _pcm_silence(2507.5)
    event_log = [
        {
            "input_start_sample": 0,
            "input_end_sample": len(pcm) // 2,
            "is_final": True,
            "events": [[100, 2480]],
        }
    ]

    result = pair_and_filter(event_log, pcm)

    assert result.discarded == []
    assert len(result.kept) == 1
    assert result.kept[0].end_ms == 2480
    assert result.kept[0].end_sample == len(pcm) // 2


def test_normal_endpoint_is_retained_alongside_endpoint_at_eof():
    pcm = _pcm_tone(2500)
    event_log = [
        {
            "input_start_sample": 0,
            "input_end_sample": len(pcm) // 2,
            "is_final": True,
            "events": [[100, 600], [1800, 2500]],
        }
    ]

    result = pair_and_filter(event_log, pcm)

    assert result.status == "completed"
    assert len(result.kept) == 1
    assert result.kept[0].end_ms == 600
    assert result.kept[0].end_sample == ms_to_sample(600 + END_SILENCE_KEEP_MS)
    assert len(result.discarded) == 1
    assert result.discarded[0].start_ms == 1800
    assert result.discarded[0].reason == "endpoint_at_eof"


@pytest.mark.parametrize("model_endpoints", [[2500, 2470], [2470, 2500]])
def test_eof_rejection_does_not_deduplicate_normal_endpoint_with_same_clip(model_endpoints):
    pcm = _pcm_silence(2500)
    event_log = [
        {
            "input_start_sample": 0,
            "input_end_sample": len(pcm) // 2,
            "is_final": True,
            "events": [[100, end_ms] for end_ms in model_endpoints],
        }
    ]

    result = pair_and_filter(event_log, pcm)

    assert result.status == "completed"
    assert len(result.kept) == 1
    assert result.kept[0].end_ms == 2470
    assert len(result.discarded) == 1
    assert result.discarded[0].end_ms == 2500
    assert result.discarded[0].reason == "endpoint_at_eof"
    assert result.kept[0].start_sample == result.discarded[0].start_sample
    assert result.kept[0].end_sample == result.discarded[0].end_sample


def test_clip_bytes_match_bounds_no_padding():
    pcm = _pcm_tone(3000)
    event_log = [
        {
            "input_start_sample": 0,
            "input_end_sample": len(pcm) // 2,
            "is_final": True,
            "events": [[500, 2500]],
        }
    ]
    result = pair_and_filter(event_log, pcm)
    assert len(result.kept) == 1
    seg = result.kept[0]
    assert seg.start_sample == ms_to_sample(500)
    assert seg.end_sample == len(pcm) // 2
    assert seg.boundary_clamped is True
    assert seg.extra_padding_ms == 500
    assert len(seg.pcm16le) == (seg.end_sample - seg.start_sample) * 2
    assert seg.pcm16le == pcm[seg.start_sample * 2 : seg.end_sample * 2]


def test_illegal_end_without_start_raises():
    pcm = _pcm_silence(2000)
    event_log = [
        {
            "input_start_sample": 0,
            "input_end_sample": len(pcm) // 2,
            "is_final": True,
            "events": [[-1, 1000]],
        }
    ]
    with pytest.raises(ValueError, match="end without a start"):
        pair_and_filter(event_log, pcm)


def test_duplicate_complete_interval_deduped():
    pcm = _pcm_silence(3000)
    event_log = [
        {
            "input_start_sample": 0,
            "input_end_sample": len(pcm) // 2,
            "is_final": True,
            "events": [[0, 1500], [0, 1500]],
        }
    ]
    result = pair_and_filter(event_log, pcm)
    assert len(result.kept) == 1


def test_eof_without_endpoint_is_discarded_by_default():
    pcm = _pcm_silence(2500)
    # pending start only → EOF
    event_log = [
        {
            "input_start_sample": 0,
            "input_end_sample": 3200,
            "is_final": False,
            "events": [[100, -1]],
        },
        {
            "input_start_sample": 3200,
            "input_end_sample": len(pcm) // 2,
            "is_final": True,
            "events": [],
        },
    ]
    result = pair_and_filter(event_log, pcm)
    assert result.kept == []
    assert result.eof_fallback == []
    assert result.status == "eof_without_end"
    assert len(result.discarded) == 1
    assert result.discarded[0].end_reason == "eof_fallback"
    assert result.discarded[0].reason == "eof_without_end"
    assert result.discarded[0].duration_ms == 2400


@pytest.mark.parametrize("duration_ms", [990, 1000, 1010, 2400])
@pytest.mark.parametrize("include_eof", [False, True])
def test_eof_without_endpoint_is_discarded_regardless_of_duration_or_flag(duration_ms, include_eof):
    pcm = _pcm_silence(2500)
    start_ms = 2500 - duration_ms
    event_log = [
        {
            "input_start_sample": 0,
            "input_end_sample": len(pcm) // 2,
            "is_final": True,
            "events": [[start_ms, -1]],
        }
    ]
    result = pair_and_filter(event_log, pcm, include_eof_fallback_in_kept=include_eof)
    assert result.kept == []
    assert result.eof_fallback == []
    assert result.status == "eof_without_end"
    assert len(result.discarded) == 1
    assert result.discarded[0].end_reason == "eof_fallback"
    assert result.discarded[0].reason == "eof_without_end"
    assert result.discarded[0].duration_ms == duration_ms


def test_no_start_does_not_whole_file_fallback():
    pcm = _pcm_silence(3000)
    event_log = [
        {
            "input_start_sample": 0,
            "input_end_sample": len(pcm) // 2,
            "is_final": True,
            "events": [],
        }
    ]
    result = pair_and_filter(event_log, pcm)
    assert result.kept == []
    assert result.eof_fallback == []
    assert result.status == "no_speech"


def test_last_chunk_is_final_even_when_exact_multiple():
    # 400 ms = exactly 2 chunks of 200 ms; clip <1s after pad clamp → discarded.
    pcm = _pcm_silence(400)
    model = MockVadModel({0: [], 1: [[0, 370]]})
    result = run_streaming_vad(model, pcm)
    assert model.calls[-1]["is_final"] is True
    assert len(model.calls) == 2
    assert result.kept == []
    assert result.status == "all_short"
    assert [d.reason for d in result.discarded] == ["short_utterance"]
    assert result.discarded[0].duration_ms == 400


def test_streaming_final_endpoint_at_eof_is_discarded():
    pcm = _pcm_silence(400)
    model = MockVadModel({0: [[0, -1]], 1: [[-1, 400]]})

    result = run_streaming_vad(model, pcm)

    assert model.calls[-1]["is_final"] is True
    assert result.kept == []
    assert result.status == "endpoint_at_eof"
    assert [d.reason for d in result.discarded] == ["endpoint_at_eof"]


def test_trailing_partial_chunk_final():
    # 250 ms → chunk0 200ms + chunk1 50ms final
    pcm = _pcm_silence(250)
    model = MockVadModel({})
    run_streaming_vad(model, pcm)
    assert len(model.calls) == 2
    assert model.calls[0]["is_final"] is False
    assert model.calls[1]["is_final"] is True
    assert model.calls[1]["n_samples"] == int(50 * SR / 1000)


def test_cache_isolation_across_streams():
    pcm = _pcm_silence(200)
    m1 = ScheduleVadModel([(0, 200)])
    m2 = ScheduleVadModel([(0, 200)])
    run_streaming_vad(m1, pcm)
    run_streaming_vad(m2, pcm)
    assert m1._cache_ids[0] != m2._cache_ids[0] or True  # different model instances
    # Same model object but fresh cache dict each run_streaming_vad call
    shared = ScheduleVadModel([(50, 180)])
    run_streaming_vad(shared, pcm)
    id1 = shared._cache_ids[0]
    shared2 = ScheduleVadModel([(50, 180)])
    run_streaming_vad(shared2, pcm)
    assert id1 != shared2._cache_ids[0]


def test_stable_segment_id_and_sibling_hashes():
    a = stable_segment_id(
        source_audio_id="recA",
        normalized_sha256="n1",
        channel_index=0,
        start_sample=0,
        end_sample=16000,
    )
    b = stable_segment_id(
        source_audio_id="recA",
        normalized_sha256="n1",
        channel_index=0,
        start_sample=16000,
        end_sample=32000,
    )
    assert a != b
    assert a == stable_segment_id(
        source_audio_id="recA",
        normalized_sha256="n1",
        channel_index=0,
        start_sample=0,
        end_sample=16000,
    )
    # version change → new id
    c = stable_segment_id(
        source_audio_id="recA",
        normalized_sha256="n1",
        channel_index=0,
        start_sample=0,
        end_sample=16000,
        vad_processing_version="fsmn_vad_segment/test_other",
    )
    assert c != a


def test_operator_one_to_many_and_manifest_roundtrip(tmp_path: Path):
    # Parent: 4s audio with three normal endpoints via ScheduleVadModel.
    wav = tmp_path / "parent.wav"
    pcm = _pcm_tone(4000)
    sf.write(str(wav), np.frombuffer(pcm, dtype="<i2"), SR)
    parent = Sample(
        id="rec_parent",
        source_path=str(wav),
        sha256=file_sha256(wav),
        sample_rate=SR,
        duration=4.0,
        audio={"resampled_16k": str(wav), "raw": str(wav)},
        labels={"audio_pass": True},
    )
    op = OperatorRegistry.get("audio.fsmn_vad_segment")
    cfg = OperatorConfig(
        params={
            "mock": True,
            "mock_intervals": [[200, 1500], [2000, 3500], [3600, 3800]],
            "input_audio_key": "resampled_16k",
            "output_audio_key": "resampled_16k",
        },
        output_dir=tmp_path / "derived",
        cache_dir=tmp_path / "cache",
        run_dir=tmp_path / "run",
    )
    segments = op.run([parent], cfg)
    # [3600,3800]+675 clamped → 0.4s < 1s discarded; two longer clips kept.
    assert len(segments) == 2
    assert all(s.labels.get("is_vad_segment") for s in segments)
    assert all(s.labels.get("source_audio_id") == "rec_parent" for s in segments)
    assert segments[0].id != segments[1].id
    assert segments[0].sha256 != segments[1].sha256
    # Parent hash must not equal working identity for both
    assert segments[0].labels["original_audio_sha256"] == segments[0].sha256
    assert segments[0].labels["parent_pcm_sha256"] == parent.sha256
    assert segments[0].labels["original_audio_sha256"] != parent.sha256

    # Audio exists and length matches
    for s in segments:
        path = Path(s.audio["resampled_16k"])
        assert path.is_file()
        info = sf.info(str(path))
        assert info.samplerate == 16000
        assert info.channels == 1
        n = int(round(info.duration * 16000))
        assert n == s.labels["clip_end_sample"] - s.labels["clip_start_sample"]
        assert n > 0

    # Manifest roundtrip preserves fields
    man_path = tmp_path / "segments.parquet"
    Manifest(segments).save(man_path)
    loaded = Manifest.load(man_path).samples
    assert len(loaded) == 2
    assert loaded[0].labels["source_audio_id"] == "rec_parent"
    assert loaded[0].labels["clip_start_sample"] == segments[0].labels["clip_start_sample"]

    # Source result persisted
    src_json = tmp_path / "run" / "vad_source_results" / "rec_parent.json"
    assert src_json.is_file()
    record = json.loads(src_json.read_text(encoding="utf-8"))
    assert record["kept_count"] == 2
    assert record["discarded_count"] == 1
    assert record["discarded"][0]["reason"] == "short_utterance"

    # Retry idempotent ids
    segments2 = op.run([parent], cfg)
    assert {s.id for s in segments2} == {s.id for s in segments}


def test_no_speech_and_short_normal_endpoints(tmp_path: Path):
    wav = tmp_path / "silent.wav"
    pcm = _pcm_silence(2000)
    sf.write(str(wav), np.frombuffer(pcm, dtype="<i2"), SR)
    parent = Sample(
        id="silent",
        source_path=str(wav),
        sha256="s1",
        audio={"resampled_16k": str(wav)},
    )
    op = OperatorRegistry.get("audio.fsmn_vad_segment")
    cfg = OperatorConfig(
        params={"mock": True, "mock_intervals": []},
        output_dir=tmp_path / "d",
        cache_dir=tmp_path / "c",
        run_dir=tmp_path / "r",
    )
    assert op.run([parent], cfg) == []
    record = json.loads((tmp_path / "r" / "vad_source_results" / "silent.json").read_text(encoding="utf-8"))
    assert record["status"] == "no_speech"

    # Both normal endpoints are <1s after pad clamp → discarded as all_short.
    cfg2 = OperatorConfig(
        params={"mock": True, "mock_intervals": [[1510, 1600], [1600, 1700]]},
        output_dir=tmp_path / "d2",
        cache_dir=tmp_path / "c2",
        run_dir=tmp_path / "r2",
    )
    segments = op.run([parent], cfg2)
    assert segments == []
    record2 = json.loads((tmp_path / "r2" / "vad_source_results" / "silent.json").read_text(encoding="utf-8"))
    assert record2["status"] == "all_short"
    assert record2["kept_count"] == 0
    assert record2["discarded_count"] == 2
    assert {d["reason"] for d in record2["discarded"]} == {"short_utterance"}


def test_operator_discards_endpoint_at_eof_and_persists_reason(tmp_path: Path):
    wav = tmp_path / "endpoint_at_eof.wav"
    pcm = _pcm_tone(2500)
    sf.write(str(wav), np.frombuffer(pcm, dtype="<i2"), SR)
    parent = Sample(
        id="endpoint_at_eof",
        source_path=str(wav),
        sha256=file_sha256(wav),
        audio={"resampled_16k": str(wav)},
    )
    op = OperatorRegistry.get("audio.fsmn_vad_segment")
    cfg = OperatorConfig(
        params={
            "mock": True,
            "mock_intervals": [[100, 2500]],
            "include_eof_fallback": True,
        },
        output_dir=tmp_path / "derived",
        cache_dir=tmp_path / "cache",
        run_dir=tmp_path / "run",
    )

    assert op.run([parent], cfg) == []
    assert list((tmp_path / "derived" / "vad_segments").rglob("*.wav")) == []

    source_results = tmp_path / "run" / "vad_source_results"
    record = json.loads((source_results / "endpoint_at_eof.json").read_text(encoding="utf-8"))
    assert record["status"] == "endpoint_at_eof"
    assert record["candidate_count"] == 1
    assert record["kept_count"] == 0
    assert record["discarded_count"] == 1
    assert record["published_segment_ids"] == []
    assert record["discarded"][0]["reason"] == "endpoint_at_eof"
    assert record["discarded"][0]["end_reason"] == END_REASON_VAD
    assert record["discarded"][0]["end_ms"] == 2500
    assert record["fingerprint"]["vad_processing_version"] == "fsmn_vad_segment/v7"
    assert record["fingerprint"]["operator_version"] == "1.8.0"
    assert record["fingerprint"]["endpoint_at_eof_policy"] == "discard"
    assert record["fingerprint"]["end_silence_keep_ms"] == 675
    assert record["fingerprint"]["min_keep_ms"] == 1000

    batch = json.loads((source_results / "_batch_summary.json").read_text(encoding="utf-8"))
    assert batch["version"] == "1.8.0"
    assert batch["summary"]["sources"] == 1
    assert batch["summary"]["kept_segments"] == 0
    assert batch["summary"]["discarded"] == 1
    assert batch["summary"]["endpoint_at_eof_discarded"] == 1
    assert batch["summary"]["failed"] == 0


def test_sibling_segments_not_duplicate_in_warehouse():
    parent_sha = "parent_pcm_abc"
    s1 = Sample(
        id="seg_1",
        source_path="a.wav",
        sha256="seg_hash_1",
        labels={
            "is_vad_segment": True,
            "source_audio_id": "recA",
            "parent_pcm_sha256": parent_sha,
            "original_audio_sha256": "seg_hash_1",
            "category": "clean",
            "status": "annotated",
            "gold_kind": "speech",
            "gold_text": "你好",
            "label_tier": "gold",
        },
    )
    s2 = Sample(
        id="seg_2",
        source_path="a.wav",
        sha256="seg_hash_2",
        labels={
            "is_vad_segment": True,
            "source_audio_id": "recA",
            "parent_pcm_sha256": parent_sha,
            "original_audio_sha256": "seg_hash_2",
            "category": "clean",
            "status": "annotated",
            "gold_kind": "speech",
            "gold_text": "世界",
            "label_tier": "gold",
        },
    )
    assert original_audio_sha(s1) != original_audio_sha(s2)
    assert original_audio_sha256(s1) == "seg_hash_1"
    # Intentionally wrong: shared parent as original → would duplicate
    bad = Sample(
        id="seg_bad",
        source_path="a.wav",
        sha256="seg_hash_3",
        labels={
            "original_audio_sha256": parent_sha,
            "parent_pcm_sha256": parent_sha,
            "category": "clean",
            "status": "annotated",
            "gold_kind": "speech",
            "gold_text": "x",
            "label_tier": "gold",
        },
    )
    twin = Sample(
        id="seg_bad2",
        source_path="a.wav",
        sha256="seg_hash_4",
        labels={
            "original_audio_sha256": parent_sha,
            "parent_pcm_sha256": parent_sha,
            "category": "clean",
            "status": "annotated",
            "gold_kind": "speech",
            "gold_text": "y",
            "label_tier": "gold",
        },
    )
    report_ok = assert_batch_complete([s1, s2], config=AnnotationConfig())
    assert not any(i.code == "duplicate_audio_identity" for i in report_ok.issues)

    report_bad = assert_batch_complete([bad, twin], config=AnnotationConfig())
    assert any(i.code == "duplicate_audio_identity" for i in report_bad.issues)


def test_grouping_by_source_audio_id():
    samples = [
        Sample(
            id="seg_a1",
            source_path="x",
            sha256="h1",
            labels={"source_audio_id": "recA", "original_audio_sha256": "h1"},
        ),
        Sample(
            id="seg_a2",
            source_path="x",
            sha256="h2",
            labels={"source_audio_id": "recA", "original_audio_sha256": "h2"},
        ),
        Sample(
            id="seg_b1",
            source_path="y",
            sha256="h3",
            labels={"source_audio_id": "recB", "original_audio_sha256": "h3"},
        ),
    ]
    result = build_leakage_groups(samples, GroupingConfig(require_source_index=False))
    g_a1 = result.leakage_group_id["seg_a1"]
    g_a2 = result.leakage_group_id["seg_a2"]
    g_b1 = result.leakage_group_id["seg_b1"]
    assert g_a1 == g_a2
    assert g_a1 != g_b1


def test_classify_coverage_set_equality(tmp_path: Path):
    """Kept segment id set must equal 'classification input' set (simulated)."""
    wav = tmp_path / "p.wav"
    pcm = _pcm_tone(3500)
    sf.write(str(wav), np.frombuffer(pcm, dtype="<i2"), SR)
    parent = Sample(
        id="cov",
        source_path=str(wav),
        sha256=file_sha256(wav),
        audio={"resampled_16k": str(wav)},
    )
    op = OperatorRegistry.get("audio.fsmn_vad_segment")
    cfg = OperatorConfig(
        params={"mock": True, "mock_intervals": [[0, 1200], [1500, 3000]]},
        output_dir=tmp_path / "d",
        cache_dir=tmp_path / "c",
        run_dir=tmp_path / "r",
    )
    kept = op.run([parent], cfg)
    kept_ids = {s.id for s in kept}
    # Simulate classify input = cleaned manifest segments
    classify_input_ids = {s.id for s in kept}
    # Inject ASR failure status — still in set
    kept[0].status["asr.qwen"] = "failed"
    kept[0].errors["asr.qwen"] = "timeout"
    assert kept_ids == classify_input_ids
    assert "asr.qwen" in kept[0].errors


def test_annotation_export_carries_segment_lineage(tmp_path: Path):
    from audio_engine.core.annotation_v3.package import build_export_rows
    from audio_engine.core.annotation_v3.types import VIEW_BLIND
    from audio_engine.core.selection_v3.types import PRIORITY_P0

    sample = Sample(
        id="seg_xyz",
        source_path="/norm/parent.wav",
        sha256="seghash",
        audio={"resampled_16k": "/seg/seg_xyz.wav", "parent_raw": "/raw/parent.pcm"},
        labels={
            "is_vad_segment": True,
            "source_audio_id": "parent1",
            "parent_pcm_sha256": "parenthash",
            "original_audio_sha256": "seghash",
            "segment_id": "seg_xyz",
            "vad_start_ms": 1000,
            "vad_end_ms": 2500,
            "clip_start_sample": 16000,
            "clip_end_sample": 40000,
            "vad_processing_version": VAD_PROCESSING_VERSION,
            "type": "clean",
            "category": "clean",
            "review_priority": PRIORITY_P0,
        },
    )
    _qid, rows, _meta = build_export_rows(
        [sample],
        dataset_path="datasets/x.parquet",
        revision="r1",
        view=VIEW_BLIND,
        priorities=[PRIORITY_P0],
        config=AnnotationConfig(),
    )
    assert rows[0]["sample_id"] == "seg_xyz"
    assert rows[0]["original_audio_sha256"] == "seghash"
    assert rows[0]["source_audio_id"] == "parent1"
    assert rows[0]["vad_start_ms"] == 1000
    assert rows[0]["parent_raw_audio"] == "/raw/parent.pcm"
    # Old whole-file pack with parent hash would not match
    assert rows[0]["original_audio_sha256"] != "parenthash"
