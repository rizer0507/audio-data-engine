"""050: mixed-scene ingest, int16 scale, cache isolation, dual VAD branches."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml

import audio_engine.operators  # noqa: F401
from audio_engine.core.audio_scene import (
    SCENE_FENSHEN,
    SCENE_SHENPI_INTERVIEW,
    SCENE_SHENPI_OUTBOUND,
    SCENE_UNKNOWN,
)
from audio_engine.core.fsmn_vad.publish import read_pcm16le_mono
from audio_engine.core.manifest import Manifest, file_sha256
from audio_engine.core.operator import OperatorConfig
from audio_engine.core.registry import OperatorRegistry
from audio_engine.core.sample import Sample
from audio_engine.core.source_naming import apply_source_name_to_single_pipeline
from audio_engine.operators.audio.shenpi_vad_segment import timestamps_to_kept


def _sine_int16(sr: int, seconds: float = 1.0, amp: float = 9800.0, freq: float = 440.0) -> np.ndarray:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.int16)


def _write_pcm(path: Path, samples: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples.astype("<i2").tofile(path)


def _cfg(tmp_path: Path, **params) -> OperatorConfig:
    return OperatorConfig(
        params=params,
        output_dir=tmp_path / "derived",
        cache_dir=tmp_path / "cache",
        run_dir=tmp_path / "run",
    )


def test_int16_resample_no_mass_clipping(tmp_path: Path):
    pcm_path = tmp_path / "raw8.pcm"
    src = _sine_int16(8000, amp=9800.0)
    _write_pcm(pcm_path, src)
    sample = Sample(
        id="shepi_ai-x-V1",
        source_path=str(pcm_path),
        sha256="a" * 64,
        audio={"raw": str(pcm_path)},
        labels={"audio_scene": SCENE_SHENPI_INTERVIEW, "source_sample_rate": 8000},
    )
    op = OperatorRegistry.get("audio.pcm_to_wav")
    result = op.process(
        sample,
        _cfg(tmp_path, sample_rate=16000, source_sample_rate=16000, input_audio_key="raw", output_audio_key="pcm_wav"),
    )
    out = Path(result.sample.audio["pcm_wav"])
    data, sr = sf.read(str(out), dtype="int16")
    assert sr == 16000
    assert len(data) == pytest.approx(16000, abs=2)
    saturated = int(np.sum(np.abs(data) >= 32767))
    assert saturated < 50
    assert result.sample.labels["source_sample_rate"] == 8000
    assert result.sample.duration == pytest.approx(1.0, abs=0.02)


def test_same_sha_different_id_cache_isolated(tmp_path: Path):
    pcm_bytes = _sine_int16(8000).tobytes()
    paths = []
    for name in ("shepi_ai-a-V1", "fenshen-twin-1"):
        path = tmp_path / f"{name}.pcm"
        path.write_bytes(pcm_bytes)
        paths.append(path)
    sha = file_sha256(paths[0])
    op = OperatorRegistry.get("audio.pcm_to_wav")
    cfg = _cfg(
        tmp_path,
        sample_rate=16000,
        source_sample_rate=16000,
        input_audio_key="raw",
        output_audio_key="pcm_wav",
    )
    s1 = Sample(
        id="shepi_ai-a-V1",
        source_path=str(paths[0]),
        sha256=sha,
        audio={"raw": str(paths[0])},
        labels={"audio_scene": SCENE_SHENPI_INTERVIEW, "source_sample_rate": 8000},
    )
    s2 = Sample(
        id="fenshen-twin-1",
        source_path=str(paths[1]),
        sha256=sha,
        audio={"raw": str(paths[1])},
        labels={"audio_scene": SCENE_FENSHEN, "source_sample_rate": 16000},
    )
    r1 = op.process(s1, cfg)
    r2 = op.process(s2, cfg)
    assert r1.sample.audio["pcm_wav"] != r2.sample.audio["pcm_wav"]
    assert Path(r1.sample.audio["pcm_wav"]).is_file()
    assert Path(r2.sample.audio["pcm_wav"]).is_file()
    d1 = sf.info(r1.sample.audio["pcm_wav"]).duration
    d2 = sf.info(r2.sample.audio["pcm_wav"]).duration
    assert d1 == pytest.approx(1.0, abs=0.02)
    # Same bytes interpreted as 16 kHz → half duration.
    assert d2 == pytest.approx(0.5, abs=0.02)


def test_resolve_audio_scene_operator_mixed_and_rejects(tmp_path: Path):
    root = tmp_path / "raw"
    root.mkdir()
    cases = {
        "fenshen-276cc46a-bf3387be.pcm": (16000, SCENE_FENSHEN),
        "shenpi_ai-c24b27s-V2610091.pcm": (8000, SCENE_SHENPI_INTERVIEW),
        "shepi_ai-7jsn7288-8j22j987.pcm": (16000, SCENE_SHENPI_OUTBOUND),
        "unknown_prefix_demo.pcm": (16000, SCENE_UNKNOWN),
        "shenpi_ai-badname.pcm": (16000, None),  # rejected
    }
    for name, (sr, _) in cases.items():
        _write_pcm(root / name, _sine_int16(sr))
    # RIFF disguised as .pcm under approval prefix → reject
    wav = tmp_path / "tmp.wav"
    sf.write(str(wav), _sine_int16(8000).astype(np.float32) / 32768.0, 8000)
    (root / "shepi_ai-h726h9999-Ph829888.pcm").write_bytes(wav.read_bytes())

    scanned = Manifest.ingest(str(root), recursive=False)
    op = OperatorRegistry.get("ingest.resolve_audio_scene")
    kept = op.run(scanned.samples, _cfg(tmp_path, drop_rejected=True))
    by_id = {s.id: s for s in kept}
    assert "shenpi_ai-badname" not in by_id
    assert "shepi_ai-h726h9999-Ph829888" not in by_id
    assert by_id["fenshen-276cc46a-bf3387be"].labels["audio_scene"] == SCENE_FENSHEN
    assert by_id["shenpi_ai-c24b27s-V2610091"].labels["source_sample_rate"] == 8000
    assert by_id["shepi_ai-7jsn7288-8j22j987"].labels["audio_scene"] == SCENE_SHENPI_OUTBOUND
    assert by_id["unknown_prefix_demo"].labels["audio_scene"] == SCENE_UNKNOWN


def test_timestamps_to_kept_local_16k_slice():
    pcm = (_sine_int16(16000)).tobytes()
    kept = timestamps_to_kept([[250, 750]], pcm, sample_rate=16000)
    assert len(kept) == 1
    assert kept[0].start_sample == 4000
    assert kept[0].end_sample == 12000
    assert len(kept[0].pcm16le) == 8000 * 2
    assert kept[0].pcm16le == pcm[4000 * 2 : 12000 * 2]


class _VadUploadHandler(BaseHTTPRequestHandler):
    seen: list[dict] = []

    def log_message(self, format, *args):  # noqa: A003
        return

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        content_type = self.headers.get("Content-Type", "")
        # Minimal multipart parse for tests.
        boundary = content_type.split("boundary=")[-1].encode()
        parts = body.split(b"--" + boundary)
        fields: dict[str, str] = {}
        file_bytes = b""
        for part in parts:
            if b"Content-Disposition" not in part:
                continue
            header, _, payload = part.partition(b"\r\n\r\n")
            payload = payload.rstrip(b"\r\n")
            header_text = header.decode("utf-8", errors="replace")
            if 'name="file"' in header_text:
                file_bytes = payload
            else:
                for token in header_text.split(";"):
                    token = token.strip()
                    if token.startswith('name="'):
                        name = token[6:-1]
                        fields[name] = payload.decode("utf-8")
        import hashlib

        record = {
            "fields": fields,
            "file_bytes": file_bytes,
            "file_sha": hashlib.sha256(file_bytes).hexdigest(),
        }
        self.seen.append(record)
        sample_rate = int(fields.get("sample_rate", "0"))
        payload_out = {
            "id": fields.get("id"),
            "status": "ok",
            "mode": fields.get("mode", "stream"),
            "sample_rate": sample_rate,
            "model_sample_rate": 16000,
            "timestamps": [[250, 750]],
            "num_segments": 1,
            "vad_config": json.loads(fields.get("vad_config") or "{}"),
            "elapsed_ms": 1.0,
        }
        raw = json.dumps(payload_out).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def test_shenpi_vad_http_stub_and_scene_select(tmp_path: Path):
    root = tmp_path / "raw"
    root.mkdir()
    files = {
        "fenshen-keep.pcm": (16000, SCENE_FENSHEN),
        "shenpi_ai-c24b27s-V2610091.pcm": (8000, SCENE_SHENPI_INTERVIEW),
        "shepi_ai-7jsn7288-8j22j987.pcm": (16000, SCENE_SHENPI_OUTBOUND),
        "unknown_skip.pcm": (16000, SCENE_UNKNOWN),
    }
    parents: list[Sample] = []
    for name, (sr, scene) in files.items():
        path = root / name
        pcm = _sine_int16(sr)
        _write_pcm(path, pcm)
        # Build normalized 16k parent WAV manually.
        if sr == 16000:
            wav_data = pcm
        else:
            # simple upsample by repeat for test fixture (exact 2x)
            wav_data = np.repeat(pcm, 2)
        wav_path = tmp_path / "norm" / f"{path.stem}.wav"
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(wav_path), wav_data, 16000, subtype="PCM_16")
        parents.append(
            Sample(
                id=path.stem,
                source_path=str(path),
                sha256=file_sha256(path),
                sample_rate=16000,
                channels=1,
                duration=1.0,
                audio={"raw": str(path), "resampled_16k": str(wav_path)},
                labels={
                    "audio_scene": scene,
                    "source_sample_rate": sr,
                    "normalized_sample_rate": 16000,
                },
            )
        )

    handler = _VadUploadHandler
    handler.seen = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        op = OperatorRegistry.get("audio.shenpi_vad_segment")
        cfg = OperatorConfig(
            params={
                "api_base": f"http://127.0.0.1:{port}",
                "profiles_path": "configs/vad/shenpi_vad_profiles.yaml",
                "source_workers": 1,
                "max_in_flight_sources": 1,
                "retries": 0,
            },
            output_dir=tmp_path / "seg_out",
            cache_dir=tmp_path / "cache",
            run_dir=tmp_path / "run_shenpi",
        )
        segments = op.run(parents, cfg)
    finally:
        server.shutdown()

    assert len(handler.seen) == 2
    rates = sorted(int(item["fields"]["sample_rate"]) for item in handler.seen)
    assert rates == [8000, 16000]
    ids = {item["fields"]["id"] for item in handler.seen}
    assert ids == {"shenpi_ai-c24b27s-V2610091", "shepi_ai-7jsn7288-8j22j987"}
    assert len(segments) == 2
    for seg in segments:
        assert seg.labels["audio_scene"] in {
            SCENE_SHENPI_INTERVIEW,
            SCENE_SHENPI_OUTBOUND,
        }
        parent = next(p for p in parents if p.id == seg.labels["source_audio_id"])
        parent_pcm, _, _ = read_pcm16le_mono(Path(parent.audio["resampled_16k"]))
        data, sr = sf.read(seg.audio["resampled_16k"], dtype="int16")
        assert sr == 16000
        assert data.tobytes() == parent_pcm[4000 * 2 : 12000 * 2]

    # Twin branch must not see approval / unknown.
    fsmn = OperatorRegistry.get("audio.fsmn_vad_segment")
    twin_cfg = OperatorConfig(
        params={
            "mock": True,
            "mock_intervals": [[250, 750]],
            "source_workers": 1,
            "max_in_flight_sources": 1,
        },
        output_dir=tmp_path / "seg_twin",
        cache_dir=tmp_path / "cache",
        run_dir=tmp_path / "run_twin",
        mock=True,
    )
    twin_segments = fsmn.run(parents, twin_cfg)
    assert twin_segments
    assert all(s.labels.get("audio_scene") == SCENE_FENSHEN for s in twin_segments)
    assert all(s.labels["source_audio_id"] == "fenshen-keep" for s in twin_segments)


def test_source_naming_shenpi_output(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ingested = tmp_path / "datasets" / "stage1" / "ingested" / "demo" / "ingested_demo.parquet"
    ingested.parent.mkdir(parents=True, exist_ok=True)
    Manifest([]).save(ingested)
    binding = apply_source_name_to_single_pipeline(
        pipeline_name="vad_segment_shenpi_batch",
        steps=[],
        source_name="demo",
    )
    assert Path(binding["input_manifest"]).resolve() == ingested.resolve()
    assert binding["output_manifest"].replace("\\", "/").endswith(
        "segments/demo-shenpi/segments_demo-shenpi.parquet"
    )


def test_profiles_yaml_matches_requirement():
    data = yaml.safe_load(Path("configs/vad/shenpi_vad_profiles.yaml").read_text(encoding="utf-8"))
    assert data["shenpi_interview"]["source_sample_rate"] == 8000
    assert data["shenpi_interview"]["vad_config"]["speech_threshold"] == 0.64
    assert data["shenpi_interview"]["vad_config"]["min_speech_frame"] == 15
    assert data["shenpi_outbound"]["source_sample_rate"] == 16000
    assert data["shenpi_outbound"]["vad_config"]["speech_threshold"] == 0.62
    assert data["shenpi_outbound"]["vad_config"]["min_speech_frame"] == 13


def test_unmarked_shenpi_blocked_on_both_branches(tmp_path: Path):
    sample = Sample(
        id="shenpi_ai-a-V1",
        source_path="x.pcm",
        sha256="b" * 64,
        audio={"raw": "x.pcm", "resampled_16k": "x.wav"},
        labels={},
    )
    fsmn = OperatorRegistry.get("audio.fsmn_vad_segment")
    with pytest.raises(ValueError, match="re-ingested"):
        fsmn.run(
            [sample],
            OperatorConfig(
                params={"mock": True, "source_workers": 1},
                output_dir=tmp_path / "o",
                cache_dir=tmp_path / "c",
                run_dir=tmp_path / "r",
                mock=True,
            ),
        )
    shenpi = OperatorRegistry.get("audio.shenpi_vad_segment")
    with pytest.raises(ValueError, match="re-ingested"):
        shenpi.run(
            [sample],
            OperatorConfig(
                params={"mock": True, "source_workers": 1, "profiles_path": "configs/vad/shenpi_vad_profiles.yaml"},
                output_dir=tmp_path / "o2",
                cache_dir=tmp_path / "c2",
                run_dir=tmp_path / "r2",
                mock=True,
            ),
        )
