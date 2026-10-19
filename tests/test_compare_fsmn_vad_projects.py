"""Diagnostic comparison and actual sibling service-method integration (no ASR)."""
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import wave

import numpy as np
import pytest

from audio_engine.core.fsmn_vad.adapter import MockVadModel
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/compare_fsmn_vad_projects.py"
spec = importlib.util.spec_from_file_location("compare_vad", SCRIPT)
compare_vad = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare_vad)
SERVICE = Path(os.getenv("VAD_COMPARE_SERVICE_PROJECT", str(ROOT.parent / "ph_ai_model_asr_rt_cpu")))


def test_same_pcm_from_wav_and_raw_and_reject_wrong_rate(tmp_path):
    pcm = np.arange(1600, dtype="<i2").tobytes()
    raw, wav = tmp_path / "a.pcm", tmp_path / "a.wav"
    raw.write_bytes(pcm)
    with wave.open(str(wav), "wb") as out:
        out.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        out.writeframes(pcm)
    assert compare_vad.read_audio(raw) == compare_vad.read_audio(wav) == pcm
    with wave.open(str(wav), "wb") as out:
        out.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
        out.writeframes(pcm)
    with pytest.raises(ValueError, match="16000 Hz"):
        compare_vad.read_audio(wav)


def test_comparison_does_not_shift_matches_after_a_discard():
    def clip(start, end):
        return compare_vad.clip_info(b"\0\0" * (end - start), start, end, end / 16)
    engine = {"status": "ok", "clips": [clip(16000, 50000)]}
    service = {"status": "ok", "clips": [clip(1600, 8000), clip(16000, 40000)]}
    result = compare_vad.compare(engine, service)
    assert result["clip_matches"][0]["service_index"] == 1
    assert result["clip_matches"][0]["engine_minus_service_end_ms"] == 625
    assert result["unmatched_service_indices"] == [0]


def test_fenshen_overlay_keeps_source_unchanged_and_requires_config(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for name in ("model.pt", "am.mvn", "configuration.json"):
        (source / name).write_bytes(b"original asset")
    original = b"model_conf:\n  max_end_silence_time: 800\n"
    (source / "config.yaml").write_bytes(original)
    config = ROOT / "configs/vad/fenshen_vad_config.yaml"
    report = {}
    runtime = compare_vad.prepare_fenshen_runtime(source, config, tmp_path / "runtime", report)
    assert (source / "config.yaml").read_bytes() == original
    assert (runtime / "config.yaml").read_bytes() == config.read_bytes()
    for name in ("model.pt", "am.mvn", "configuration.json"):
        assert (runtime / name).read_bytes() == (source / name).read_bytes()
    with pytest.raises(FileNotFoundError, match="no default-config fallback"):
        compare_vad.prepare_fenshen_runtime(source, tmp_path / "missing.yaml", tmp_path / "other", {})
    with pytest.raises(ValueError, match="Not the required fenshen"):
        compare_vad.prepare_fenshen_runtime(source, source / "config.yaml", tmp_path / "other", {})


def test_loaded_model_options_cannot_silently_fall_back_to_defaults():
    config = yaml.safe_load((ROOT / "configs/vad/fenshen_vad_config.yaml").read_text(encoding="utf-8"))
    loaded = dict(config["model_conf"])
    compare_vad.validate_loaded_config(config, loaded)
    loaded["speech_noise_thres"] = 0.6
    with pytest.raises(RuntimeError, match="speech_noise_thres"):
        compare_vad.validate_loaded_config(config, loaded)


@pytest.mark.skipif(not (SERVICE / "app/core/vad_asr_core_new.py").is_file(), reason="Sibling service checkout required")
@pytest.mark.parametrize("duration_ms,events,end_sample,reported_start", [
    (6000, {5: [[1000, -1]], 16: [[-1, 3000]]}, 48000, 900),
    (3350, {5: [[1000, -1]], 16: [[-1, 3000]]}, 48000, 900),
    (4000, {19: [[1000, 3060]]}, 64000, 1000),
    (1000, {2: [[400, -1]]}, 16000, 400),
])
def test_actual_service_clipping_and_eof(duration_ms, events, end_sample, reported_start):
    model = MockVadModel(events)
    report = {"calls": [], "clips": [], "internal_endpoints": []}
    pcm = np.arange(duration_ms * 16, dtype="<i2").tobytes()
    payloads = asyncio.run(compare_vad.replay_service(
        SERVICE, {"send_end_result": True, "firered_vad_switch": False}, model, pcm, 100,
        report, compare_vad.QuietLog()))
    assert len(payloads) == len(report["clips"]) == 1
    clip = report["clips"][0]
    assert clip["clip_end_sample"] == end_sample
    assert clip["reported_start_ms"] == reported_start
    assert payloads[0] == pcm[clip["clip_start_sample"] * 2:end_sample * 2]
    if duration_ms == 1000:
        assert clip["end_reason"] == "eof_fallback"
        assert clip["model_end_ms"] is None
        assert not any(c["is_final"] for c in report["calls"])


def test_parent_writes_combined_json_when_both_workers_fail(tmp_path):
    pcm = tmp_path / "a.pcm"
    pcm.write_bytes(b"\0\0" * 3200)
    output = tmp_path / "result.json"
    completed = subprocess.run([sys.executable, str(SCRIPT), "--engine-project", str(tmp_path),
                                "--service-project", str(tmp_path), "--audio", str(pcm),
                                "--output", str(output)], capture_output=True, text=True)
    assert completed.returncode == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "error"
    assert set(report["runs"]) == {"engine", "service"}
    assert all(run["status"] == "error" for run in report["runs"].values())
    assert "error" in report["runs"]["engine"]


@pytest.mark.skipif(not (SERVICE / "app/core/vad_asr_core_new.py").is_file(), reason="Sibling service checkout required")
def test_full_two_process_comparison_with_stub_acoustic_model(tmp_path):
    package = tmp_path / "packages/funasr"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('''
from types import SimpleNamespace
from pathlib import Path
import yaml
class FsmnVADStreaming:
    def __init__(self, opts):
        self.vad_opts = SimpleNamespace(**opts)
class AutoModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        config = yaml.safe_load((Path(kwargs["model"]) / "config.yaml").read_text())
        self.model = FsmnVADStreaming(config["model_conf"])
    def generate(self, input, cache, **kwargs):
        before = cache.get("n", 0)
        after = before + len(input)
        cache["n"] = after
        events = []
        if before < 19200 <= after:
            events.append([1000, -1])
        if before < 51200 <= after:
            events.append([-1, 3000])
        return [{"value": events}]
''', encoding="utf-8")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    for name in ("model.pt", "am.mvn", "configuration.json"):
        (model_dir / name).write_bytes(b"stub")
    (model_dir / "config.yaml").write_text("model: FsmnVADStreaming\n", encoding="utf-8")
    app_config = tmp_path / "service.yaml"
    app_config.write_text("firered_vad_switch: false\nsend_end_result: true\n", encoding="utf-8")
    audio = tmp_path / "audio.pcm"
    audio.write_bytes(np.arange(3350 * 16, dtype="<i2").tobytes())
    output = tmp_path / "comparison.json"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(package.parent) + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run([sys.executable, str(SCRIPT), "--engine-project", str(ROOT),
                                "--service-project", str(SERVICE), "--audio", str(audio),
                                "--engine-model-dir", str(model_dir), "--service-model-dir", str(model_dir),
                                "--service-config", str(app_config), "--output", str(output), "--save-clips"],
                               env=env, capture_output=True, text=True)
    report = json.loads(output.read_text(encoding="utf-8"))
    assert completed.returncode == 0, report
    assert "[engine] fenshen verified: max_end_silence_time=1100, speech_noise_thres=0.85" in completed.stdout
    assert "[service] fenshen verified: max_end_silence_time=1100, speech_noise_thres=0.85" in completed.stdout
    assert report["status"] == "ok"
    assert report["comparison"]["raw_event_sequence_equal"]
    match = report["comparison"]["clip_matches"][0]
    assert match["engine_minus_service_end_ms"] == 350
    assert match["engine_boundary_clamped"]
    for role in ("engine", "service"):
        run = report["runs"][role]
        assert run["fenshen_config"]["loaded_model_conf_verified"]
        assert run["source_model_dir"] == str(model_dir)
        assert Path(run["model_dir"]) != model_dir
        assert run["effective_model_yaml"]["model_conf"]["max_end_silence_time"] == 1100
        assert run["detector"]["vad_opts_loaded"]["speech_noise_thres"] == 0.85
        assert run["detector"]["vad_opts_loaded"]["max_single_segment_time"] == 10000
        clip = report["runs"][role]["clips"][0]
        assert Path(clip["wav_path"]).is_file()
        assert compare_vad.read_audio(clip["wav_path"]) == audio.read_bytes()[clip["clip_start_sample"] * 2:clip["clip_end_sample"] * 2]
    assert (model_dir / "config.yaml").read_text() == "model: FsmnVADStreaming\n"
