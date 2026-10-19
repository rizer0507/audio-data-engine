#!/usr/bin/env python3
"""Compare both projects' real FSMN VAD and clipping without calling ASR.

Copy this file anywhere on the server. Only the workers need numpy, PyYAML,
FunASR and the project dependencies. --help needs only the Python standard library.
Input must be mono, little-endian PCM16 at 16000 Hz (WAV or headerless PCM).
Both models always use their project's full fenshen VAD YAML. Model directories
provide weights/assets only; isolated runtime copies are created beside the JSON.
"""
from __future__ import annotations

import argparse
import ast
import asyncio
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from types import SimpleNamespace
import wave


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temp, path)


def read_audio(path):
    path = Path(path)
    with path.open("rb") as f:
        head = f.read(12)
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        with wave.open(str(path), "rb") as f:
            fmt = (f.getframerate(), f.getnchannels(), f.getsampwidth(), f.getcomptype())
            if fmt != (16000, 1, 2, "NONE"):
                raise ValueError(f"WAV must be 16000 Hz mono PCM16, got {fmt}; normalize once before comparison")
            pcm = f.readframes(f.getnframes())
            if len(pcm) != f.getnframes() * 2:
                raise ValueError("Truncated WAV payload")
    elif path.suffix.lower() in {".pcm", ".raw"}:
        pcm = path.read_bytes()
    else:
        raise ValueError("Input must be PCM16 WAV or headerless .pcm/.raw (assumed 16000 Hz mono)")
    if not pcm or len(pcm) % 2:
        raise ValueError("PCM payload must be nonempty and contain whole int16 samples")
    return pcm


def config_get(data, key, default=None):
    for part in key.split("."):
        if not isinstance(data, dict) or part not in data:
            return default
        data = data[part]
    return data


def absolute(value, root):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def prepare_fenshen_runtime(source, config_path, runtime, report):
    """Copy assets and the complete fenshen YAML without changing the source."""
    import yaml

    if not config_path.is_file():
        raise FileNotFoundError(f"Required fenshen VAD config not found: {config_path}; no default-config fallback")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    required = {"model": "FsmnVADStreaming", "frontend": "WavFrontendOnline",
                "frontend_conf.fs": 16000, "model_conf.sample_rate": 16000,
                "model_conf.max_end_silence_time": 1100,
                "model_conf.max_single_segment_time": 10000,
                "model_conf.sil_to_speech_time_thres": 200,
                "model_conf.speech_to_sil_time_thres": 50,
                "model_conf.speech_noise_thres": 0.85}
    mismatches = {k: {"expected": v, "actual": config_get(data, k)}
                  for k, v in required.items() if config_get(data, k) != v}
    if mismatches:
        raise ValueError(f"Not the required fenshen VAD configuration: {config_path}: {mismatches}")
    assets = ("model.pt", "am.mvn", "configuration.json")
    for name in assets:
        if not (source / name).is_file():
            raise FileNotFoundError(f"Missing model asset: {source / name}")
    report["source_model_dir"] = str(source)
    report["source_config_sha256"] = sha(source / "config.yaml") if (source / "config.yaml").is_file() else None
    report["fenshen_config"] = {"path": str(config_path), "sha256": sha(config_path),
                                 "mode": "required_full_yaml_overlay", "required_values": required}
    runtime.mkdir(parents=True, exist_ok=False)
    for name in assets:
        shutil.copy2(source / name, runtime / name)
    shutil.copyfile(config_path, runtime / "config.yaml")
    if sha(runtime / "config.yaml") != report["fenshen_config"]["sha256"]:
        raise RuntimeError("Runtime config differs from the selected fenshen YAML")
    return runtime


def validate_loaded_config(expected, actual):
    mismatches = {k: {"expected": v, "actual": actual.get(k)}
                  for k, v in expected["model_conf"].items() if actual.get(k) != v}
    if mismatches:
        raise RuntimeError(f"Loaded FSMN options do not match the full fenshen model_conf: {mismatches}")


class QuietLog:
    def __init__(self):
        self.errors = []

    def __getattr__(self, name):
        def log(*args, **kwargs):
            if name in {"error", "exception"}:
                self.errors.append(" ".join(map(str, args)))
        return log


def extract_service_methods(project, config, model, logger):
    """Compile unchanged repository methods, avoiding service/ASR imports and startup."""
    import numpy as np
    from scipy.signal import resample

    path = project / "app/core/vad_asr_core_new.py"
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "WSConnectionNew")
    wanted = {"process_audio_chunk", "process_vad_and_asr", "_pcm_to_float32"}
    methods = [n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in wanted]
    if {n.name for n in methods} != wanted:
        raise RuntimeError("Service methods changed; cannot safely reproduce this checkout")
    namespace = {"np": np, "resample": resample, "time": time, "traceback": traceback,
                 "config": config, "logger": logger, "SAMPLE_RATE": 16000,
                 "ASR_CONTEXT_PROMPT": "", "PUNCTUATION_MAP": str.maketrans({}),
                 "model_manager": SimpleNamespace(get_models=lambda: (model, None))}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def service_model(project, config):
    """Evaluate the actual AutoModel constructor expression in asr_model.py."""
    from funasr import AutoModel

    path = project / "app/core/asr_model.py"
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "AutoModel"]
    if len(calls) != 1:
        raise RuntimeError("Expected exactly one service AutoModel constructor; inspect changed asr_model.py")
    expression = ast.Expression(body=calls[0])
    return eval(compile(expression, str(path), "eval"), {"AutoModel": AutoModel, "config": config})


def runtime_kwargs(model):
    allowed = {"chunk_size", "dynamic_silence", "silence_schedule", "max_end_silence_time",
               "speech_noise_thres", "is_streaming_input", "device", "ncpu", "fs"}
    return {k: v for k, v in getattr(model, "kwargs", {}).items() if k in allowed}


def cache_stats(cache):
    stats = cache.get("stats")
    names = ("max_end_sil_frame_cnt_thresh", "speech_noise_thres",
             "continous_silence_frame_count", "vad_state_machine", "frm_cnt")
    return {k: getattr(stats, k, None) for k in names} if stats is not None else {}


def instrument(model, report, buffer_samples):
    import numpy as np

    original = model.generate
    cursor = 0

    def generate(**kwargs):
        nonlocal cursor
        data = kwargs["input"]
        count = len(data)
        cache = kwargs.get("cache", {})
        entry = {"index": len(report["calls"]), "input_start_sample": cursor,
                 "input_end_sample": cursor + count, "input_samples": count,
                 "submitted_audio_end_ms": (cursor + count) / 16,
                 "buffer_end_ms": buffer_samples() / 16,
                 "is_final": bool(kwargs.get("is_final")), "chunk_size": kwargs.get("chunk_size"),
                 "float32_input_sha256": hashlib.sha256(np.asarray(data, dtype="<f4").tobytes()).hexdigest(),
                 "cache_stats_before": cache_stats(cache)}
        cursor += count
        report["calls"].append(entry)
        try:
            result = original(**kwargs)
            events = result[0].get("value", []) if result else []
            entry["events"] = [[float(a), float(b)] for a, b in events]
            entry["runtime_kwargs"] = runtime_kwargs(model)
            entry["cache_stats_after"] = cache_stats(cache)
            return result
        except Exception:
            entry["error"] = traceback.format_exc()
            raise

    model.generate = generate
    detector = getattr(model, "model", None)
    endpoint = getattr(detector, "OnVoiceEnd", None)
    if callable(endpoint):
        signature = inspect.signature(endpoint)

        def on_end(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            values = bound.arguments
            cache = values.get("cache") or {}
            report["internal_endpoints"].append({
                "call_index": len(report["calls"]) - 1,
                "end_frame": values.get("end_frame"), "fake_result": values.get("fake_result"),
                "is_last_frame": values.get("is_last_frame"), "effective_stats": cache_stats(cache)})
            return endpoint(*args, **kwargs)

        detector.OnVoiceEnd = on_end
    return model


def clip_info(payload, start_sample, end_sample, raw_end_ms=None):
    return {"clip_start_sample": start_sample, "clip_end_sample": end_sample,
            "clip_start_ms": start_sample / 16, "clip_end_ms": end_sample / 16,
            "duration_ms": len(payload) / 32, "model_end_ms": raw_end_ms,
            "pcm_sha256": hashlib.sha256(payload).hexdigest()}


async def replay_service(project, config, model, pcm, packet_ms, report, logger):
    import numpy as np

    methods = extract_service_methods(project, config, model, logger)
    obj = SimpleNamespace(pcm_bytes=b"", pending_start_ms=None, signal="start", vad_cache={},
                          connection_id="vad-compare", request_id="vad-compare", asr_context_prompt="",
                          hotwords=[], emitted=set(), results=[], chunk_num=0, chunk_idx=0,
                          vad_accum=b"", sample_rate=16000, speech=np.array([]),
                          number_processor=SimpleNamespace(to_number=lambda text: text))
    instrument(model, report, lambda: len(obj.pcm_bytes) // 2)
    payloads = []

    async def capture(segment, prompt):
        # Read the actual crop offset from the unchanged service coroutine.
        frame = inspect.currentframe().f_back
        if frame is None or frame.f_code.co_name != "process_vad_and_asr":
            raise RuntimeError("Cannot identify service cropping frame")
        local = frame.f_locals
        if "start_pcm" not in local:
            raise RuntimeError("Service cropping variables changed")
        start_byte = int(local["start_pcm"])
        if start_byte % 2:
            raise RuntimeError("Service produced an unaligned PCM16 offset")
        last_call = report["calls"][-1] if report["calls"] else None
        raw_end = local.get("end") if local.get("events") else None
        if raw_end is not None and raw_end < 0:
            raw_end = None
        item = clip_info(segment, start_byte // 2, (start_byte + len(segment)) // 2, raw_end)
        item["call_index"] = last_call["index"] if last_call else None
        item["buffer_end_ms"] = len(obj.pcm_bytes) / 32
        item["end_reason"] = "eof_fallback" if raw_end is None else "vad_event"
        if segment != pcm[start_byte:start_byte + len(segment)]:
            raise RuntimeError("Service segment does not match source PCM at its recorded offset")
        report["clips"].append(item)
        payloads.append(segment)
        return "vad_compare_stub"

    async def send_result(text, start, end):
        if report["clips"]:
            report["clips"][-1]["reported_start_ms"] = start
            report["clips"][-1]["reported_end_ms"] = end

    async def noop(*args):
        pass

    obj._call_vllm_transcribe = capture
    obj.clean_asr_text = lambda text, prompt: text
    obj.send_result = send_result
    obj.send_partial_result = noop
    obj._pcm_to_float32 = lambda data: methods["_pcm_to_float32"](obj, data)
    obj.process_vad_and_asr = lambda data: methods["process_vad_and_asr"](obj, data)
    packet_bytes = packet_ms * 32
    for left in range(0, len(pcm), packet_bytes):
        await methods["process_audio_chunk"](obj, pcm[left:left + packet_bytes])
    obj.signal = "end"
    if config.get("send_end_result", True):
        await methods["process_audio_chunk"](obj, b"")
    report["application_errors"] = logger.errors
    if logger.errors:
        raise RuntimeError("Service methods logged errors; see application_errors")
    report["pending_start_at_end_ms"] = obj.pending_start_ms
    return payloads


def worker(settings, role, report):
    import yaml

    project = Path(settings[f"{role}_project"])
    os.chdir(project)
    sys.path.insert(0, str(project / "src" if role == "engine" else project))
    pcm = read_audio(settings["audio"])
    report.update({"status": "running", "role": role, "project": str(project), "calls": [],
                   "clips": [], "internal_endpoints": [], "python": sys.executable,
                   "python_version": sys.version, "input_pcm_sha256": hashlib.sha256(pcm).hexdigest()})
    report["dependencies"] = {}
    for name in ("funasr", "torch", "torchaudio", "numpy", "modelscope", "PyYAML"):
        try:
            report["dependencies"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["dependencies"][name] = None
    params = {}
    if role == "engine":
        if settings.get("engine_pipeline"):
            pipeline_path = absolute(settings["engine_pipeline"], project)
            pipeline = yaml.safe_load(pipeline_path.read_text(encoding="utf-8"))
            steps = [s for s in pipeline.get("pipeline", []) if s.get("operator") == "audio.fsmn_vad_segment"]
            if len(steps) != 1:
                raise ValueError("Pipeline must contain exactly one audio.fsmn_vad_segment step")
            params = dict(steps[0].get("params") or {})
            if params.get("mock"):
                raise ValueError("Pipeline enables mock; real-model comparison requires mock=false")
            report["pipeline"] = {"path": str(pipeline_path), "sha256": sha(pipeline_path), "vad_params": params}
        selected = settings.get("engine_model_dir") or params.get("model_runtime_dir") or os.getenv("FSMN_VAD_RUNTIME_DIR") or "var/models/fsmn-vad-fenshen-v1"
        source_files = ["src/audio_engine/core/fsmn_vad/adapter.py", "src/audio_engine/core/fsmn_vad/pairing.py",
                        "src/audio_engine/core/fsmn_vad/types.py", "src/audio_engine/core/fsmn_vad/clip.py"]
    else:
        cfg_path = absolute(settings.get("service_config") or f"app/config/{settings['service_profile']}_{settings['service_scene']}.yaml", project)
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        report["application_config"] = {"path": str(cfg_path), "sha256": sha(cfg_path),
                                       "vad_values": {k: cfg.get(k) for k in ("firered_vad_switch", "vad_eos", "send_end_result", "semantic_judgment", "extension_eos")}}
        if cfg.get("firered_vad_switch", False):
            raise ValueError("Selected service configuration enables FireRed, not FSMN; choose the actual FSMN config")
        selected = settings.get("service_model_dir") or config_get(cfg, "models.vad_model", "fsmn-vad")
        source_files = ["app/core/asr_model.py", "app/core/vad_asr_core_new.py", "app/config/settings.py"]
    source_model_dir = absolute(selected, project)
    default_config = "configs/vad/fenshen_vad_config.yaml" if role == "engine" else "model_configs/fenshen_vad_config.yaml"
    config_path = absolute(settings.get(f"{role}_vad_config") or default_config, project)
    model_dir = prepare_fenshen_runtime(source_model_dir, config_path,
                                       Path(settings["artifact_dir"]) / f"{role}_fenshen_runtime", report)
    report["model_dir"] = str(model_dir)
    report["source_files"] = {f: {"path": str(project / f), "sha256": sha(project / f)} for f in source_files}
    report["assets"] = {}
    for name in ("model.pt", "am.mvn", "config.yaml", "configuration.json"):
        path = model_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing actual runtime asset: {path}; specify --{role}-model-dir")
        report["assets"][name] = {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}
    report["effective_model_yaml"] = yaml.safe_load((model_dir / "config.yaml").read_text(encoding="utf-8"))
    # Runtime options from an optional pipeline must not undo the required YAML.
    for key in ("max_end_silence_time", "speech_noise_thres"):
        value = (params.get("generate_options") or {}).get(key)
        if value is not None and value != report["effective_model_yaml"]["model_conf"][key]:
            raise ValueError(f"generate_options.{key}={value} conflicts with the required fenshen config")
    if role == "engine":
        from audio_engine.core.fsmn_vad.adapter import load_funasr_vad, run_streaming_vad
        from audio_engine.core.fsmn_vad.types import (
            END_SILENCE_KEEP_MS,
            PACKET_MS,
            VAD_PROCESSING_VERSION,
        )
        model = load_funasr_vad(model_dir, ncpu=params.get("ncpu", 1))
        engine_packet_ms = int(params.get("packet_ms") or settings.get("service_packet_ms") or PACKET_MS)
        report["processing"] = {
            "version": VAD_PROCESSING_VERSION,
            "end_silence_keep_ms": END_SILENCE_KEEP_MS,
            "packet_ms": engine_packet_ms,
            "final_mode": "original_service_send_end_result",
            "send_end_result": True,
            "generate_options": params.get("generate_options") or {},
        }
    else:
        cfg_obj = SimpleNamespace(get=lambda key, default=None: config_get(cfg, key, default), vad_model=str(model_dir))
        model = service_model(project, cfg_obj)
        report["processing"] = {"packet_ms": settings["service_packet_ms"], "eos": cfg.get("vad_eos"),
                                "final_mode": "original_service_send_end_result"}
    detector = getattr(model, "model", None)
    detector_path = inspect.getsourcefile(type(detector)) if detector is not None else None
    report["detector"] = {"class": type(detector).__name__, "source_path": detector_path,
                          "source_sha256": sha(detector_path) if detector_path else None,
                          "vad_opts_loaded": dict(vars(detector.vad_opts)) if hasattr(detector, "vad_opts") else None,
                          "runtime_kwargs": runtime_kwargs(model)}
    if type(detector).__name__ != "FsmnVADStreaming":
        raise RuntimeError("Loaded model is not FsmnVADStreaming")
    validate_loaded_config(report["effective_model_yaml"], report["detector"]["vad_opts_loaded"] or {})
    report["fenshen_config"]["loaded_model_conf_verified"] = True
    if role == "engine":
        instrument(model, report, lambda: len(pcm) // 2)
        engine_packet_ms = int(report["processing"]["packet_ms"])
        result = run_streaming_vad(
            model,
            pcm,
            packet_ms=engine_packet_ms,
            send_end_result=True,
            generate_options=params.get("generate_options") or None,
        )
        report["vad_status"] = result.status
        report["event_log"] = result.event_log
        report["discarded"] = [vars(d) for d in result.discarded]
        payloads = []
        for seg in result.kept:
            item = clip_info(seg.pcm16le, seg.start_sample, seg.end_sample, seg.end_ms)
            item.update({
                "extra_padding_ms": seg.extra_padding_ms,
                "boundary_clamped": seg.boundary_clamped,
                "end_reason": seg.end_reason,
                "clip_branch": seg.clip_branch,
                "reported_start_ms": seg.reported_start_ms,
                "reported_end_ms": seg.reported_end_ms,
                "buffer_end_sample": seg.buffer_end_sample,
            })
            report["clips"].append(item)
            payloads.append(seg.pcm16le)
    else:
        payloads = asyncio.run(replay_service(project, cfg_obj, model, pcm, settings["service_packet_ms"], report, QuietLog()))
    if settings["save_clips"]:
        directory = Path(settings["artifact_dir"]) / role
        directory.mkdir(parents=True, exist_ok=True)
        for i, (item, payload) in enumerate(zip(report["clips"], payloads)):
            path = directory / f"{i:04d}.wav"
            with wave.open(str(path), "wb") as out:
                out.setnchannels(1)
                out.setsampwidth(2)
                out.setframerate(16000)
                out.writeframes(payload)
            item["wav_path"] = str(path)
    report["status"] = "ok"


def flatten(value, prefix=""):
    result = {}
    if isinstance(value, dict):
        for key, item in value.items():
            result.update(flatten(item, f"{prefix}.{key}" if prefix else key))
    else:
        result[prefix] = value
    return result


def compare(engine, service):
    result = {"both_succeeded": engine.get("status") == service.get("status") == "ok"}
    for name, path in (("dependencies", "dependencies"), ("model_config", "effective_model_yaml")):
        a, b = flatten(engine.get(path, {})), flatten(service.get(path, {}))
        result[name + "_differences"] = {k: {"engine": a.get(k), "service": b.get(k)}
                                            for k in sorted(a.keys() | b.keys()) if a.get(k) != b.get(k)}
    result["asset_hash_equal"] = {name: bool(engine.get("assets", {}).get(name, {}).get("sha256")) and
                                   engine["assets"][name]["sha256"] == service.get("assets", {}).get(name, {}).get("sha256")
                                   for name in ("model.pt", "am.mvn", "config.yaml", "configuration.json")}
    ends = lambda run: [e for c in run.get("calls", []) for e in c.get("events", [])]
    result["raw_event_sequence_equal"] = ends(engine) == ends(service)
    result["raw_events"] = {"engine": ends(engine), "service": ends(service)}
    a, b = engine.get("clips", []), service.get("clips", [])
    result["clip_counts"] = {"engine": len(a), "service": len(b)}
    # Match by exact start sample first. Do not silently shift index after dropped clips.
    used = set()
    matches = []
    for i, clip in enumerate(a):
        candidates = [(j, x) for j, x in enumerate(b) if j not in used
                      and x["clip_start_sample"] == clip["clip_start_sample"]]
        if not candidates:
            matches.append({"engine_index": i, "service_index": None, "reason": "no_exact_start_match"})
            continue
        j, other = min(candidates, key=lambda pair: abs(pair[1]["clip_end_ms"] - clip["clip_end_ms"]))
        used.add(j)
        matches.append({"engine_index": i, "service_index": j,
                        "engine_model_end_ms": clip.get("model_end_ms"), "service_model_end_ms": other.get("model_end_ms"),
                        "engine_clip_end_ms": clip["clip_end_ms"], "service_clip_end_ms": other["clip_end_ms"],
                        "engine_minus_service_end_ms": clip["clip_end_ms"] - other["clip_end_ms"],
                        "engine_extra_padding_ms": clip.get("extra_padding_ms"),
                        "engine_boundary_clamped": clip.get("boundary_clamped"),
                        "pcm_equal": clip["pcm_sha256"] == other["pcm_sha256"]})
    result["clip_matches"] = matches
    result["unmatched_service_indices"] = [j for j in range(len(b)) if j not in used]
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--engine-project", type=Path, required=True, help="audio-data-engine checkout")
    p.add_argument("--service-project", type=Path, required=True, help="ph_ai_model_asr_rt_cpu checkout")
    p.add_argument("--audio", type=Path, required=True, help="Same 16k mono PCM16 WAV/PCM")
    p.add_argument("--output", type=Path, required=True, help="Combined result JSON")
    p.add_argument("--engine-model-dir", help="Engine weight/asset source; fenshen config is applied in an isolated copy")
    p.add_argument("--service-model-dir", help="Service weight/asset source; fenshen config is applied in an isolated copy")
    p.add_argument("--engine-vad-config", help="Full fenshen VAD YAML; default ENGINE_PROJECT/configs/vad/fenshen_vad_config.yaml")
    p.add_argument("--service-vad-config", help="Full fenshen VAD YAML; default SERVICE_PROJECT/model_configs/fenshen_vad_config.yaml")
    p.add_argument("--engine-pipeline", help="Optional pipeline YAML to read ncpu/generate_options/model_runtime_dir")
    p.add_argument("--service-config", help="Actual service app YAML; overrides profile/scene")
    p.add_argument("--service-profile", default=os.getenv("env_profile", "stg"))
    p.add_argument("--service-scene", choices=("fenshen",), default="fenshen")
    p.add_argument("--service-packet-ms", type=int, default=100, help="Original client packet duration; default 100")
    p.add_argument("--save-clips", action="store_true", help="Also write each actual cropped WAV next to the JSON")
    p.add_argument("--timeout", type=int, default=1800, help="Seconds per project, default 1800")
    return p


def main():
    if len(sys.argv) == 5 and sys.argv[1] == "--_worker":
        settings = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
        report = {}
        try:
            worker(settings, sys.argv[3], report)
        except Exception:
            report.update({"status": "error", "error": traceback.format_exc()})
        write_json(sys.argv[4], report)
        return 0 if report["status"] == "ok" else 1
    args = parser().parse_args()
    if args.service_packet_ms < 1 or args.timeout < 1:
        parser().error("packet-ms and timeout must be positive")
    settings = vars(args).copy()
    for key in ("engine_project", "service_project", "audio", "output"):
        settings[key] = str(settings[key].expanduser().resolve())
    output = Path(settings["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    artifacts = Path(tempfile.mkdtemp(prefix=output.stem + "_files_", dir=output.parent))
    settings["artifact_dir"] = str(artifacts)
    report = {"schema_version": "fsmn-vad-project-comparison/v2",
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "settings": settings,
              "scope": "Required full fenshen YAML in isolated model copies. Real VAD and original crop methods; ASR/network/server startup disabled. Fixed packet replay, not a network capture.",
              "runs": {}}
    try:
        pcm = read_audio(settings["audio"])
        report["input"] = {"path": settings["audio"], "file_sha256": sha(settings["audio"]),
                           "pcm_sha256": hashlib.sha256(pcm).hexdigest(), "sample_rate": 16000,
                           "channels": 1, "samples": len(pcm) // 2, "duration_ms": len(pcm) / 32}
        request = artifacts / "request.json"
        write_json(request, settings)
        for role in ("engine", "service"):
            print(f"Running {role} VAD ...", flush=True)
            target, log = artifacts / f"{role}.json", artifacts / f"{role}.log"
            started = time.monotonic()
            try:
                with log.open("w", encoding="utf-8") as stream:
                    completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--_worker", str(request), role, str(target)],
                                               cwd=settings[f"{role}_project"], stdout=stream, stderr=subprocess.STDOUT, timeout=args.timeout)
                run = json.loads(target.read_text(encoding="utf-8")) if target.is_file() else {"status": "error", "error": "Worker did not write JSON"}
                run["exit_code"] = completed.returncode
            except Exception:
                run = {"status": "error", "error": traceback.format_exc()}
            run["elapsed_seconds"] = round(time.monotonic() - started, 3)
            run["log_path"] = str(log)
            run["log_tail"] = log.read_text(encoding="utf-8", errors="replace")[-16000:] if log.is_file() else ""
            report["runs"][role] = run
            if run.get("status") == "ok":
                opts = run["detector"]["vad_opts_loaded"]
                print(f"[{role}] fenshen verified: max_end_silence_time={opts['max_end_silence_time']}, "
                      f"speech_noise_thres={opts['speech_noise_thres']}, "
                      f"config_sha256={run['fenshen_config']['sha256']}", flush=True)
            write_json(output, report)
        report["comparison"] = compare(report["runs"]["engine"], report["runs"]["service"])
        report["status"] = "ok" if report["comparison"]["both_succeeded"] else "error"
    except Exception:
        report.update({"status": "error", "error": traceback.format_exc()})
    write_json(output, report)
    print(f"Result: {output}", flush=True)
    print(f"Status: {report['status']}", flush=True)
    if report.get("error"):
        print(report["error"], file=sys.stderr, flush=True)
    for role, run in report["runs"].items():
        if run.get("status") != "ok":
            print(f"[{role}] {run.get('error') or run.get('log_tail') or 'Unknown error'}", file=sys.stderr, flush=True)
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
