#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全面自检 Silero VAD：依赖、模型、配置、底表音频路径、抽样推理。

Audio I/O uses soundfile（与 audio_energy 一致），避开 Silero 官方 read_audio
→ torchaudio/torchcodec（torch 2.9+ 上 libtorchcodec_*.so 常挂）。

示例（服务器）：

  cd /data2/data-cp/lizi/audio-data-engine
  PYTHONPATH=src python scripts/check_silero_vad.py \\
    --batch 20260813-1 --limit 50

  PYTHONPATH=src python scripts/check_silero_vad.py \\
    --cleaned datasets/stage1/cleaned/cleaned_20260813-1.parquet \\
    --path-map /old/prefix=/data2/data-cp/lizi/audio-data-engine \\
    --limit 20

  PYTHONPATH=src python scripts/check_silero_vad.py --audio /path/to/one.wav

退出码：0=路径抽样可读且至少 1 条 VAD success；1=有阻断项；2=仅警告（路径全挂或 VAD 全失败）。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import re
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DEFAULT_VAD_CONFIG = ROOT / "configs/quality/silero_vad_v1.yaml"
DEFAULT_AUDIO_KEY = "resampled_16k"
VENDOR_SILERO_SRC = ROOT / "silero-vad-master" / "src"
VENDOR_ONNX = VENDOR_SILERO_SRC / "silero_vad" / "data" / "silero_vad.onnx"
VENDOR_JIT = VENDOR_SILERO_SRC / "silero_vad" / "data" / "silero_vad.jit"


def _ok(msg: str) -> None:
    print(f"[OK]  {msg}", flush=True)


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}", flush=True)


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}", flush=True)


def _info(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)


def _section(title: str) -> None:
    print(f"\n=== {title} ===", flush=True)


def _pkg_version(name: str) -> str | None:
    try:
        mod = __import__(name)
        return str(getattr(mod, "__version__", "unknown"))
    except Exception as exc:  # noqa: BLE001
        return f"UNAVAILABLE: {exc}"


def _module_file(name: str) -> str | None:
    try:
        spec = importlib.util.find_spec(name)
        if spec is None:
            return None
        return str(spec.origin or spec.submodule_search_locations)
    except Exception:  # noqa: BLE001
        return None


def parse_path_maps(raw: list[str] | None) -> list[tuple[str, str]]:
    maps: list[tuple[str, str]] = []
    for item in raw or []:
        if "=" not in item:
            raise ValueError(f"--path-map 格式应为 OLD=NEW，收到: {item!r}")
        old, new = item.split("=", 1)
        if not old:
            raise ValueError(f"--path-map OLD 为空: {item!r}")
        maps.append((old, new))
    return maps


def rewrite_path(path: str, *, path_maps: list[tuple[str, str]], audio_root: Path | None) -> str:
    text = str(path or "")
    for old, new in path_maps:
        if text.startswith(old):
            text = new + text[len(old) :]
            break
    candidate = Path(text)
    if audio_root is not None and not candidate.is_absolute():
        candidate = audio_root / candidate
    return str(candidate)


def classify_path_style(path: str) -> str:
    text = str(path or "")
    if not text or text == "__missing__":
        return "missing"
    if re.match(r"^[A-Za-z]:[\\/]", text) or text.startswith("\\\\"):
        return "windows_abs"
    if text.startswith("/"):
        return "posix_abs"
    return "relative"


def path_probe(path: str) -> dict[str, Any]:
    p = Path(path) if path else None
    out: dict[str, Any] = {
        "path": path,
        "style": classify_path_style(path),
        "exists": False,
        "is_file": False,
        "size_bytes": None,
        "readable": False,
        "error": None,
    }
    if p is None or not str(path):
        out["error"] = "empty_path"
        return out
    try:
        out["exists"] = p.exists()
        out["is_file"] = p.is_file()
        if out["is_file"]:
            out["size_bytes"] = p.stat().st_size
            with p.open("rb") as fh:
                fh.read(16)
            out["readable"] = True
    except Exception as exc:  # noqa: BLE001
        out["error"] = str(exc)
    return out


def common_prefix_stats(paths: list[str], *, top_n: int = 8) -> list[tuple[str, int]]:
    """Count first 1~3 path segments as rough location clusters."""
    counter: Counter[str] = Counter()
    for raw in paths:
        style = classify_path_style(raw)
        text = raw.replace("\\", "/")
        if style == "relative":
            parts = [p for p in text.split("/") if p]
            key = "/".join(parts[:2]) if parts else "(empty)"
        elif style == "windows_abs":
            parts = [p for p in text.split("/") if p]
            key = "/".join(parts[:3]) if parts else "(empty)"
        else:
            parts = [p for p in text.split("/") if p]
            key = "/" + "/".join(parts[:3]) if parts else "(empty)"
        counter[key] += 1
    return counter.most_common(top_n)


def check_environment(report: dict[str, Any]) -> list[str]:
    _section("1. 运行环境")
    blockers: list[str] = []
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "repo_root": str(ROOT),
        "cwd_is_repo": Path.cwd().resolve() == ROOT.resolve(),
        "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
        "executable": sys.executable,
    }
    report["environment"] = env
    _info(f"python={env['python']}  cwd={env['cwd']}")
    _info(f"repo_root={env['repo_root']}")
    if not env["cwd_is_repo"]:
        _warn("当前 cwd 不是仓库根；相对路径底表解析可能偏离服务器约定")
    else:
        _ok("cwd == 仓库根")

    deps = {
        "onnxruntime": _pkg_version("onnxruntime"),
        "torch": _pkg_version("torch"),
        "torchaudio": _pkg_version("torchaudio"),
        "soundfile": _pkg_version("soundfile"),
        "yaml": _pkg_version("yaml"),
    }
    report["dependencies"] = deps
    for name, ver in deps.items():
        if ver and str(ver).startswith("UNAVAILABLE"):
            level = _warn if name in {"torchaudio", "soundfile"} else _fail
            # torch is a hard import dependency of vendored silero_vad even when onnx=true.
            if name == "torch":
                level = _fail
            level(f"{name}: {ver}")
            if name == "onnxruntime":
                blockers.append("缺少 onnxruntime（默认 onnx=true 时必需）")
            elif name == "torch":
                blockers.append(
                    "缺少 torch：vendored silero_vad 在 onnx=true 时仍会 import torch"
                    "（utils_vad.py 顶层依赖）；这会导致全量 vad_status=failed"
                )
        else:
            _ok(f"{name}={ver}")
    return blockers


def check_silero_vendor_and_import(report: dict[str, Any], *, onnx: bool) -> list[str]:
    _section("2. Silero 代码/权重")
    blockers: list[str] = []
    vendor = {
        "vendor_src": str(VENDOR_SILERO_SRC),
        "vendor_src_exists": VENDOR_SILERO_SRC.is_dir(),
        "vendor_onnx": str(VENDOR_ONNX),
        "vendor_onnx_exists": VENDOR_ONNX.is_file(),
        "vendor_jit": str(VENDOR_JIT),
        "vendor_jit_exists": VENDOR_JIT.is_file(),
        "pip_silero_origin": _module_file("silero_vad"),
    }
    report["silero_vendor"] = vendor

    if vendor["vendor_src_exists"]:
        _ok(f"vendored src: {VENDOR_SILERO_SRC}")
    else:
        _fail(f"缺少 vendored Silero: {VENDOR_SILERO_SRC}")
        blockers.append("silero-vad-master/src 不存在")

    weight = VENDOR_ONNX if onnx else VENDOR_JIT
    if weight.is_file():
        _ok(f"权重可读: {weight} ({weight.stat().st_size} bytes)")
    else:
        _warn(f"vendored 权重缺失: {weight}（若走 pip 包内置权重可能仍可用）")

    # Prefer the same path injection production code uses.
    try:
        from audio_engine.core.quality.silero_vad import (
            DEFAULT_SILERO_ROOT,
            _ensure_silero_on_path,
        )

        ensured = _ensure_silero_on_path()
        report["silero_vendor"]["DEFAULT_SILERO_ROOT"] = str(DEFAULT_SILERO_ROOT)
        report["silero_vendor"]["ensured_on_path"] = str(ensured)
        if Path(ensured).resolve() != VENDOR_SILERO_SRC.resolve():
            _warn(
                f"DEFAULT_SILERO_ROOT={ensured} 与预期 {VENDOR_SILERO_SRC} 不一致"
            )
        else:
            _ok(f"DEFAULT_SILERO_ROOT 对齐: {ensured}")
    except Exception as exc:  # noqa: BLE001
        _fail(f"无法导入 audio_engine.core.quality.silero_vad: {exc}")
        blockers.append(f"import silero_vad core: {exc}")
        return blockers

    try:
        import silero_vad

        origin = getattr(silero_vad, "__file__", None)
        report["silero_vendor"]["imported_from"] = str(origin)
        _ok(f"import silero_vad OK ← {origin}")
        for attr in ("load_silero_vad", "read_audio", "get_speech_timestamps"):
            if not hasattr(silero_vad, attr):
                _fail(f"silero_vad 缺少 API: {attr}")
                blockers.append(f"missing API {attr}")
            else:
                _ok(f"API {attr} 存在")
    except Exception as exc:  # noqa: BLE001
        _fail(f"import silero_vad 失败: {exc}")
        blockers.append(f"import silero_vad: {exc}")
        report["silero_vendor"]["import_error"] = str(exc)
        report["silero_vendor"]["import_traceback"] = traceback.format_exc()
    return blockers


def check_config(report: dict[str, Any], config_path: Path) -> tuple[list[str], Any]:
    _section("3. VAD 配置")
    blockers: list[str] = []
    cfg = None
    if not config_path.is_file():
        _fail(f"配置不存在: {config_path}")
        blockers.append(f"missing config: {config_path}")
        report["config"] = {"path": str(config_path), "exists": False}
        return blockers, None

    from audio_engine.core.quality.silero_vad import load_silero_vad_config

    cfg = load_silero_vad_config(config_path)
    info = {
        "path": str(config_path.resolve()),
        "policy_version": cfg.policy_version,
        "input_audio_key": cfg.input_audio_key,
        "sample_rate": cfg.sample_rate,
        "onnx": cfg.onnx,
        "opset_version": cfg.opset_version,
        "model_path": cfg.model_path,
        "min_speech_ratio_present": cfg.min_speech_ratio_present,
        "max_speech_ratio_absent": cfg.max_speech_ratio_absent,
        "min_speech_duration_ms": cfg.min_speech_duration_ms,
        "get_speech_timestamps": cfg.get_speech_timestamps,
    }
    report["config"] = info
    _ok(f"loaded {config_path}")
    for k, v in info.items():
        if k == "path":
            continue
        _info(f"  {k}={v}")

    if cfg.model_path:
        mp = Path(cfg.model_path).expanduser()
        if mp.is_file():
            _ok(f"显式 model_path 可读: {mp}")
        else:
            _fail(f"显式 model_path 不可读: {mp}")
            blockers.append(f"model_path missing: {mp}")
    return blockers, cfg


def load_manifest_samples(
    *,
    batch: str | None,
    cleaned: Path | None,
    limit: int,
    seed: int,
) -> tuple[list[Any], Path]:
    from audio_engine.core.manifest import Manifest

    if cleaned is None:
        if not batch:
            raise ValueError("须提供 --batch 或 --cleaned")
        cleaned = ROOT / f"datasets/stage1/cleaned/cleaned_{batch}.parquet"
    if not cleaned.is_file():
        raise FileNotFoundError(f"cleaned 底表不存在: {cleaned}")

    samples = list(Manifest.load(cleaned))
    if not samples:
        raise ValueError(f"cleaned 为空: {cleaned}")
    if limit > 0 and limit < len(samples):
        # Stable subsample: stride through the list (deterministic without shuffle bias).
        step = max(1, len(samples) // limit)
        picked = samples[::step][:limit]
        # Also keep first/last for path-layout edge cases.
        for edge in (samples[0], samples[-1]):
            if edge not in picked:
                picked.append(edge)
        samples = picked[: max(limit, 2)]
    _ = seed  # reserved for future random mode
    return samples, cleaned


def check_manifest_paths(
    report: dict[str, Any],
    samples: list[Any],
    *,
    audio_key: str,
    path_maps: list[tuple[str, str]],
    audio_root: Path | None,
    path_sample: int,
) -> list[str]:
    _section("4. 底表音频路径")
    blockers: list[str] = []
    raw_paths: list[str] = []
    rewritten_paths: list[str] = []
    style_counts: Counter[str] = Counter()
    raw_ok = 0
    rewritten_ok = 0
    missing_key = 0
    examples_bad: list[dict[str, Any]] = []
    examples_rewritten: list[dict[str, Any]] = []

    for sample in samples:
        audio = getattr(sample, "audio", None) or {}
        raw = audio.get(audio_key)
        if not raw:
            missing_key += 1
            raw = ""
        raw_s = str(raw) if raw else ""
        raw_paths.append(raw_s or "__missing__")
        style_counts[classify_path_style(raw_s or "__missing__")] += 1

        raw_info = path_probe(raw_s) if raw_s else {"path": "", "readable": False, "error": "missing_key"}
        if raw_info.get("readable"):
            raw_ok += 1
        elif len(examples_bad) < 8:
            examples_bad.append(
                {
                    "id": getattr(sample, "id", None),
                    "raw": raw_info,
                    "source_path": getattr(sample, "source_path", None),
                    "audio_keys": sorted(audio.keys()),
                }
            )

        rewritten = rewrite_path(raw_s, path_maps=path_maps, audio_root=audio_root)
        rewritten_paths.append(rewritten)
        rw_info = path_probe(rewritten) if rewritten else {"readable": False, "error": "empty"}
        if rw_info.get("readable"):
            rewritten_ok += 1
            if rewritten != raw_s and len(examples_rewritten) < 5:
                examples_rewritten.append(
                    {"id": getattr(sample, "id", None), "from": raw_s, "to": rewritten}
                )

    n = max(1, len(samples))
    path_report = {
        "audio_key": audio_key,
        "samples_checked": len(samples),
        "missing_audio_key": missing_key,
        "style_counts": dict(style_counts),
        "raw_readable": raw_ok,
        "raw_readable_rate": round(raw_ok / n, 4),
        "rewritten_readable": rewritten_ok,
        "rewritten_readable_rate": round(rewritten_ok / n, 4),
        "path_maps": path_maps,
        "audio_root": str(audio_root) if audio_root else None,
        "common_prefixes_raw": common_prefix_stats(raw_paths),
        "common_prefixes_rewritten": common_prefix_stats(rewritten_paths),
        "examples_unreadable": examples_bad[:path_sample],
        "examples_rewritten": examples_rewritten,
    }
    report["paths"] = path_report

    _info(f"audio_key={audio_key}  抽样={len(samples)}  missing_key={missing_key}")
    _info(f"路径风格: {dict(style_counts)}")
    _info(f"原始路径可读: {raw_ok}/{len(samples)} ({path_report['raw_readable_rate']:.1%})")
    if path_maps or audio_root:
        _info(
            f"重写后可读: {rewritten_ok}/{len(samples)} ({path_report['rewritten_readable_rate']:.1%})"
        )
    _info(f"常见前缀(原始): {path_report['common_prefixes_raw']}")

    if missing_key == len(samples):
        _fail(f"全部样本缺少 audio.{audio_key}")
        blockers.append(f"all missing audio.{audio_key}")
    elif raw_ok == 0 and rewritten_ok == 0:
        _fail("抽样路径全部不可读（原始与重写后）——高度疑似路径前缀/挂载不一致")
        blockers.append("all audio paths unreadable")
        for ex in examples_bad[:5]:
            _info(f"  不可读例: id={ex.get('id')} path={ex.get('raw', {}).get('path')}")
    elif raw_ok == 0 and rewritten_ok > 0:
        _warn("原始路径全不可读，但 --path-map/--audio-root 后可读；分类入口须同步改路径或重写底表")
    elif raw_ok < len(samples):
        _warn(f"部分路径不可读: {len(samples) - raw_ok}/{len(samples)}")
    else:
        _ok("抽样原始路径全部可读")

    # Cross-OS smell
    if platform.system() != "Windows" and style_counts.get("windows_abs"):
        _fail("当前是非 Windows 主机，但底表含 Windows 绝对路径")
        blockers.append("windows paths on non-windows host")
    if platform.system() == "Windows" and style_counts.get("posix_abs") and raw_ok == 0:
        _warn("本地 Windows 上看到 POSIX 绝对路径且不可读：请到服务器跑本脚本，或用 --path-map")

    return blockers


def check_model_load(report: dict[str, Any], cfg: Any) -> tuple[list[str], Any]:
    _section("5. 模型加载（不跑音频）")
    blockers: list[str] = []
    session = None
    try:
        from audio_engine.core.quality.silero_vad import SileroVadSession

        session = SileroVadSession(cfg)
        t0 = time.perf_counter()
        session._load()
        elapsed = time.perf_counter() - t0
        report["model_load"] = {
            "ok": True,
            "elapsed_sec": round(elapsed, 3),
            "runtime": session.runtime,
            "model_digest": session.model_digest,
        }
        _ok(
            f"模型加载成功 runtime={session.runtime} digest={session.model_digest} "
            f"({elapsed:.2f}s)"
        )
    except Exception as exc:  # noqa: BLE001
        report["model_load"] = {
            "ok": False,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        _fail(f"模型加载失败: {exc}")
        _info(traceback.format_exc())
        blockers.append(f"model load: {exc}")
    return blockers, session


def run_vad_samples(
    report: dict[str, Any],
    *,
    cfg: Any,
    session: Any,
    samples: list[Any] | None,
    audio_files: list[Path] | None,
    audio_key: str,
    path_maps: list[tuple[str, str]],
    audio_root: Path | None,
    limit: int,
) -> list[str]:
    _section("6. 抽样 VAD 推理")
    blockers: list[str] = []
    from audio_engine.core.quality.silero_vad import SileroVadSession

    if session is None:
        session = SileroVadSession(cfg)

    targets: list[tuple[str, str]] = []  # (label, path)
    if audio_files:
        for p in audio_files:
            targets.append((p.name, str(p)))
    elif samples:
        for sample in samples:
            audio = getattr(sample, "audio", None) or {}
            raw = str(audio.get(audio_key) or "")
            path = rewrite_path(raw, path_maps=path_maps, audio_root=audio_root)
            targets.append((str(getattr(sample, "id", "?")), path))
    if limit > 0:
        targets = targets[:limit]

    status_counts: Counter[str] = Counter()
    state_counts: Counter[str] = Counter()
    error_counts: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    success = 0

    for label, path in targets:
        probe = path_probe(path)
        t0 = time.perf_counter()
        fields = session.analyze(path)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        status = str(fields.get("vad_status") or "")
        state = str(fields.get("vad_speech_state") or "")
        err = fields.get("vad_error")
        status_counts[status] += 1
        state_counts[state] += 1
        if status == "success":
            success += 1
        if err:
            # Collapse long paths in error keys
            key = str(err)
            if key.startswith("audio_unreadable:"):
                key = "audio_unreadable"
            elif len(key) > 120:
                key = key[:120] + "…"
            error_counts[key] += 1
        row = {
            "id": label,
            "path": path,
            "path_readable": bool(probe.get("readable")),
            "path_style": probe.get("style"),
            "vad_status": status,
            "vad_speech_state": state,
            "speech_ratio": fields.get("speech_ratio"),
            "speech_duration_ms": fields.get("speech_duration_ms"),
            "speech_segment_count": fields.get("speech_segment_count"),
            "vad_error": err,
            "elapsed_ms": elapsed_ms,
        }
        rows.append(row)
        mark = "OK" if status == "success" else "FAIL"
        extra = (
            f"state={state} ratio={fields.get('speech_ratio')}"
            if status == "success"
            else f"error={err}"
        )
        print(f"  [{mark}] {label}: {extra}", flush=True)

    vad_report = {
        "n": len(targets),
        "success": success,
        "success_rate": round(success / max(1, len(targets)), 4),
        "vad_status_counts": dict(status_counts),
        "vad_speech_state_counts": dict(state_counts),
        "vad_error_counts": dict(error_counts),
        "rows": rows,
    }
    report["vad_probe"] = vad_report

    _info(f"成功 {success}/{len(targets)}  status={dict(status_counts)}  state={dict(state_counts)}")
    if error_counts:
        _info(f"vad_error 分布: {dict(error_counts)}")

    if not targets:
        _fail("没有可跑的音频目标")
        blockers.append("no vad targets")
    elif success == 0:
        # Distinguish path vs runtime
        path_readable = sum(1 for r in rows if r.get("path_readable"))
        if path_readable == 0:
            _fail("全部失败且路径均不可读 → 先修路径/挂载/--path-map")
            blockers.append("vad all failed: unreadable paths")
        else:
            _fail(
                f"路径可读 {path_readable}/{len(rows)} 但 VAD 全失败 → 依赖/模型/read_audio/"
                f"get_speech_timestamps 异常（见 vad_error）"
            )
            blockers.append("vad all failed: runtime")
            # Show one traceback-level hint by re-raising path through analyze fields
            sample_err = next((r.get("vad_error") for r in rows if r.get("vad_error")), None)
            if sample_err:
                _info(f"代表性 vad_error: {sample_err}")
    else:
        _ok(f"VAD 抽样可用（success_rate={vad_report['success_rate']:.1%}）")
    return blockers


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="全面检查 Silero VAD 依赖、权重、底表路径与抽样推理"
    )
    p.add_argument("--batch", help="source-name，默认读 datasets/stage1/cleaned/cleaned_{batch}.parquet")
    p.add_argument("--cleaned", type=Path, help="显式 cleaned parquet/jsonl")
    p.add_argument(
        "--audio",
        type=Path,
        action="append",
        default=None,
        help="直接指定 wav（可重复）；提供时跳过底表抽样推理，仍可做环境检查",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_VAD_CONFIG,
        help=f"VAD 配置（默认 {DEFAULT_VAD_CONFIG}）",
    )
    p.add_argument(
        "--audio-key",
        default=DEFAULT_AUDIO_KEY,
        help=f"Sample.audio 键（默认 {DEFAULT_AUDIO_KEY}）",
    )
    p.add_argument(
        "--path-map",
        action="append",
        default=None,
        help="路径前缀重写 OLD=NEW，可重复；用于服务器/本地前缀不一致",
    )
    p.add_argument(
        "--audio-root",
        type=Path,
        default=None,
        help="相对路径时拼到该根目录下（例如仓库根或数据盘挂载点）",
    )
    p.add_argument("--limit", type=int, default=30, help="底表/推理抽样条数（默认 30）")
    p.add_argument("--seed", type=int, default=0, help="预留随机种子")
    p.add_argument(
        "--path-sample",
        type=int,
        default=8,
        help="报告中保留的不可读路径样例数",
    )
    p.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="把完整报告写成 JSON",
    )
    p.add_argument(
        "--skip-manifest",
        action="store_true",
        help="只检查环境/模型；不读底表（可与 --audio 联用）",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report: dict[str, Any] = {"ok": False, "blockers": [], "warnings": []}
    blockers: list[str] = []

    print("Silero VAD 全面自检", flush=True)
    print(f"time={time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    try:
        path_maps = parse_path_maps(args.path_map)
    except ValueError as exc:
        _fail(str(exc))
        return 1

    audio_root = args.audio_root.expanduser().resolve() if args.audio_root else None
    if audio_root is not None and not audio_root.is_dir():
        _warn(f"--audio-root 不是目录: {audio_root}")

    blockers += check_environment(report)

    # Need onnx flag from config early; fall back to yaml default true.
    onnx_flag = True
    if args.config.is_file():
        try:
            import yaml

            raw = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
            onnx_flag = bool(raw.get("onnx", True))
        except Exception:  # noqa: BLE001
            pass

    blockers += check_silero_vendor_and_import(report, onnx=onnx_flag)
    cfg_blockers, cfg = check_config(report, args.config)
    blockers += cfg_blockers
    if cfg is None:
        # Minimal default so later steps can still attempt if user only has --audio
        from audio_engine.core.quality.silero_vad import SileroVadConfig

        cfg = SileroVadConfig()

    samples = None
    cleaned_path = None
    if not args.skip_manifest and (args.batch or args.cleaned):
        try:
            samples, cleaned_path = load_manifest_samples(
                batch=args.batch,
                cleaned=args.cleaned,
                limit=max(args.limit, 1),
                seed=args.seed,
            )
            report["manifest"] = {
                "cleaned": str(cleaned_path),
                "total_loaded_for_check": len(samples),
            }
            _ok(f"底表: {cleaned_path}（本检查载入 {len(samples)} 条）")
            blockers += check_manifest_paths(
                report,
                samples,
                audio_key=args.audio_key,
                path_maps=path_maps,
                audio_root=audio_root,
                path_sample=args.path_sample,
            )
        except Exception as exc:  # noqa: BLE001
            _fail(f"读取底表失败: {exc}")
            blockers.append(f"manifest: {exc}")
            report["manifest"] = {"error": str(exc), "traceback": traceback.format_exc()}
    elif not args.audio:
        _warn("未提供 --batch/--cleaned/--audio；跳过路径与抽样推理（仅环境/模型）")

    load_blockers, session = check_model_load(report, cfg)
    blockers += load_blockers

    audio_files = [p.expanduser().resolve() for p in (args.audio or [])]
    if audio_files or samples:
        if session is None and not load_blockers:
            # model load failed already recorded
            pass
        blockers += run_vad_samples(
            report,
            cfg=cfg,
            session=session,
            samples=samples,
            audio_files=audio_files or None,
            audio_key=args.audio_key,
            path_maps=path_maps,
            audio_root=audio_root,
            limit=args.limit if not audio_files else 0,
        )

    _section("结论")
    report["blockers"] = blockers
    # Soft warnings already printed; treat path-rewrite-needed as blocker if listed.
    if not blockers:
        report["ok"] = True
        _ok("自检通过：依赖/模型可用，且抽样 VAD 至少有成功（或仅环境检查无阻断）")
        exit_code = 0
    else:
        report["ok"] = False
        _fail(f"发现 {len(blockers)} 项阻断：")
        for b in blockers:
            print(f"  - {b}", flush=True)
        # Heuristic exit: path issues vs runtime
        exit_code = 2 if any("unreadable" in b or "path" in b for b in blockers) else 1
        _info(
            "排查顺序建议: "
            "1) 路径风格/挂载/--path-map  "
            "2) torch / onnxruntime / silero-vad-master 权重  "
            "3) 单条 --audio 看完整 vad_error"
            "（读音频已走 soundfile，不应再出现 torchcodec lib 加载错误）"
        )
        err_map = (report.get("vad_probe") or {}).get("vad_error_counts") or {}
        if any("torchcodec" in str(k).lower() for k in err_map):
            _info(
                "HINT: 仍出现 torchcodec 错误 → 确认服务器已同步 "
                "src/audio_engine/core/quality/silero_vad.py（soundfile I/O）"
            )
        elif any("torch" in b.lower() for b in blockers):
            _info(
                "HINT: vendored silero_vad 模型侧仍需要 torch；"
                "wav I/O 已改为 soundfile，与 torchcodec 无关。"
            )

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        _info(f"JSON 报告: {args.json_out}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
