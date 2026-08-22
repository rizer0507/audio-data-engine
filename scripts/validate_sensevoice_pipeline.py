#!/usr/bin/env python3
"""SenseVoice 可用性与空输出诊断脚本。

对照官方 FunASR 调用与本仓库 asr.sensevoice_batch 路径，抽样验证模型能否产出非空文本，
并把原始返回、解析结果、音频探测信息写入 JSON，便于定位「识别结果全是空」的原因。

示例：
  CUDA_VISIBLE_DEVICES=4 PYTHONPATH=src python scripts/validate_sensevoice_pipeline.py \\
    --manifest datasets/manifests/qwen_asr_source_A.parquet \\
    --model-path /data/models/SenseVoiceSmall \\
    --samples 3 \\
    --batch-size 1

  PYTHONPATH=src python scripts/validate_sensevoice_pipeline.py \\
    --audio /path/to/a.wav --model-path iic/SenseVoiceSmall --device cpu
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    text = getattr(value, "text", None)
    if text is not None:
        return {"type": type(value).__name__, "text": str(text), "repr": repr(value)[:500]}
    return {"type": type(value).__name__, "repr": repr(value)[:500]}


def _probe_audio(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else None,
    }
    if not path.exists():
        info["error"] = "file not found"
        return info
    try:
        import soundfile as sf

        meta = sf.info(str(path))
        info.update(
            {
                "sample_rate": meta.samplerate,
                "channels": meta.channels,
                "duration_sec": round(float(meta.duration), 4),
                "format": meta.format,
                "subtype": meta.subtype,
                "valid": True,
            }
        )
    except Exception as exc:  # noqa: BLE001
        info.update({"valid": False, "error": str(exc)})
    return info


def _resolve_model_path(cli_path: str | None) -> str:
    return (
        cli_path
        or os.environ.get("SENSEVOICE_MODEL_PATH")
        or "iic/SenseVoiceSmall"
    )


def _env_report(model_path: str, device: str) -> dict[str, Any]:
    report: dict[str, Any] = {
        "python": sys.version.split()[0],
        "cwd": str(Path.cwd()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "sensevoice_model_path_env": os.environ.get("SENSEVOICE_MODEL_PATH"),
        "resolved_model_path": model_path,
        "device": device,
        "funasr_importable": False,
        "torch": {},
    }
    local = Path(model_path)
    if local.exists():
        report["model_path_exists"] = True
        report["model_path_is_dir"] = local.is_dir()
        report["model_files"] = sorted(p.name for p in local.iterdir())[:30]
    else:
        report["model_path_exists"] = False
        report["model_path_note"] = "路径不存在（若为 ModelScope id 则首次会下载）"

    try:
        import funasr  # type: ignore

        report["funasr_importable"] = True
        report["funasr_version"] = getattr(funasr, "__version__", "unknown")
    except Exception as exc:  # noqa: BLE001
        report["funasr_error"] = str(exc)
        report["funasr_fix"] = traceback.format_exc(limit=12)
        err = str(exc)
        if "pkg_resources" in err:
            report["funasr_fix_hint"] = (
                "funasr 依赖 setuptools 提供的 pkg_resources；"
                "请执行: pip install -U setuptools"
            )
        elif "No module named 'funasr'" in err:
            report["funasr_fix_hint"] = "pip install 'audio-data-engine[sensevoice]'"
        else:
            report["funasr_fix_hint"] = f"funasr 导入失败: {err}"

    try:
        import torch

        report["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "current_device": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        }
    except Exception as exc:  # noqa: BLE001
        report["torch"] = {"error": str(exc)}
    return report


def _load_samples(
    manifest: Path | None,
    audio_files: list[Path],
    input_key: str,
    limit: int,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for path in audio_files:
        items.append(
            {
                "id": path.stem,
                "audio_path": str(path.resolve()),
                "qwen_text": None,
                "source": "cli-audio",
            }
        )

    if manifest is not None:
        from audio_engine.core.manifest import Manifest

        loaded = Manifest.load(manifest)
        for sample in loaded.samples:
            try:
                audio_path = sample.audio_path(input_key)
            except KeyError:
                audio_path = sample.source_path
            qwen = sample.transcripts.get("qwen", {})
            qwen_text = qwen.get("text") if isinstance(qwen, dict) else None
            items.append(
                {
                    "id": sample.id,
                    "audio_path": audio_path,
                    "qwen_text": qwen_text,
                    "source": str(manifest),
                    "has_input_key": input_key in sample.audio,
                }
            )
            if len(items) >= limit:
                break

    if not items:
        raise SystemExit(
            "未找到待测音频：请传 --manifest 或 --audio，并确认 input_audio_key 存在。"
        )
    return items[:limit]


def _extract_raw_text(raw: Any) -> str:
    if isinstance(raw, dict):
        return str(raw.get("text", "") or "")
    return str(getattr(raw, "text", raw) or "")


def _diagnose_text(raw_text: str, parsed_text: str) -> list[str]:
    hints: list[str] = []
    if not raw_text.strip():
        hints.append("FunASR 原始 text 为空：更像模型/音频问题，而非标签解析问题")
    elif not parsed_text.strip():
        if "<|nospeech|>" in raw_text.lower() or "nospeech" in raw_text.lower():
            hints.append("原始输出含 nospeech 标签：模型判定无语音，解析后文本为空属预期")
        else:
            hints.append("原始 text 非空但解析后为空：可能只含控制标签，或标签解析过宽")
    if raw_text and raw_text == parsed_text and raw_text.startswith("<|"):
        hints.append("标签未被剥离：检查 parse_sensevoice_text 与 FunASR 标签格式是否一致")
    return hints


def _generate_variants(
    model: Any,
    audio_path: str,
    language: str,
    use_itn: bool,
    batch_size: int,
) -> list[dict[str, Any]]:
    """用几组常见 generate 参数试跑，定位 API 兼容问题。"""
    variants = [
        {
            "name": "project_default",
            "kwargs": {
                "input": [audio_path],
                "language": language,
                "use_itn": use_itn,
                "batch_size": batch_size,
            },
        },
        {
            "name": "official_batch_size_s",
            "kwargs": {
                "input": audio_path,
                "language": language,
                "use_itn": use_itn,
                "batch_size_s": 60,
            },
        },
        {
            "name": "minimal",
            "kwargs": {
                "input": audio_path,
                "language": language,
                "use_itn": use_itn,
            },
        },
    ]
    results = []
    for variant in variants:
        entry: dict[str, Any] = {"name": variant["name"], "kwargs": variant["kwargs"]}
        started = time.perf_counter()
        try:
            raw = model.generate(**variant["kwargs"])
            elapsed = time.perf_counter() - started
            entry["ok"] = True
            entry["elapsed_sec"] = round(elapsed, 4)
            entry["raw"] = _jsonable(raw)
            if isinstance(raw, list) and raw:
                raw_text = _extract_raw_text(raw[0])
            else:
                raw_text = _extract_raw_text(raw)
            entry["raw_text"] = raw_text
            entry["raw_text_empty"] = not bool(raw_text.strip())
        except Exception as exc:  # noqa: BLE001
            entry["ok"] = False
            entry["elapsed_sec"] = round(time.perf_counter() - started, 4)
            entry["error"] = str(exc)
            entry["traceback"] = traceback.format_exc(limit=8)
        results.append(entry)
    return results


def _run_project_path(
    audio_path: str,
    model_path: str,
    device: str,
    language: str,
    use_itn: bool,
    batch_size: int,
) -> dict[str, Any]:
    from audio_engine.operators.asr import sensevoice as sv

    settings = {
        "model": "sensevoice-small",
        "model_path": model_path,
        "model_version": "validate",
        "device": device,
        "language": language,
        "use_itn": use_itn,
        "batch_size": batch_size,
        "disable_update": True,
    }
    started = time.perf_counter()
    try:
        model = sv._load_sensevoice_model(settings)
        parsed_list = sv._transcribe_many(model, [audio_path], settings)
        elapsed = time.perf_counter() - started
        parsed = parsed_list[0]
        return {
            "ok": True,
            "elapsed_sec": round(elapsed, 4),
            "parsed": parsed,
            "text": parsed.get("text", ""),
            "text_empty": not bool(str(parsed.get("text", "")).strip()),
            "hints": _diagnose_text(
                str(parsed.get("extra", {}).get("raw_text", "")),
                str(parsed.get("text", "")),
            ),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "elapsed_sec": round(time.perf_counter() - started, 4),
            "error": str(exc),
            "traceback": traceback.format_exc(limit=12),
        }


def _load_model(model_path: str, device: str, with_vad: bool) -> tuple[Any, dict[str, Any]]:
    from funasr import AutoModel  # type: ignore

    kwargs: dict[str, Any] = {
        "model": model_path,
        "device": device,
        "disable_update": True,
    }
    if with_vad:
        kwargs["vad_model"] = "fsmn-vad"
        kwargs["vad_kwargs"] = {"max_single_segment_time": 30000}
    started = time.perf_counter()
    model = AutoModel(**kwargs)
    return model, {
        "load_kwargs": kwargs,
        "load_elapsed_sec": round(time.perf_counter() - started, 4),
        "with_vad": with_vad,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate SenseVoice inference / empty-output diagnosis")
    parser.add_argument("--manifest", type=Path, help="输入 manifest（parquet/jsonl）")
    parser.add_argument("--audio", type=Path, action="append", default=[], help="直接指定音频，可重复")
    parser.add_argument("--model-path", type=str, default=None, help="本地模型目录或 ModelScope id")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--samples", type=int, default=3, help="从 manifest 抽样条数")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input-audio-key", type=str, default="resampled_16k")
    parser.add_argument("--language", type=str, default="auto")
    parser.add_argument("--no-itn", action="store_true", help="关闭 ITN")
    parser.add_argument("--with-vad", action="store_true", help="加载 fsmn-vad（长音频更稳）")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/sensevoice_validation.json"),
        help="诊断报告输出路径",
    )
    args = parser.parse_args()

    model_path = _resolve_model_path(args.model_path)
    use_itn = not args.no_itn
    report: dict[str, Any] = {
        "ok": False,
        "empty_text_count": 0,
        "sample_count": 0,
        "env": _env_report(model_path, args.device),
        "args": {
            "manifest": str(args.manifest) if args.manifest else None,
            "audio": [str(p) for p in args.audio],
            "model_path": model_path,
            "device": args.device,
            "samples": args.samples,
            "batch_size": args.batch_size,
            "input_audio_key": args.input_audio_key,
            "language": args.language,
            "use_itn": use_itn,
            "with_vad": args.with_vad,
        },
        "model_load": {},
        "samples": [],
        "summary_hints": [],
    }

    if not report["env"].get("funasr_importable"):
        hint = report["env"].get("funasr_fix_hint") or (
            "funasr 未安装：pip install 'audio-data-engine[sensevoice]'"
        )
        report["summary_hints"].append(hint)
        _write_report(args.output, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 2

    try:
        items = _load_samples(args.manifest, args.audio, args.input_audio_key, args.samples)
    except SystemExit as exc:
        report["summary_hints"].append(str(exc))
        _write_report(args.output, report)
        print(str(exc), file=sys.stderr)
        return 2

    try:
        model, load_info = _load_model(model_path, args.device, args.with_vad)
        report["model_load"] = {"ok": True, **load_info}
    except Exception as exc:  # noqa: BLE001
        report["model_load"] = {
            "ok": False,
            "error": str(exc),
            "traceback": traceback.format_exc(limit=12),
        }
        report["summary_hints"].append(f"模型加载失败: {exc}")
        _write_report(args.output, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 2

    empty = 0
    for item in items:
        audio_path = Path(item["audio_path"])
        sample_report: dict[str, Any] = {
            "id": item["id"],
            "source": item["source"],
            "qwen_text": item.get("qwen_text"),
            "has_input_key": item.get("has_input_key"),
            "audio": _probe_audio(audio_path),
        }
        if not audio_path.exists():
            sample_report["error"] = "audio missing"
            sample_report["text_empty"] = True
            empty += 1
            report["samples"].append(sample_report)
            continue

        sample_report["funasr_variants"] = _generate_variants(
            model,
            str(audio_path),
            args.language,
            use_itn,
            args.batch_size,
        )
        sample_report["project_path"] = _run_project_path(
            str(audio_path),
            model_path,
            args.device,
            args.language,
            use_itn,
            args.batch_size,
        )

        project = sample_report["project_path"]
        text_empty = bool(project.get("text_empty", True))
        # 若项目路径失败，再用官方风格第一成功结果判断
        if not project.get("ok"):
            ok_variant = next(
                (v for v in sample_report["funasr_variants"] if v.get("ok") and not v.get("raw_text_empty")),
                None,
            )
            text_empty = ok_variant is None
            if ok_variant is not None:
                sample_report["fallback_non_empty_variant"] = ok_variant["name"]
                sample_report["summary_hints"] = [
                    "官方 generate 有非空结果，但项目 _transcribe_many 失败/为空：优先查 batch_size 参数与返回条数校验"
                ]

        sample_report["text_empty"] = text_empty
        if text_empty:
            empty += 1
            hints = list(project.get("hints") or [])
            for variant in sample_report["funasr_variants"]:
                if variant.get("ok") and variant.get("raw_text_empty"):
                    hints.append(f"{variant['name']}: 原始 text 为空")
                elif not variant.get("ok"):
                    hints.append(f"{variant['name']}: 调用失败 -> {variant.get('error')}")
            audio_meta = sample_report["audio"]
            if audio_meta.get("duration_sec") is not None and audio_meta["duration_sec"] < 0.2:
                hints.append("音频时长 < 0.2s，可能被判定为无语音")
            if audio_meta.get("sample_rate") and audio_meta["sample_rate"] not in (8000, 16000):
                hints.append(
                    f"采样率 {audio_meta['sample_rate']} 非 8k/16k，建议确认是否已走 resampled_16k"
                )
            if item.get("has_input_key") is False:
                hints.append(
                    f"样本缺少 audio['{args.input_audio_key}']，当前回退路径可能不是期望输入"
                )
            sample_report["hints"] = hints
        report["samples"].append(sample_report)

    report["sample_count"] = len(report["samples"])
    report["empty_text_count"] = empty
    report["ok"] = empty == 0 and report["sample_count"] > 0

    if report["ok"]:
        report["summary_hints"].append("全部抽样样本均得到非空 SenseVoice 文本，模型链路可用")
    else:
        report["summary_hints"].append(
            f"{empty}/{report['sample_count']} 条文本为空，请查看 samples[*].funasr_variants 与 project_path"
        )
        if any(
            (s.get("project_path") or {}).get("ok") is False
            and any(v.get("ok") and not v.get("raw_text_empty") for v in s.get("funasr_variants", []))
            for s in report["samples"]
        ):
            report["summary_hints"].append(
                "线索：FunASR 直调有文本，但仓库解析路径异常 —— 重点核对 batch_size vs batch_size_s、返回 list 长度"
            )
        if any(
            all(v.get("ok") and v.get("raw_text_empty") for v in s.get("funasr_variants", []) if "ok" in v)
            for s in report["samples"]
            if s.get("funasr_variants")
        ):
            report["summary_hints"].append(
                "线索：多组 generate 参数下原始 text 都为空 —— 查模型权重、音频内容/静音、device，或加 --with-vad 重试"
            )

    _write_report(args.output, report)
    _print_human_summary(report, args.output)
    return 0 if report["ok"] else 1


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def _print_human_summary(report: dict[str, Any], output: Path) -> None:
    print("=" * 60)
    print("SenseVoice 验证结果")
    print("=" * 60)
    print(f"模型: {report['args']['model_path']}")
    print(f"设备: {report['args']['device']}")
    print(f"样本: {report['sample_count']}  空文本: {report['empty_text_count']}")
    print(f"整体: {'PASS' if report['ok'] else 'FAIL'}")
    print(f"报告: {output}")
    for hint in report.get("summary_hints", []):
        print(f"- {hint}")
    for sample in report.get("samples", []):
        status = "EMPTY" if sample.get("text_empty") else "OK"
        project = sample.get("project_path") or {}
        text = project.get("text") or ""
        preview = text if len(text) <= 80 else text[:77] + "..."
        print(f"  [{status}] {sample.get('id')}: {preview!r}")
        for h in sample.get("hints") or []:
            print(f"      · {h}")


if __name__ == "__main__":
    raise SystemExit(main())
