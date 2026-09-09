#!/usr/bin/env python3
"""Probe a Kimi-Audio vLLM service with one WAV file or a WAV directory."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from audio_engine.core.operator import OperatorConfig  # noqa: E402
from audio_engine.operators.asr import kimi  # noqa: E402
from audio_engine.operators.audio.kimi_pad import pad_wav_file, plan_kimi_pad  # noqa: E402


def discover_wavs(input_path: Path, recursive: bool = False) -> list[Path]:
    """Return a stable list of WAV inputs, rejecting unsupported paths early."""
    path = input_path.expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() != ".wav":
            raise ValueError(f"仅支持 WAV 文件: {path}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"输入不存在: {path}")
    pattern = "**/*" if recursive else "*"
    wavs = sorted(
        item for item in path.glob(pattern) if item.is_file() and item.suffix.lower() == ".wav"
    )
    if not wavs:
        raise ValueError(f"目录中没有 WAV 文件: {path}")
    return wavs


def pad_probe_wavs(paths: list[Path], work_dir: Path) -> tuple[list[Path], list[dict[str, object]]]:
    """Pad probe WAVs with the same Kimi-vLLM buckets used by the pipeline."""
    padded: list[Path] = []
    notes: list[dict[str, object]] = []
    for path in paths:
        plan = plan_kimi_pad(path)
        dest = work_dir / f"{path.stem}{path.suffix.lower()}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if plan.needs_write:
            out = pad_wav_file(path, dest, plan)
        else:
            out = path
        padded.append(out)
        notes.append(
            {
                "audio": str(path),
                "padded_audio": str(out),
                "kimi_pad_mode": plan.mode,
                "kimi_pad_target_s": plan.target_s,
                "source_duration": plan.source_duration,
            }
        )
    return padded, notes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="用单个 WAV 或 WAV 文件夹探测 Kimi-Audio vLLM transcription 接口"
    )
    parser.add_argument("input", type=Path, help="单个 .wav 文件或包含 .wav 的目录")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/asr/kimi.yaml")
    parser.add_argument(
        "--api-base",
        help="覆盖 KIMI_ASR_API_BASE / 配置文件；多台用逗号分隔",
    )
    parser.add_argument("--model", help="覆盖 KIMI_ASR_MODEL / 配置文件")
    parser.add_argument("--concurrency", type=int, help="目录探针并发请求数")
    parser.add_argument("--recursive", action="store_true", help="递归查找子目录中的 WAV")
    parser.add_argument(
        "--pad",
        action="store_true",
        help="按 Kimi-vLLM 时长桶尾部静音 pad 后再发请求（长短混合探针必开）",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="同一批音频连续识别次数；2 用于文本稳定性探针",
    )
    return parser


def probe_pad_bucket_key(
    source: Path,
    sent: Path,
    note: dict[str, object] | None,
) -> tuple[str, int | None]:
    """Use pad notes when present; otherwise plan the file that will be uploaded."""
    if note is not None:
        return kimi.pad_bucket_key_from_values(note.get("kimi_pad_mode"), note.get("kimi_pad_target_s"))
    plan = plan_kimi_pad(sent if sent.exists() else source)
    return (plan.mode, plan.target_s)


def transcribe_probe_paths(
    send_paths: list[Path],
    source_paths: list[Path],
    settings: dict,
    pad_notes: list[dict[str, object]],
) -> list[dict]:
    """Keep source order. Concurrent HTTP only within one pad bucket."""
    note_by_source = {str(note["audio"]): note for note in pad_notes}
    keys = [
        probe_pad_bucket_key(source, sent, note_by_source.get(str(source)))
        for source, sent in zip(source_paths, send_paths)
    ]
    batch_size = max(
        1,
        min(
            int(settings["concurrency"]),
            int(settings.get("batch_size", settings["concurrency"])),
        ),
    )
    windows = kimi.iter_pad_bucket_windows(keys, batch_size=batch_size)
    ordered: list[dict | None] = [None] * len(send_paths)
    for window in windows:
        chunk_settings = dict(settings)
        if keys[window[0]][0] == "over_30s":
            chunk_settings["concurrency"] = 1
        transcripts = kimi._transcribe_many(
            [str(send_paths[index]) for index in window],
            chunk_settings,
            sample_ids=[source_paths[index].stem for index in window],
        )
        for index, transcript in zip(window, transcripts):
            ordered[index] = transcript
    if any(item is None for item in ordered):
        raise RuntimeError("探针按桶识别结果不完整")
    return [item for item in ordered if item is not None]


def _print_result(path: Path, result: dict, extra: dict[str, object] | None = None) -> str:
    text = str(result.get("text", "")).strip()
    payload: dict[str, object] = {
        "ok": bool(text),
        "audio": str(path),
        "text": text,
        "language": result.get("language"),
    }
    if extra:
        payload.update(extra)
    print(json.dumps(payload, ensure_ascii=False))
    return text


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.repeat < 1:
        print(json.dumps({"ok": False, "error": "--repeat 必须 >= 1"}, ensure_ascii=False), file=sys.stderr)
        return 1
    params: dict[str, object] = {"config_path": str(args.config)}
    for key in ("api_base", "model", "concurrency"):
        value = getattr(args, key)
        if value is not None:
            params[key] = value

    try:
        source_paths = discover_wavs(args.input, args.recursive)
        settings = kimi._resolve_settings(OperatorConfig(params=params))
        send_paths = source_paths
        pad_notes: list[dict[str, object]] = []
        tmp_dir = None
        if args.pad:
            tmp_dir = Path(tempfile.mkdtemp(prefix="kimi_vllm_pad_probe_"))
            send_paths, pad_notes = pad_probe_wavs(source_paths, tmp_dir)
        note_by_source = {str(note["audio"]): note for note in pad_notes}
        if not args.pad:
            planned = [plan_kimi_pad(path) for path in send_paths]
            if len({(plan.mode, plan.target_s) for plan in planned}) > 1:
                print(
                    json.dumps(
                        {
                            "warning": "目录含多种时长但未加 --pad；vLLM 0.19 Kimi 混 T 并发会打挂 EngineCore，请加 --pad",
                        },
                        ensure_ascii=False,
                    ),
                    file=sys.stderr,
                )
        texts_by_round: list[list[str]] = []
        empty = 0
        for round_index in range(args.repeat):
            results = transcribe_probe_paths(send_paths, source_paths, settings, pad_notes)
            round_texts: list[str] = []
            for source, sent, result in zip(source_paths, send_paths, results):
                extra = {"round": round_index + 1} if args.repeat > 1 else None
                if args.pad:
                    extra = {**(extra or {}), **note_by_source[str(source)]}
                    extra["audio"] = str(source)
                text = _print_result(sent if not args.pad else source, result, extra)
                round_texts.append(text)
                if round_index == 0:
                    empty += not bool(text)
            texts_by_round.append(round_texts)
    except Exception as exc:  # noqa: BLE001 - probe must print a concise operational error
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1

    unstable = 0
    if args.repeat > 1:
        first = texts_by_round[0]
        for later in texts_by_round[1:]:
            for left, right in zip(first, later):
                unstable += left != right

    summary = {
        "ok": empty == 0 and unstable == 0,
        "total": len(source_paths),
        "non_empty": len(source_paths) - empty,
        "empty": empty,
        "repeat": args.repeat,
        "unstable": unstable,
        "pad": bool(args.pad),
        "api_bases": list(settings.get("api_bases") or []),
    }
    print(json.dumps({"summary": summary}, ensure_ascii=False), file=sys.stderr)
    if empty:
        return 2
    if unstable:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
