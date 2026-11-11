#!/usr/bin/env python3
# coding=utf-8
"""TTS 后处理音频目录：子文件夹名=真实转写，用 5555 qwen-base 转写并汇总错例。

目录约定::

  /data2/data-cp/xijujun/Project/TTS/后处理音频/
    不需要/xxx.wav
    暂时不用/yyy.wav

每个一级子文件夹名就是该目录下全部音频的金标文本。脚本调用本机
``http://127.0.0.1:5555`` 的 vLLM ``/v1/chat/completions``（与
``scripts/run_qwen_asr_batch.py`` 同口径）。

计分：先 ``plain_transcript_text``，再删
``configs/normalization/fenshen_1015_filler_tokens.yaml`` 语气词；
删完后编辑距离为 0 不算 mismatch（只差语气词不算错）。

``qwen_base_mismatch.parquet`` 可直接给 audio-label-master「转换 parquet」导入：

- 数据单位选整段 / legacy（``is_vad_segment=false``）
- 播放音频：``audio.resampled_16k``
- 预标注来源勾选 ``tts_gold``（文本=子文件夹名）
- 原始机转参考勾选 ``qwen_base``（也可认顶列 ``candidate_text``）

产物（默认 ``tmp/tts_folder_qwen_base/``）::

  qwen_base_asr.parquet          全量 ASR 断点（可续跑）
  qwen_base_eval.parquet         带字准/对错标记的评测表
  qwen_base_mismatch.parquet     标注导入包（真错 / 失败；不含纯语气词差）
  evaluation_tts_folder_qwen_base.xlsx
  summary.json

示例::

  export QWEN_ASR_API_BASE=http://127.0.0.1:5555
  export QWEN_ASR_MODEL=qwen3-asr
  python scripts/eval_tts_folder_qwen_base.py

  python scripts/eval_tts_folder_qwen_base.py \\
    --audio-root /data2/data-cp/xijujun/Project/TTS/后处理音频 \\
    --output-dir tmp/1106/tts_post_qwen_base \\
    --concurrency 8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import run_qwen_asr_batch as qwen_batch  # noqa: E402
from audio_engine.core.artifacts import atomic_path, atomic_write_json  # noqa: E402
from audio_engine.core.manifest import file_sha256  # noqa: E402
from audio_engine.core.stage1.identity import fetch_model_identity  # noqa: E402
from audio_engine.core.transcript_reconcile import (  # noqa: E402
    levenshtein_ops,
    plain_transcript_text,
)

DEFAULT_AUDIO_ROOT = Path("/data2/data-cp/xijujun/Project/TTS/后处理音频")
DEFAULT_API_BASE = "http://127.0.0.1:5555"
DEFAULT_OUTPUT_DIR = ROOT / "tmp" / "tts_folder_qwen_base"
DEFAULT_CONFIG = ROOT / "configs" / "asr" / "qwen_asr.yaml"
FILLER_TOKENS_YAML = ROOT / "configs" / "normalization" / "fenshen_1015_filler_tokens.yaml"
AUDIO_EXTS = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".pcm", ".raw"}
# 标注平台：预标注来源 / 原始机转参考可选的 transcripts 键。
PREANNOTATION_KEY = "tts_gold"
QWEN_REF_KEY = "qwen_base"


def _info(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}", flush=True)


def _cell(value: Any) -> str:
    if value is None:
        return ""
    try:
        if value != value:  # NaN
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text.lower() in {"nan", "none"}:
        return ""
    return text


def load_filler_tokens(path: Path | None = None) -> tuple[str, ...]:
    """读取语气词 yaml。删除时按字数从长到短，与 speedup_negative_qwen_diff 一致。"""
    target = FILLER_TOKENS_YAML if path is None else path
    if not target.is_file():
        raise SystemExit(f"[ERROR] 缺少语气词表: {target}")
    raw = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    tokens = raw.get("tokens") if isinstance(raw, dict) else None
    if not isinstance(tokens, list) or not tokens:
        raise SystemExit(f"[ERROR] 语气词表 tokens 为空: {target}")
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in tokens:
        text = str(item).strip()
        if not text:
            raise SystemExit(f"[ERROR] 语气词表含空词: {target}")
        if text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
    return tuple(cleaned)


_FILLER_TOKENS = load_filler_tokens()
_FILLER_SORTED = tuple(sorted(_FILLER_TOKENS, key=len, reverse=True))


def drop_spoken_tokens(text: str) -> str:
    dropped = str(text or "")
    for token in _FILLER_SORTED:
        dropped = dropped.replace(token, "")
    return dropped


def comparable_text(text: str) -> str:
    """计分文本：plain 后再删语气词。"""
    return drop_spoken_tokens(plain_transcript_text(text))


def stable_id(folder: str, audio_path: Path, root: Path) -> str:
    """路径相对根目录的稳定 id；过长时附短 hash，避免 Excel/工具截断混淆。"""
    try:
        rel = audio_path.resolve().relative_to(root.resolve())
    except ValueError:
        rel = Path(folder) / audio_path.name
    text = rel.as_posix()
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    stem = f"{folder}/{audio_path.stem}"
    if len(stem) > 180:
        return f"{folder[:40]}/{audio_path.stem[:80]}_{digest}"
    return f"{stem}__{digest}"


def collect_items(audio_root: Path) -> list[dict[str, str]]:
    if not audio_root.is_dir():
        raise SystemExit(f"[ERROR] 音频根目录不存在: {audio_root}")
    items: list[dict[str, str]] = []
    folders = sorted(path for path in audio_root.iterdir() if path.is_dir())
    if not folders:
        raise SystemExit(f"[ERROR] 根目录下没有子文件夹: {audio_root}")
    for folder in folders:
        gold = folder.name.strip()
        if not gold:
            _warn(f"跳过空名文件夹: {folder}")
            continue
        files = sorted(
            path
            for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in AUDIO_EXTS
        )
        if not files:
            _warn(f"文件夹无音频: {folder}")
            continue
        for path in files:
            items.append(
                {
                    "id": stable_id(gold, path, audio_root),
                    "folder": gold,
                    "gold_text": gold,
                    "wav_path": str(path.resolve()),
                }
            )
    if not items:
        raise SystemExit(f"[ERROR] 未找到任何音频文件: {audio_root}")
    return items


def load_yaml_settings(config: Path) -> dict[str, Any]:
    if not config.is_file():
        return {}
    raw = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    return raw if isinstance(raw, dict) else {}


def build_settings(args: argparse.Namespace) -> dict[str, Any]:
    settings = load_yaml_settings(args.config)
    settings["api_base"] = (
        args.api_base
        or os.environ.get("QWEN_ASR_API_BASE")
        or settings.get("api_base")
        or DEFAULT_API_BASE
    )
    settings["api_key"] = (
        args.api_key
        or os.environ.get("QWEN_ASR_API_KEY")
        or settings.get("api_key")
        or "dummy"
    )
    settings["model"] = (
        args.model
        or os.environ.get("QWEN_ASR_MODEL")
        or settings.get("model")
        or "qwen3-asr"
    )
    if args.concurrency is not None:
        settings["concurrency"] = args.concurrency
    else:
        settings["concurrency"] = int(settings.get("concurrency") or 4)
    if args.timeout is not None:
        settings["timeout"] = args.timeout
    else:
        settings["timeout"] = float(settings.get("timeout") or 120)
    settings.setdefault("temperature", 0)
    if args.no_context:
        settings["context"] = ""
        settings["prompt"] = ""
    return settings


def ensure_qwen_reachable(api_base: str, model: str | None, *, timeout_s: float = 10.0) -> list[str]:
    try:
        identity = fetch_model_identity(api_base, timeout_s=timeout_s)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"[ERROR] Qwen 端口未联通: {api_base} ({exc})") from exc
    model_ids = [str(item) for item in identity.model_ids]
    if model and not identity.matches(model):
        raise SystemExit(
            f"[ERROR] Qwen 已连通，但没有模型 {model!r}。"
            f" 实际 ids={model_ids} @ {api_base}"
        )
    _info(f"qwen reachable models={model_ids} @ {api_base}")
    return model_ids


def score_pair(gold: str, hyp: str) -> dict[str, Any]:
    """plain + 删语气词后编辑距离为 0 才算正确；只差语气词不算 mismatch。"""
    gold_plain = plain_transcript_text(gold)
    hyp_plain = plain_transcript_text(hyp)
    gold_score = comparable_text(gold)
    hyp_score = comparable_text(hyp)
    ops = levenshtein_ops(gold_score, hyp_score)
    edits = int(ops["total"] or 0)
    ref_len = int(ops["ref_len"] or 0)
    exact = int(edits == 0)
    if exact:
        char_acc = 1.0
    elif ref_len <= 0:
        char_acc = 0.0
    else:
        char_acc = round(max(0.0, 1.0 - edits / ref_len), 6)
    plain_diff = int(gold_plain != hyp_plain)
    filler_only_diff = int(exact == 1 and plain_diff == 1)
    return {
        "gold_plain": gold_plain,
        "qwen_plain": hyp_plain,
        "gold_score": gold_score,
        "qwen_score": hyp_score,
        "edit_distance": edits,
        "ref_chars": ref_len,
        "char_acc": char_acc,
        "exact_correct": exact,
        "is_mismatch": int(exact != 1),
        "plain_diff": plain_diff,
        "filler_only_diff": filler_only_diff,
    }


def load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        return {}
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"[ERROR] 无法读取断点 parquet: {path} ({exc})") from exc
    out: dict[str, dict[str, Any]] = {}
    for row in frame.to_dict(orient="records"):
        sample_id = _cell(row.get("id"))
        if not sample_id:
            continue
        out[sample_id] = {key: row.get(key) for key in frame.columns}
    return out


def write_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    with atomic_path(path) as tmp:
        frame.to_parquet(tmp, index=False)


def row_from_item(
    item: dict[str, str],
    *,
    hyp: str = "",
    language: str = "",
    status: str = "pending",
    error: str = "",
    model: str = "",
    api_base: str = "",
) -> dict[str, Any]:
    if status == "ok":
        scored = score_pair(item["gold_text"], hyp)
    else:
        gold_plain = plain_transcript_text(item["gold_text"])
        gold_score = drop_spoken_tokens(gold_plain)
        scored = {
            "gold_plain": gold_plain,
            "qwen_plain": "",
            "gold_score": gold_score,
            "qwen_score": "",
            "edit_distance": None,
            "ref_chars": len(gold_score),
            "char_acc": None,
            "exact_correct": 0,
            "is_mismatch": 1,
            "plain_diff": 1,
            "filler_only_diff": 0,
        }
    return {
        "id": item["id"],
        "folder": item["folder"],
        "gold_text": item["gold_text"],
        "wav_path": item["wav_path"],
        "qwen_base_text": hyp,
        "qwen_language": language,
        "status": status,
        "error": error,
        "model": model,
        "api_base": api_base,
        **scored,
    }


def resolve_sha256(wav_path: str, cache: dict[str, str]) -> str:
    if wav_path in cache:
        return cache[wav_path]
    path = Path(wav_path)
    if not path.is_file():
        raise FileNotFoundError(f"音频不存在，无法算 sha256: {wav_path}")
    digest = file_sha256(path)
    cache[wav_path] = digest
    return digest


def build_label_row(row: dict[str, Any], *, sha_cache: dict[str, str]) -> dict[str, Any]:
    """audio-label-master 转换 parquet 行：预标注=文件夹名，机转参考可选 qwen_base。"""
    gold = _cell(row.get("gold_text")) or _cell(row.get("folder"))
    hyp = _cell(row.get("qwen_base_text"))
    wav_path = _cell(row.get("wav_path"))
    sample_id = _cell(row.get("id"))
    sha = resolve_sha256(wav_path, sha_cache) if wav_path else ""
    transcripts = {
        PREANNOTATION_KEY: {"text": gold},
        QWEN_REF_KEY: {"text": hyp},
    }
    audio = {"resampled_16k": wav_path} if wav_path else {}
    status = _cell(row.get("status")) or "failed"
    bucket = "failed" if status != "ok" else "mismatch"
    return {
        "id": sample_id,
        "sha256": sha,
        "is_vad_segment": False,
        "folder": _cell(row.get("folder")),
        "label": gold,
        "gold_text": gold,
        "candidate_text": hyp,
        "qwen_base_text": hyp,
        "char_acc": row.get("char_acc"),
        "exact_correct": int(row.get("exact_correct") or 0),
        "edit_distance": row.get("edit_distance"),
        "ref_chars": row.get("ref_chars"),
        "gold_plain": _cell(row.get("gold_plain")),
        "qwen_plain": _cell(row.get("qwen_plain")),
        "gold_score": _cell(row.get("gold_score")),
        "qwen_score": _cell(row.get("qwen_score")),
        "filler_only_diff": int(row.get("filler_only_diff") or 0),
        "bucket": bucket,
        "status": status,
        "error": _cell(row.get("error")),
        "model": _cell(row.get("model")),
        "api_base": _cell(row.get("api_base")),
        "wav_path": wav_path,
        # 整段音频：播放走 audio.resampled_16k；source_path 留空，避免被当成父音频。
        "source_path": None,
        "audio": json.dumps(audio, ensure_ascii=False),
        "transcripts": json.dumps(transcripts, ensure_ascii=False),
        "preannotation_key": PREANNOTATION_KEY,
        "reference_key": QWEN_REF_KEY,
    }


def transcribe_all(
    items: list[dict[str, str]],
    settings: dict[str, Any],
    *,
    checkpoint: Path,
    force: bool,
    limit: int | None,
    checkpoint_every: int,
) -> list[dict[str, Any]]:
    previous = {} if force else load_checkpoint(checkpoint)
    rows_by_id: dict[str, dict[str, Any]] = {}
    pending: list[dict[str, str]] = []
    for item in items:
        old = previous.get(item["id"])
        if old and not force and _cell(old.get("status")) in {"ok", "failed"}:
            # 路径或金标变更时仍重跑。
            if (
                _cell(old.get("wav_path")) == item["wav_path"]
                and _cell(old.get("gold_text")) == item["gold_text"]
            ):
                rows_by_id[item["id"]] = dict(old)
                continue
        pending.append(item)

    if limit is not None:
        pending = pending[: max(0, limit)]

    _info(
        f"total={len(items)} resume={len(rows_by_id)} pending={len(pending)} "
        f"concurrency={settings.get('concurrency')}"
    )
    if not pending:
        return [rows_by_id[item["id"]] for item in items if item["id"] in rows_by_id]

    started = time.perf_counter()
    done = 0
    since_checkpoint = 0
    concurrency = max(1, int(settings.get("concurrency") or 4))

    def one(item: dict[str, str]) -> dict[str, Any]:
        try:
            result = qwen_batch.call_vllm_chat_asr(item["wav_path"], settings)
            return row_from_item(
                item,
                hyp=_cell(result.get("text")),
                language=_cell(result.get("language")),
                status="ok",
                model=_cell(settings.get("model")),
                api_base=_cell(settings.get("api_base")),
            )
        except Exception as exc:  # noqa: BLE001 - 单条失败不拖垮整批
            return row_from_item(
                item,
                status="failed",
                error=str(exc),
                model=_cell(settings.get("model")),
                api_base=_cell(settings.get("api_base")),
            )

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(one, item): item for item in pending}
        for future in as_completed(futures):
            row = future.result()
            rows_by_id[row["id"]] = row
            done += 1
            since_checkpoint += 1
            if row["status"] != "ok":
                _warn(f"failed id={row['id']} error={row['error']}")
            if since_checkpoint >= checkpoint_every:
                ordered = [rows_by_id[item["id"]] for item in items if item["id"] in rows_by_id]
                write_parquet(ordered, checkpoint)
                since_checkpoint = 0
                elapsed = time.perf_counter() - started
                rate = done / elapsed if elapsed else 0.0
                _info(f"checkpoint {done}/{len(pending)} rate={rate:.2f}/s -> {checkpoint}")

    ordered = []
    for item in items:
        if item["id"] in rows_by_id:
            ordered.append(rows_by_id[item["id"]])
        elif limit is None:
            # limit 模式下未跑到的条不写。
            pass
    write_parquet(ordered, checkpoint)
    return ordered


def summarize(rows: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    ok_rows = [row for row in rows if _cell(row.get("status")) == "ok"]
    failed_rows = [row for row in rows if _cell(row.get("status")) == "failed"]
    mismatch_rows = [
        row
        for row in ok_rows
        if int(row.get("is_mismatch") or 0) == 1
    ]
    filler_only_rows = [
        row
        for row in ok_rows
        if int(row.get("filler_only_diff") or 0) == 1
    ]
    total_edits = sum(int(row.get("edit_distance") or 0) for row in ok_rows)
    total_ref = sum(int(row.get("ref_chars") or 0) for row in ok_rows)
    correct = sum(1 for row in ok_rows if int(row.get("exact_correct") or 0) == 1)
    overall_char = (
        round(max(0.0, 1.0 - total_edits / total_ref), 6) if total_ref > 0 else (1.0 if total_edits == 0 else 0.0)
    )
    overall = {
        "scope": "overall",
        "样本数": len(ok_rows),
        "正确数(字准=1)": correct,
        "正确率": f"{(correct / len(ok_rows) * 100):.2f}%" if ok_rows else "",
        "错误数": len(mismatch_rows),
        "仅语气词差": len(filler_only_rows),
        "失败数": len(failed_rows),
        "总编辑距离": total_edits,
        "总基准字数": total_ref,
        "总字准": f"{overall_char * 100:.2f}%",
    }

    by_folder_counter: Counter[str] = Counter()
    by_folder_correct: Counter[str] = Counter()
    by_folder_edits: Counter[str] = Counter()
    by_folder_chars: Counter[str] = Counter()
    by_folder_fail: Counter[str] = Counter()
    for row in rows:
        folder = _cell(row.get("folder"))
        if _cell(row.get("status")) == "failed":
            by_folder_fail[folder] += 1
            continue
        if _cell(row.get("status")) != "ok":
            continue
        by_folder_counter[folder] += 1
        by_folder_edits[folder] += int(row.get("edit_distance") or 0)
        by_folder_chars[folder] += int(row.get("ref_chars") or 0)
        if int(row.get("exact_correct") or 0) == 1:
            by_folder_correct[folder] += 1

    folder_rows: list[dict[str, Any]] = []
    for folder in sorted(set(by_folder_counter) | set(by_folder_fail)):
        n = by_folder_counter[folder]
        c = by_folder_correct[folder]
        edits = by_folder_edits[folder]
        chars = by_folder_chars[folder]
        char_acc = (1.0 - edits / chars) if chars > 0 else (1.0 if edits == 0 else 0.0)
        folder_rows.append(
            {
                "folder": folder,
                "gold_text": folder,
                "样本数": n,
                "正确数(字准=1)": c,
                "正确率": f"{(c / n * 100):.2f}%" if n else "",
                "错误数": n - c,
                "失败数": by_folder_fail[folder],
                "总编辑距离": edits,
                "总基准字数": chars,
                "总字准": f"{char_acc * 100:.2f}%",
            }
        )

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total": len(rows),
        "ok": len(ok_rows),
        "failed": len(failed_rows),
        "mismatch": len(mismatch_rows),
        "filler_only_diff": len(filler_only_rows),
        "exact_correct": correct,
        "exact_rate": (correct / len(ok_rows)) if ok_rows else None,
        "total_edit_distance": total_edits,
        "total_ref_chars": total_ref,
        "char_accuracy": overall_char,
        "folders": len(folder_rows),
        "score_rule": "plain + drop fenshen_1015_filler_tokens; edits==0 => correct",
        "convert_import": {
            "file": "qwen_base_mismatch.parquet",
            "input_unit": "legacy / 整段音频",
            "playable_audio": "audio.resampled_16k",
            "preannotation_key": PREANNOTATION_KEY,
            "preannotation_text": "子文件夹名（真实转写）",
            "reference_key": QWEN_REF_KEY,
            "candidate_text": "qwen_base 转写原文",
        },
    }
    accuracy = pd.DataFrame([overall])
    by_folder = pd.DataFrame(folder_rows)
    return accuracy, by_folder, summary


def write_xlsx(
    *,
    path: Path,
    accuracy: pd.DataFrame,
    by_folder: pd.DataFrame,
    detail: pd.DataFrame,
    mismatch: pd.DataFrame,
    notes: pd.DataFrame,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        accuracy.to_excel(writer, index=False, sheet_name="正确率")
        by_folder.to_excel(writer, index=False, sheet_name="按文件夹")
        mismatch.to_excel(writer, index=False, sheet_name="错例")
        detail.to_excel(writer, index=False, sheet_name="明细")
        notes.to_excel(writer, index=False, sheet_name="说明")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TTS 子文件夹名=金标，调用 5555 qwen-base 转写并汇总 xlsx/parquet"
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=DEFAULT_AUDIO_ROOT,
        help="TTS 后处理音频根目录，一级子文件夹名即真实转写",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="输出目录，默认 tmp/tts_folder_qwen_base",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Qwen ASR yaml")
    parser.add_argument("--api-base", default=None, help="覆盖 QWEN_ASR_API_BASE，默认 5555")
    parser.add_argument("--api-key", default=None, help="覆盖 QWEN_ASR_API_KEY")
    parser.add_argument("--model", default=None, help="须等于 vLLM --served-model-name")
    parser.add_argument("--concurrency", type=int, default=None, help="并发请求数")
    parser.add_argument("--timeout", type=float, default=None, help="单条 HTTP 超时秒数")
    parser.add_argument("--checkpoint-every", type=int, default=50, help="每处理多少条落一次盘")
    parser.add_argument("--limit", type=int, default=None, help="只转写这么多尚未完成的样本")
    parser.add_argument("--force", action="store_true", help="忽略断点，全部重跑")
    parser.add_argument(
        "--no-context",
        action="store_true",
        help="不把 configs/asr/qwen_asr.yaml 里的 vocabulary context 发给模型",
    )
    parser.add_argument(
        "--skip-asr",
        action="store_true",
        help="不请求服务，只根据已有 qwen_base_asr.parquet 重算评测表",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.checkpoint_every < 1:
        raise SystemExit("[ERROR] --checkpoint-every 必须 >= 1")

    audio_root = args.audio_root
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    asr_path = output_dir / "qwen_base_asr.parquet"
    eval_path = output_dir / "qwen_base_eval.parquet"
    mismatch_path = output_dir / "qwen_base_mismatch.parquet"
    xlsx_path = output_dir / "evaluation_tts_folder_qwen_base.xlsx"
    summary_path = output_dir / "summary.json"

    items = collect_items(audio_root)
    folder_counts = Counter(item["folder"] for item in items)
    _info(f"audio_root={audio_root} files={len(items)} folders={len(folder_counts)}")
    for name, count in folder_counts.most_common():
        _info(f"  folder={name!r} n={count}")

    settings = build_settings(args)
    if args.skip_asr:
        rows = list(load_checkpoint(asr_path).values())
        if not rows:
            raise SystemExit(f"[ERROR] --skip-asr 但找不到断点: {asr_path}")
        # 保持扫描顺序。
        by_id = {_cell(row.get("id")): row for row in rows}
        ordered = [by_id[item["id"]] for item in items if item["id"] in by_id]
        rows = ordered
        _info(f"skip-asr loaded={len(rows)} from {asr_path}")
    else:
        ensure_qwen_reachable(str(settings["api_base"]), str(settings.get("model") or "") or None)
        rows = transcribe_all(
            items,
            settings,
            checkpoint=asr_path,
            force=args.force,
            limit=args.limit,
            checkpoint_every=args.checkpoint_every,
        )

    # 失败行也进入评测表，便于排查。
    scored_rows: list[dict[str, Any]] = []
    for row in rows:
        status = _cell(row.get("status"))
        if status == "ok":
            scored = score_pair(_cell(row.get("gold_text")), _cell(row.get("qwen_base_text")))
            merged = dict(row)
            merged.update(scored)
            scored_rows.append(merged)
        else:
            scored_rows.append(row_from_item(
                {
                    "id": _cell(row.get("id")),
                    "folder": _cell(row.get("folder")),
                    "gold_text": _cell(row.get("gold_text")),
                    "wav_path": _cell(row.get("wav_path")),
                },
                hyp=_cell(row.get("qwen_base_text")),
                language=_cell(row.get("qwen_language")),
                status=status or "failed",
                error=_cell(row.get("error")),
                model=_cell(row.get("model")) or _cell(settings.get("model")),
                api_base=_cell(row.get("api_base")) or _cell(settings.get("api_base")),
            ))

    write_parquet(scored_rows, eval_path)
    mismatch_source = [
        row
        for row in scored_rows
        if _cell(row.get("status")) != "ok" or int(row.get("is_mismatch") or 0) == 1
    ]
    sha_cache: dict[str, str] = {}
    label_rows = [build_label_row(row, sha_cache=sha_cache) for row in mismatch_source]
    write_parquet(label_rows, mismatch_path)

    accuracy, by_folder, summary = summarize(scored_rows)
    detail_cols = [
        "id",
        "folder",
        "gold_text",
        "qwen_base_text",
        "gold_plain",
        "qwen_plain",
        "gold_score",
        "qwen_score",
        "char_acc",
        "exact_correct",
        "is_mismatch",
        "filler_only_diff",
        "edit_distance",
        "ref_chars",
        "status",
        "error",
        "wav_path",
        "model",
        "api_base",
    ]
    detail = pd.DataFrame(scored_rows)
    for col in detail_cols:
        if col not in detail.columns:
            detail[col] = ""
    detail = detail[detail_cols]
    xlsx_mismatch_cols = [
        "id",
        "folder",
        "label",
        "candidate_text",
        "char_acc",
        "exact_correct",
        "edit_distance",
        "gold_score",
        "qwen_score",
        "bucket",
        "status",
        "error",
        "wav_path",
        "preannotation_key",
        "reference_key",
    ]
    mismatch_df = pd.DataFrame(label_rows)
    for col in xlsx_mismatch_cols:
        if col not in mismatch_df.columns:
            mismatch_df[col] = ""
    if not mismatch_df.empty:
        mismatch_df = mismatch_df[xlsx_mismatch_cols]
    else:
        mismatch_df = pd.DataFrame(columns=xlsx_mismatch_cols)

    notes = pd.DataFrame(
        [
            ("音频根目录", str(audio_root)),
            ("金标口径", "一级子文件夹名 = 该目录下音频的真实转写"),
            ("模型服务", str(settings.get("api_base"))),
            ("served-model-name", str(settings.get("model"))),
            ("vocabulary context", "默认沿用 configs/asr/qwen_asr.yaml；--no-context 关闭"),
            (
                "正确判定",
                "plain 后删 fenshen_1015_filler_tokens；编辑距离为 0 算正确。"
                "只差语气词不算 mismatch，不进标注包",
            ),
            ("字准", "1 - 编辑距离 / 删语气词后金标字数"),
            ("标注包", str(mismatch_path)),
            ("预标注来源键", f"transcripts.{PREANNOTATION_KEY} = 子文件夹名"),
            ("原始机转参考键", f"transcripts.{QWEN_REF_KEY} / candidate_text = qwen 转写"),
            ("播放音频", "audio.resampled_16k；数据单位选 legacy/整段"),
            ("全量 parquet", str(eval_path)),
            ("ASR 断点", str(asr_path)),
            ("样本数", str(summary["total"])),
            ("成功转写", str(summary["ok"])),
            ("转写失败", str(summary["failed"])),
            ("真 mismatch", str(summary["mismatch"])),
            ("仅语气词差(已排除)", str(summary["filler_only_diff"])),
        ],
        columns=["字段/口径", "说明"],
    )
    write_xlsx(
        path=xlsx_path,
        accuracy=accuracy,
        by_folder=by_folder,
        detail=detail,
        mismatch=mismatch_df,
        notes=notes,
    )

    summary_payload = {
        **summary,
        "audio_root": str(audio_root),
        "output_dir": str(output_dir),
        "api_base": settings.get("api_base"),
        "model": settings.get("model"),
        "asr_parquet": str(asr_path),
        "eval_parquet": str(eval_path),
        "mismatch_parquet": str(mismatch_path),
        "xlsx": str(xlsx_path),
        "folder_counts": dict(folder_counts),
    }
    atomic_write_json(summary_path, summary_payload)

    _info(
        f"done ok={summary['ok']} mismatch={summary['mismatch']} "
        f"filler_only={summary['filler_only_diff']} failed={summary['failed']} "
        f"exact_rate={summary['exact_rate']}"
    )
    _info(f"xlsx={xlsx_path}")
    _info(f"eval_parquet={eval_path}")
    _info(f"mismatch_parquet={mismatch_path}")
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
