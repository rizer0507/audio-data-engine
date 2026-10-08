#!/usr/bin/env python3
"""semantic_risk 人工听写标注（VS Code Remote / Mac Mini 播放器场景）。

流程：
  1) prepare  — 从五分类 parquet 筛 semantic_risk，解析 wav，写队列 + 可选软链目录
  2) label    — 交互听写：自动打开下一条 wav（cursor/code），输入转写写入 annotations.parquet；
                 完成本条后尝试关闭当前 wav 预览（优先关闭助手扩展，否则热键兜底）
  3) merge    — 按 sample_id + original_audio_sha256 回填到五分类 parquet（未标注行不变）

合并键与 annotation_v3 / selection_v3 一致，便于后续正式 review import。

示例::

  export BATCH=20260813-1
  python scripts/annotate_semantic_risk_transcripts.py prepare --batch "$BATCH"
  python scripts/annotate_semantic_risk_transcripts.py label --batch "$BATCH" --annotator-id ann1
  python scripts/annotate_semantic_risk_transcripts.py merge --batch "$BATCH"
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from audio_engine.core.annotation_v3.types import (
    GOLD_KIND_NON_SPEECH,
    GOLD_KIND_SPEECH,
    STATE_ANNOTATED,
)
from audio_engine.core.manifest import Manifest
from audio_engine.core.sample import Sample
from audio_engine.core.selection_v3.input_contract import original_audio_sha256

DEFAULT_RULE_STEM = "classified_five_class_v3"
CATEGORY = "semantic_risk"
AUDIO_KEYS = ("resampled_16k", "pcm_wav", "pcm_to_wav", "raw", "pcm_wav_path")
CLOSER_EXT_DIR = Path(__file__).resolve().parent / "annotate_audio_closer_ext"
CLOSER_HOST = "127.0.0.1"
CLOSER_PORT = 17321
PLAYER_CLI_CANDIDATES = ("cursor", "code", "code-insiders")

HELP_LABEL = """
命令（单独一行）：
  /skip     跳过本条（可稍后回来）
  /empty    确认无语音（gold_kind=non_speech, gold_text 空串）
  /replay   重新打开当前 wav
  /show     再打印一次路径与候选文
  /quit     保存进度并退出
  /help     显示本帮助

直接输入转写文本后回车 = 写入本条并自动进入下一条。
完成本条（转写 /empty /skip）后会尝试关闭当前 wav 预览。
""".strip()


@dataclass
class OpenedWav:
    path: Path
    via: str


# ──────────────────────────────────────────────
# paths
# ──────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def resolve_path(raw: str | Path) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    return path


def default_classified_path(batch: str, *, rule_stem: str = DEFAULT_RULE_STEM) -> Path:
    return ROOT / "datasets" / "stage1" / "derived" / f"{rule_stem}_{batch}.parquet"


def default_work_dir(batch: str) -> Path:
    return ROOT / "datasets" / "stage1" / "review" / "semantic_risk_transcripts" / batch


def queue_path(work_dir: Path) -> Path:
    return work_dir / "queue.parquet"


def annotations_path(work_dir: Path) -> Path:
    return work_dir / "annotations.parquet"


def wav_dir(work_dir: Path) -> Path:
    return work_dir / "wav"


def playlist_path(work_dir: Path) -> Path:
    return work_dir / "playlist.md"


def default_merged_path(batch: str, *, rule_stem: str = DEFAULT_RULE_STEM) -> Path:
    return ROOT / "datasets" / "stage1" / "derived" / f"{rule_stem}_{batch}_with_gold.parquet"


# ──────────────────────────────────────────────
# wav resolve
# ──────────────────────────────────────────────

def resolve_sample_wav(sample: Sample) -> Path | None:
    candidates: list[Path] = []
    for key in AUDIO_KEYS:
        raw = sample.audio.get(key)
        if raw:
            candidates.append(Path(str(raw)))
    if sample.source_path:
        sp = Path(sample.source_path)
        if sp.suffix.lower() in {".wav", ".flac", ".ogg", ".mp3", ".m4a"}:
            candidates.append(sp)
    for cand in candidates:
        try:
            if cand.is_file():
                return cand.resolve()
        except OSError:
            continue
    return None


def category_of(sample: Sample) -> str:
    return str(sample.labels.get("category") or sample.labels.get("type") or "").strip()


# ──────────────────────────────────────────────
# parquet helpers
# ──────────────────────────────────────────────

def atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=str(path.parent),
        suffix=".parquet.tmp",
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        df.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def load_annotations(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(
            columns=[
                "sample_id",
                "original_audio_sha256",
                "batch",
                "category",
                "queue_index",
                "wav_path",
                "candidate_text",
                "gold_text",
                "gold_kind",
                "decision",
                "status",
                "annotator_id",
                "annotated_at",
            ]
        )
    df = pd.read_parquet(path)
    if "sample_id" not in df.columns:
        raise ValueError(f"annotations parquet 缺少 sample_id: {path}")
    return df


def upsert_annotation(df: pd.DataFrame, row: dict[str, Any]) -> pd.DataFrame:
    sid = str(row["sample_id"])
    mask = df["sample_id"].astype(str) == sid if len(df) else pd.Series([], dtype=bool)
    new_df = pd.DataFrame([row])
    if mask.any():
        df = df.loc[~mask].copy()
    return pd.concat([df, new_df], ignore_index=True)


def _closer_base_url() -> str:
    return f"http://{CLOSER_HOST}:{CLOSER_PORT}"


def closer_healthcheck(timeout: float = 0.4) -> bool:
    try:
        with urlopen(f"{_closer_base_url()}/health", timeout=timeout) as resp:
            return int(getattr(resp, "status", 200) or 200) == 200
    except (URLError, TimeoutError, OSError, ValueError):
        return False


def closer_install_hint() -> str:
    return (
        "一次性安装音频关闭助手（安装后请 Reload Window）：\n"
        f"  命令面板 → Extensions: Install from Location… → {CLOSER_EXT_DIR}\n"
        "或打包后: cursor --install-extension <vsix>"
    )


def close_wav_via_closer_ext(wav: Path | None, *, timeout: float = 1.5) -> str | None:
    """POST to annotate-audio-closer extension; return status or None if unavailable."""
    payload = json.dumps({"path": str(wav.resolve()) if wav else ""}).encode("utf-8")
    url = f"{_closer_base_url()}/close" if wav else f"{_closer_base_url()}/close-active"
    req = Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            data = {}
        closed = int(data.get("closed") or 0)
        return f"closed_via:ext closed={closed}"
    except (URLError, TimeoutError, OSError, ValueError):
        return None


def _focus_wav_in_editor(wav: Path, via: str | None) -> str | None:
    abs_path = str(wav.resolve())
    candidates = [via] if via in PLAYER_CLI_CANDIDATES else []
    candidates.extend(c for c in PLAYER_CLI_CANDIDATES if c not in candidates)
    for cli in candidates:
        if not cli:
            continue
        try:
            subprocess.Popen(
                [cli, "-r", abs_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            return cli
        except FileNotFoundError:
            continue
        except OSError:
            continue
    return None


def _send_close_hotkey() -> bool:
    """Best-effort Ctrl/Cmd+W to the IDE window. Works mainly on local GUI sessions."""
    if sys.platform == "darwin":
        script = (
            'tell application "System Events"\n'
            'set frontApp to name of first application process whose frontmost is true\n'
            'end tell\n'
            'tell application "System Events" to keystroke "w" using command down\n'
        )
        try:
            subprocess.run(
                ["osascript", "-e", script],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2,
            )
            return True
        except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
            return False

    if sys.platform == "win32":
        ps = (
            "Add-Type -AssemblyName System.Windows.Forms; "
            "[System.Windows.Forms.SendKeys]::SendWait('^w')"
        )
        try:
            subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=3,
            )
            return True
        except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
            return False

    # Linux local GUI: try xdotool if present
    try:
        subprocess.run(
            ["xdotool", "key", "ctrl+w"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        return True
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return False


def open_wav_for_remote_player(wav: Path) -> OpenedWav | None:
    """尽量在 VS Code / Cursor Remote 客户端打开 wav（触发内置播放器）。"""
    abs_path = str(wav.resolve())
    for cli in PLAYER_CLI_CANDIDATES:
        try:
            subprocess.Popen(
                [cli, abs_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            return OpenedWav(path=wav.resolve(), via=cli)
        except FileNotFoundError:
            continue
        except OSError:
            continue
    return None


def close_wav_for_remote_player(
    opened: OpenedWav | None,
    *,
    allow_hotkey: bool = True,
) -> str:
    """Close the previously opened wav preview after labeling one item."""
    if opened is None:
        return "no_open"
    wav = opened.path
    via_ext = close_wav_via_closer_ext(wav)
    if via_ext is not None:
        return via_ext

    focused = _focus_wav_in_editor(wav, opened.via)
    if focused and allow_hotkey:
        time.sleep(0.35)
        if _send_close_hotkey():
            return f"closed_via:hotkey after_focus:{focused}"
        return f"focus_only:{focused}"
    return "manual_close"


# ──────────────────────────────────────────────
# prepare
# ──────────────────────────────────────────────

def cmd_prepare(args: argparse.Namespace) -> int:
    batch = str(args.batch).strip()
    classified = resolve_path(args.classified) if args.classified else default_classified_path(
        batch, rule_stem=args.rule_stem
    )
    work_dir = resolve_path(args.work_dir) if args.work_dir else default_work_dir(batch)
    if not classified.is_file():
        raise FileNotFoundError(f"分类 parquet 不存在: {classified}")

    print(f"[INFO] 读取: {classified}")
    manifest = Manifest.load(classified)
    rows: list[dict[str, Any]] = []
    missing_wav = 0
    for sample in manifest.samples:
        if category_of(sample) != CATEGORY:
            continue
        ohash = original_audio_sha256(sample)
        wav = resolve_sample_wav(sample)
        if wav is None:
            missing_wav += 1
        rows.append(
            {
                "queue_index": 0,  # filled below
                "sample_id": sample.id,
                "original_audio_sha256": ohash,
                "batch": batch,
                "category": CATEGORY,
                "wav_path": str(wav) if wav else "",
                "source_path": sample.source_path,
                "candidate_text": str(sample.labels.get("candidate_text") or ""),
                "selected_family": str(sample.labels.get("selected_family") or ""),
                "duration": sample.duration,
                "status": "pending",
            }
        )

    if not rows:
        print(f"[WARN] 未找到 category={CATEGORY} 样本")
        return 1

    rows.sort(key=lambda r: r["sample_id"])
    for i, row in enumerate(rows, start=1):
        row["queue_index"] = i

    work_dir.mkdir(parents=True, exist_ok=True)
    qdf = pd.DataFrame(rows)
    q_path = queue_path(work_dir)
    atomic_write_parquet(qdf, q_path)
    print(f"[OK] 队列 {len(rows)} 条 → {q_path}")
    print(f"[INFO] 缺 wav 路径: {missing_wav}")

    if args.link_wav:
        out_wav = wav_dir(work_dir)
        if out_wav.exists() and args.overwrite_links:
            shutil.rmtree(out_wav)
        out_wav.mkdir(parents=True, exist_ok=True)
        linked = 0
        for row in rows:
            if not row["wav_path"]:
                continue
            src = Path(row["wav_path"])
            name = f"{int(row['queue_index']):05d}_{row['sample_id']}.wav"
            dst = out_wav / name
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            try:
                os.symlink(src, dst)
            except OSError:
                # Windows / 无 symlink 权限时退化为复制硬提示路径清单
                shutil.copy2(src, dst)
            linked += 1
        print(f"[OK] wav 软链/副本 {linked} → {out_wav}")
        print("     在 VS Code 资源管理器中打开该目录，可按序号点击播放。")

    lines = [
        f"# semantic_risk 听写播放列表 · `{batch}`",
        "",
        f"- 分类源: `{classified}`",
        f"- 队列: `{q_path}`",
        f"- 条数: {len(rows)}（缺 wav: {missing_wav}）",
        "",
        "| # | sample_id | wav | candidate_text |",
        "|---|-----------|-----|----------------|",
    ]
    for row in rows:
        wav_cell = f"`{row['wav_path']}`" if row["wav_path"] else "_(missing)_"
        cand = str(row["candidate_text"]).replace("|", "\\|")[:80]
        lines.append(
            f"| {row['queue_index']} | `{row['sample_id']}` | {wav_cell} | {cand} |"
        )
    pl = playlist_path(work_dir)
    pl.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] 播放清单 → {pl}")
    print()
    print("下一步:")
    print(
        f'  python scripts/annotate_semantic_risk_transcripts.py label '
        f'--batch "{batch}" --annotator-id <你的id>'
    )
    return 0


# ──────────────────────────────────────────────
# label
# ──────────────────────────────────────────────

def _print_item(row: pd.Series, *, done: int, total: int) -> None:
    print()
    print("=" * 72)
    print(f"[{done + 1}/{total}] queue=#{int(row['queue_index'])}  id={row['sample_id']}")
    print(f"sha256={row['original_audio_sha256']}")
    wav = str(row.get("wav_path") or "").strip()
    if wav:
        print(f"WAV: {wav}")
        print("     ↑ 若未自动打开，在 VS Code 资源管理器中点击该文件播放")
    else:
        print("WAV: (缺失路径 — 请对照 playlist.md / source_path)")
        print(f"source_path: {row.get('source_path', '')}")
    cand = str(row.get("candidate_text") or "")
    if cand:
        print(f"候选文(仅参考，勿照抄): {cand}")
    print("-" * 72)


def cmd_label(args: argparse.Namespace) -> int:
    batch = str(args.batch).strip()
    work_dir = resolve_path(args.work_dir) if args.work_dir else default_work_dir(batch)
    q_path = queue_path(work_dir)
    a_path = annotations_path(work_dir)
    if not q_path.is_file():
        raise FileNotFoundError(f"队列不存在，请先 prepare: {q_path}")

    queue = pd.read_parquet(q_path)
    if queue.empty:
        print("[WARN] 队列为空")
        return 1
    queue = queue.sort_values("queue_index").reset_index(drop=True)
    ann = load_annotations(a_path)
    done_ids = set()
    if len(ann):
        done_ids = set(
            ann.loc[ann["status"].astype(str).isin({"annotated", "empty"}), "sample_id"]
            .astype(str)
            .tolist()
        )
    # skipped 不算完成，可再做；若 --resume-skipped 则跳过 annotated/empty 即可

    annotator = str(args.annotator_id).strip()
    if not annotator:
        raise ValueError("必须提供 --annotator-id")

    pending = queue[~queue["sample_id"].astype(str).isin(done_ids)].reset_index(drop=True)
    if args.start_index is not None:
        pending = pending[pending["queue_index"] >= int(args.start_index)].reset_index(drop=True)

    total = len(queue)
    already = len(done_ids)
    print(f"[INFO] 队列 {total}，已标注/空语音 {already}，本次待处理 {len(pending)}")
    print(HELP_LABEL)
    if not pending.empty:
        print(f"[INFO] 从 queue_index={int(pending.iloc[0]['queue_index'])} 开始")

    if closer_healthcheck():
        print(f"[INFO] 音频关闭助手已就绪 (:{CLOSER_PORT})")
    else:
        print("[WARN] 未检测到音频关闭助手；完成本条后将尽量关闭预览。")
        print(closer_install_hint())

    opened: OpenedWav | None = None
    need_open = True

    def _open_current(wav_raw: str) -> None:
        nonlocal opened
        if opened is not None:
            close_status = close_wav_for_remote_player(opened)
            print(f"[close] {close_status}")
            opened = None
        if wav_raw and Path(wav_raw).is_file():
            opened = open_wav_for_remote_player(Path(wav_raw))
            if opened is not None:
                print(f"[play] opened_via:{opened.via}")
            else:
                print("[play] manual_click")
        elif wav_raw:
            print(f"[WARN] wav 文件不存在: {wav_raw}")

    def _finish_current() -> None:
        nonlocal opened
        if opened is None:
            return
        close_status = close_wav_for_remote_player(opened)
        print(f"[close] {close_status}")
        opened = None

    i = 0
    while i < len(pending):
        row = pending.iloc[i]
        sid = str(row["sample_id"])
        wav_raw = str(row.get("wav_path") or "").strip()
        if need_open:
            _print_item(row, done=already + i, total=total)
            _open_current(wav_raw)
            need_open = False

        try:
            user_in = input("转写> ").rstrip("\n")
        except (EOFError, KeyboardInterrupt):
            _finish_current()
            print("\n[INFO] 中断，已保存进度")
            break

        text = user_in.strip()
        if text in {"/help", "help", "?"}:
            print(HELP_LABEL)
            continue
        if text in {"/quit", "/exit", "quit", "exit"}:
            _finish_current()
            print("[INFO] 退出")
            break
        if text in {"/replay", "/play"}:
            _open_current(wav_raw)
            continue
        if text == "/show":
            _print_item(row, done=already + i, total=total)
            continue
        if text in {"/skip", "skip"}:
            rec = {
                "sample_id": sid,
                "original_audio_sha256": str(row["original_audio_sha256"]),
                "batch": batch,
                "category": CATEGORY,
                "queue_index": int(row["queue_index"]),
                "wav_path": wav_raw,
                "candidate_text": str(row.get("candidate_text") or ""),
                "gold_text": "",
                "gold_kind": "",
                "decision": "skip",
                "status": "skipped",
                "annotator_id": annotator,
                "annotated_at": _now_iso(),
            }
            ann = upsert_annotation(ann, rec)
            atomic_write_parquet(ann, a_path)
            print(f"[skip] {sid}  （可稍后重跑 label 覆盖）")
            _finish_current()
            i += 1
            need_open = True
            continue

        if text in {"/empty", "empty"}:
            gold_text = ""
            gold_kind = GOLD_KIND_NON_SPEECH
            decision = "empty"
            status = "empty"
        else:
            if not text:
                print("[WARN] 空输入无效；无语音请用 /empty，跳过用 /skip")
                continue
            gold_text = text
            gold_kind = GOLD_KIND_SPEECH
            decision = "accept"
            status = "annotated"

        rec = {
            "sample_id": sid,
            "original_audio_sha256": str(row["original_audio_sha256"]),
            "batch": batch,
            "category": CATEGORY,
            "queue_index": int(row["queue_index"]),
            "wav_path": wav_raw,
            "candidate_text": str(row.get("candidate_text") or ""),
            "gold_text": gold_text,
            "gold_kind": gold_kind,
            "decision": decision,
            "status": status,
            "annotator_id": annotator,
            "annotated_at": _now_iso(),
        }
        ann = upsert_annotation(ann, rec)
        atomic_write_parquet(ann, a_path)
        preview = gold_text if gold_text else "(empty/non_speech)"
        print(f"[saved] {sid} → {preview!r}")
        _finish_current()
        i += 1
        need_open = True

    n_ann = int((ann["status"] == "annotated").sum()) if len(ann) else 0
    n_empty = int((ann["status"] == "empty").sum()) if len(ann) else 0
    print(f"[OK] annotations → {a_path}  (annotated={n_ann}, empty={n_empty}, total_rows={len(ann)})")
    print(
        f'下一步 merge:\n  python scripts/annotate_semantic_risk_transcripts.py merge --batch "{batch}"'
    )
    return 0


# ──────────────────────────────────────────────
# merge
# ──────────────────────────────────────────────

def _apply_annotation_to_sample(sample: Sample, row: dict[str, Any]) -> bool:
    """回填 labels；成功返回 True。不改 category。"""
    status = str(row.get("status") or "")
    if status not in {"annotated", "empty"}:
        return False
    ohash_ann = str(row.get("original_audio_sha256") or "").strip()
    ohash_sample = original_audio_sha256(sample)
    if ohash_ann and ohash_sample and ohash_ann != ohash_sample:
        raise ValueError(
            f"sha256 不匹配 sample_id={sample.id}: ann={ohash_ann} sample={ohash_sample}"
        )

    gold_kind = str(row.get("gold_kind") or "").strip() or GOLD_KIND_SPEECH
    gold_text = row.get("gold_text")
    if gold_text is None or (isinstance(gold_text, float) and pd.isna(gold_text)):
        gold_text = ""
    else:
        gold_text = str(gold_text)

    sample.labels["gold_text"] = gold_text
    sample.labels["gold_kind"] = gold_kind
    sample.labels["annotation_state"] = STATE_ANNOTATED
    sample.labels["annotator_id"] = str(row.get("annotator_id") or "")
    sample.labels["annotated_at"] = str(row.get("annotated_at") or _now_iso())
    sample.labels["label_source"] = "human_transcript_cli"
    sample.labels["transcript_annotation_protocol"] = "semantic_risk_cli_v1"
    # 听写完成即标记人工确认转写；正式双审金标仍走 annotation_v3 review import
    sample.labels["is_human_verified"] = True
    if gold_kind == GOLD_KIND_SPEECH and gold_text:
        sample.labels["label"] = gold_text
    sample.add_lineage(
        "annotate_semantic_risk_transcripts",
        "1.0.0",
        {
            "status": status,
            "annotator_id": sample.labels["annotator_id"],
            "decision": str(row.get("decision") or ""),
        },
    )
    return True


def cmd_merge(args: argparse.Namespace) -> int:
    batch = str(args.batch).strip()
    classified = resolve_path(args.classified) if args.classified else default_classified_path(
        batch, rule_stem=args.rule_stem
    )
    work_dir = resolve_path(args.work_dir) if args.work_dir else default_work_dir(batch)
    a_path = resolve_path(args.annotations) if args.annotations else annotations_path(work_dir)
    out = resolve_path(args.output) if args.output else default_merged_path(
        batch, rule_stem=args.rule_stem
    )

    if not classified.is_file():
        raise FileNotFoundError(f"分类 parquet 不存在: {classified}")
    if not a_path.is_file():
        raise FileNotFoundError(f"annotations 不存在: {a_path}")
    if out.exists() and not args.overwrite:
        raise FileExistsError(f"输出已存在（加 --overwrite）: {out}")

    ann = load_annotations(a_path)
    usable = ann[ann["status"].astype(str).isin(["annotated", "empty"])].copy()
    if usable.empty:
        print("[WARN] 没有可合并的 annotated/empty 行")
        return 1

    by_id: dict[str, dict[str, Any]] = {}
    for _, row in usable.iterrows():
        sid = str(row["sample_id"])
        by_id[sid] = row.to_dict()

    print(f"[INFO] 读取分类: {classified}")
    manifest = Manifest.load(classified)
    applied = 0
    missing = 0
    hash_fail = 0
    for sample in manifest.samples:
        row = by_id.get(sample.id)
        if row is None:
            continue
        try:
            if _apply_annotation_to_sample(sample, row):
                applied += 1
        except ValueError as exc:
            hash_fail += 1
            print(f"[ERR] {exc}")
            if args.strict:
                raise

    for sid in by_id:
        if not any(s.id == sid for s in manifest.samples):
            missing += 1
            print(f"[WARN] 标注有、分类无: {sid}")

    report = {
        "batch": batch,
        "classified": str(classified),
        "annotations": str(a_path),
        "output": str(out),
        "annotation_usable": int(len(usable)),
        "applied": applied,
        "missing_in_classified": missing,
        "sha256_mismatches": hash_fail,
        "classified_total": len(manifest.samples),
        "merged_at": _now_iso(),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    # 原子写 Manifest
    with tempfile.TemporaryDirectory(dir=str(out.parent)) as tmp:
        tmp_out = Path(tmp) / out.name
        manifest.save(tmp_out)
        os.replace(tmp_out, out)
    report_path = out.with_suffix(out.suffix + ".merge_report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] 合并完成 applied={applied}/{len(usable)} → {out}")
    print(f"[OK] 报告 → {report_path}")
    if missing or hash_fail:
        print(f"[WARN] missing_in_classified={missing} sha256_mismatches={hash_fail}")
        return 2 if args.strict else 0
    return 0


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="semantic_risk 听写标注：prepare → label → merge",
    )
    sub = p.add_subparsers(dest="command", required=True)

    p_prep = sub.add_parser("prepare", help="筛 semantic_risk 并生成队列/播放清单")
    p_prep.add_argument("--batch", required=True)
    p_prep.add_argument("--classified", default=None, help="五分类 parquet；默认 derived 下 v2.3 stem")
    p_prep.add_argument("--rule-stem", default=DEFAULT_RULE_STEM)
    p_prep.add_argument("--work-dir", default=None)
    p_prep.add_argument(
        "--link-wav",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="在 work-dir/wav 建序号软链（默认开；--no-link-wav 关闭）",
    )
    p_prep.add_argument("--overwrite-links", action="store_true")
    p_prep.set_defaults(func=cmd_prepare)

    p_label = sub.add_parser("label", help="交互听写写入 annotations.parquet")
    p_label.add_argument("--batch", required=True)
    p_label.add_argument("--work-dir", default=None)
    p_label.add_argument("--annotator-id", required=True)
    p_label.add_argument("--start-index", type=int, default=None, help="从 queue_index 起（含）")
    p_label.set_defaults(func=cmd_label)

    p_merge = sub.add_parser("merge", help="标注回填到五分类 parquet")
    p_merge.add_argument("--batch", required=True)
    p_merge.add_argument("--classified", default=None)
    p_merge.add_argument("--annotations", default=None)
    p_merge.add_argument("--work-dir", default=None)
    p_merge.add_argument("--output", default=None)
    p_merge.add_argument("--rule-stem", default=DEFAULT_RULE_STEM)
    p_merge.add_argument("--overwrite", action="store_true")
    p_merge.add_argument("--strict", action="store_true", help="sha256 不匹配或标注孤儿则失败")
    p_merge.set_defaults(func=cmd_merge)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"[ERR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
