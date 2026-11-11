#!/usr/bin/env python3
"""把 run_fenshen_1015_eval.sh 的转写 parquet 收成字准 xlsx。

每次评测创建一个时间戳目录，每个切分一份 XLSX::

  datasets/stage3/asr/eval_v030/1110_1430/v030_train_full.xlsx
  datasets/stage3/asr/eval_v030/1110_1430/v030_dev_full.xlsx

用法::

  python scripts/eval_fenshen_1015_xlsx.py full v015
  python scripts/eval_fenshen_1015_xlsx.py lora-full v030
  python scripts/eval_fenshen_1015_xlsx.py lora v015 --split dev
  python scripts/eval_fenshen_1015_xlsx.py rl v025 --split dev --ckpt best
  python scripts/eval_fenshen_1015_xlsx.py full v015 --split dev --ckpt best
  python scripts/eval_fenshen_1015_xlsx.py full v015 --split dev --ckpt over

qwen-base 是 baseline。只转写 best 或 over 时不重跑 base，字准表仍读取已有的 qwen-base parquet 做比较。
train 的基座文件名是 qwen-base-full，dev 是 qwen-base-dev。

金标非空时：先 plain，再删
``configs/normalization/fenshen_1015_filler_tokens.yaml`` 语气词，再收起 3 次及以上连续重复，然后算字准。
金标为空（噪声 / non_speech）：不去语气词；识别 plain 后非空则字准=0，空则字准=1；只累计编辑距离，不参与总字准率。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from itertools import count
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SCORE_DROP_YAML = ROOT / "configs" / "normalization" / "fenshen_1015_filler_tokens.yaml"

from audio_engine.core.manifest import Manifest  # noqa: E402
from audio_engine.core.transcript_reconcile import (  # noqa: E402
    levenshtein_ops,
    plain_transcript_text,
)


def load_score_drop_tokens(path: Path | None = None) -> tuple[str, ...]:
    """读取语气词 yaml。``tokens`` 为非空字符串列表，重复项忽略。长词由计分按字数优先删除。"""
    target = SCORE_DROP_YAML if path is None else path
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


_SCORE_DROP_TOKENS = load_score_drop_tokens()
_SCORE_DROP_SORTED = tuple(sorted(_SCORE_DROP_TOKENS, key=len, reverse=True))

SCOPES = ("overall", "auto_gold", "hardcase", "non_speech")
DETAIL_LEAD = (
    "id",
    "type",
    "label",
    "candidate_text",
    "gold_text_ann",
    "gold_vs_candidate_mismatch",
    "category",
    "decision",
    "gold_kind",
    "noise_kind",
    "exclude_from_training",
    "uncertain",
    "label_source",
    "wav_path",
)


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


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    text = _cell(value)
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def classify_type(gold_kind: str, decision: str, candidate: str, gold_text: str) -> str:
    """non_speech 优先；speech 按预标注和金标 plain 后是否一致分 auto_gold / hardcase。"""
    if _cell(gold_kind) == "non_speech" or _cell(decision) == "empty":
        return "non_speech"
    if plain_transcript_text(candidate) == plain_transcript_text(gold_text):
        return "auto_gold"
    return "hardcase"


def _is_false_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return not value
    return _cell(value).lower() in {"0", "false", "no", "n"}


def skip_type_split(rows: list[dict[str, Any]], names: set[str]) -> bool:
    """金标每行 label_eval_type_split 都为 false 时，字准表只出 overall。"""
    if "label_eval_type_split" not in names or not rows:
        return False
    return all(_is_false_flag(row.get("label_eval_type_split")) for row in rows)


def drop_score_tokens(text: str) -> str:
    """plain 之后删掉语气词表。金标和识别用同一份词表。"""
    dropped = plain_transcript_text(text)
    for token in _SCORE_DROP_SORTED:
        dropped = dropped.replace(token, "")
    return dropped


def collapse_abnormal_loops(text: str, *, max_unit: int = 16) -> str:
    """连续重复 3 次及以上的片段收成 1 次。连续 2 次保留。

    优先按更短的片段折叠，所以「不用不用不用不用」收成「不用」，不是「不用不用」。
    """
    if len(text) < 3:
        return text
    pieces: list[str] = []
    index = 0
    size = len(text)
    changed = False
    while index < size:
        folded = False
        upper = min(max_unit, (size - index) // 3)
        for unit_len in range(1, upper + 1):
            unit = text[index : index + unit_len]
            repeats = 1
            cursor = index + unit_len
            while cursor + unit_len <= size and text[cursor : cursor + unit_len] == unit:
                repeats += 1
                cursor += unit_len
            if repeats >= 3:
                pieces.append(unit)
                index = cursor
                folded = True
                changed = True
                break
        if not folded:
            pieces.append(text[index])
            index += 1
    if not changed:
        return text
    return collapse_abnormal_loops("".join(pieces), max_unit=max_unit)


def comparable_text(text: str) -> str:
    """计分文本：plain、删语气词，再把 3 次及以上的连续重复收成 1 次。"""
    return collapse_abnormal_loops(drop_score_tokens(text))


def score_pair(reference: str, hypothesis: str) -> tuple[float, int, int, int]:
    """返回字准率、exact_correct、编辑距离、基准字数。exact 只在编辑距离为 0 时为 1。

    金标为空：只对识别做 plain，不去语气词、不折叠重复。plain 后非空则字准=0；
    编辑距离为插入字数，基准字数为 0（不参与总字准率）。

    金标非空：先 plain，再删语气词，再把连续 3 次及以上的重复收成 1 次后算编辑距离。
    """
    if not _cell(reference):
        hyp_plain = plain_transcript_text(hypothesis)
        edits = len(hyp_plain)
        exact = int(edits == 0)
        return (1.0 if exact else 0.0), exact, edits, 0
    ops = levenshtein_ops(comparable_text(reference), comparable_text(hypothesis))
    edits = int(ops["total"] or 0)
    ref_len = int(ops["ref_len"] or 0)
    exact = int(edits == 0)
    if exact:
        accuracy = 1.0
    elif ref_len <= 0:
        accuracy = 0.0
    else:
        accuracy = round(max(0.0, 1.0 - edits / ref_len), 6)
    return accuracy, exact, edits, ref_len


def _pct_number(correct: int, total: int) -> float | None:
    if total <= 0:
        return None
    return round(correct / total * 100, 2)


def _char_number(edits: int, ref_chars: int) -> float | None:
    if ref_chars <= 0:
        return None
    return round((1.0 - edits / ref_chars) * 100, 2)


def _pct_text(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.2f}%"


def _signed_pct(value: float | None, base: float | None) -> str:
    if value is None or base is None:
        return ""
    return f"{round(value - base, 2):+.2f}%"


def _signed_int(value: int, base: int) -> str:
    return f"{int(value) - int(base):+d}"


def _wav_path(audio: Any, source_path: Any) -> str:
    payload = _as_dict(audio)
    return _cell(payload.get("resampled_16k") or source_path)


def _category(row: dict[str, Any]) -> str:
    for key in ("label_human_category", "label_source_category", "label_classification_bucket"):
        text = _cell(row.get(key))
        if text:
            return text
    return ""


def load_gold_rows(path: Path) -> list[dict[str, Any]]:
    table = pq.read_table(path)
    names = set(table.schema.names)
    rows = table.to_pylist()
    unclassified = skip_type_split(rows, names)
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in rows:
        sample_id = _cell(raw.get("id"))
        if not sample_id:
            raise SystemExit(f"[ERROR] 金标存在空 id: {path}")
        if sample_id in seen:
            raise SystemExit(f"[ERROR] 金标 id 重复: {sample_id}")
        seen.add(sample_id)
        gold_text = _cell(raw.get("label_gold_text"))
        candidate = _cell(raw.get("qwen_1_text"))
        kind = _cell(raw.get("label_source_gold_kind"))
        decision = _cell(raw.get("label_decision"))
        explicit = _cell(raw.get("label_eval_type"))
        empty_ref = explicit == "non_speech" or kind == "non_speech" or decision == "empty"
        if unclassified:
            row_type = ""
        elif explicit in {"non_speech", "auto_gold", "hardcase"}:
            row_type = explicit
        else:
            row_type = classify_type(kind, decision, candidate, gold_text)
        reference = "" if empty_ref else gold_text
        shown = "" if empty_ref else gold_text
        out.append(
            {
                "id": sample_id,
                "type": row_type,
                "label": shown,
                "candidate_text": candidate,
                "gold_text_ann": shown,
                "gold_vs_candidate_mismatch": int(
                    plain_transcript_text(candidate) != plain_transcript_text(gold_text)
                ),
                "category": _category(raw),
                "decision": decision,
                "gold_kind": kind,
                "noise_kind": _cell(raw.get("label_noise_kind")),
                "exclude_from_training": raw.get("label_exclude_from_training"),
                "uncertain": _cell(raw.get("label_uncertain")) if "label_uncertain" in names else "",
                "label_source": _cell(raw.get("label_label_source")),
                "wav_path": _wav_path(raw.get("audio"), raw.get("source_path")),
                "reference": reference,
            }
        )
    if not out:
        raise SystemExit(f"[ERROR] 金标没有样本: {path}")
    return out


def load_asr_text(path: Path, alias: str, *, require_alias: bool = False) -> dict[str, str]:
    texts: dict[str, str] = {}
    for sample in Manifest.load(path):
        if sample.id in texts:
            raise SystemExit(f"[ERROR] 转写 id 重复: {path} {sample.id}")
        if alias in sample.transcripts:
            texts[sample.id] = _cell(sample.get_transcript_text(alias))
        elif require_alias:
            raise SystemExit(f"[ERROR] {path.name} 缺少转写 {alias}: {sample.id}")
        elif len(sample.transcripts) == 1:
            only = next(iter(sample.transcripts))
            texts[sample.id] = _cell(sample.get_transcript_text(only))
        else:
            texts[sample.id] = ""
    if not texts:
        raise SystemExit(f"[ERROR] 转写 parquet 没有样本: {path}")
    return texts


def _model_stats(rows: list[dict[str, Any]], texts: dict[str, str]) -> dict[str, Any]:
    scored = [row for row in rows if row["id"] in texts]
    correct = 0
    char_edits = 0
    empty_edits = 0
    ref_chars = 0
    per_id: dict[str, tuple[str, float, int]] = {}
    for row in scored:
        hypothesis = texts[row["id"]]
        accuracy, exact, row_edits, row_ref = score_pair(row["reference"], hypothesis)
        correct += exact
        if _cell(row["reference"]):
            # 空金标不进总字准：编辑距离与基准字数只累计非空金标。
            char_edits += row_edits
            ref_chars += row_ref
        else:
            empty_edits += row_edits
        per_id[row["id"]] = (hypothesis, accuracy, exact)
    return {
        "n": len(scored),
        "correct": correct,
        # 有非空金标时 edits 只含字准编辑距离；纯空金标 scope 改报 empty_edits。
        "edits": empty_edits if ref_chars <= 0 else char_edits,
        "char_edits": char_edits,
        "empty_edits": empty_edits,
        "ref_chars": ref_chars,
        "per_id": per_id,
    }


def build_frames(
    gold_rows: list[dict[str, Any]],
    models: list[tuple[str, dict[str, str]]],
    *,
    gold_path: Path,
    model_paths: dict[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not models:
        raise SystemExit("[ERROR] 没有可比较的模型")
    baseline_name = models[0][0]
    rel_rate_col = "相对 qwen-base" if baseline_name == "qwen-base" else f"相对 {baseline_name}"
    rel_char_col = "相对 qwen-base总字准" if baseline_name == "qwen-base" else f"相对 {baseline_name}总字准"
    names = [name for name, _ in models]
    by_scope = {
        scope: [row for row in gold_rows if scope == "overall" or row["type"] == scope]
        for scope in SCOPES
    }
    stats = {
        scope: {name: _model_stats(rows, texts) for name, texts in models}
        for scope, rows in by_scope.items()
        if scope == "overall" or rows
    }

    accuracy_rows: list[dict[str, Any]] = []
    pivot_rows: list[dict[str, Any]] = []
    for scope, model_stats in stats.items():
        base = model_stats[baseline_name]
        base_rate = _pct_number(base["correct"], base["n"])
        base_char = _char_number(base["edits"], base["ref_chars"])
        base_integer = base["ref_chars"] <= 0
        scope_ids = {row["id"] for row in by_scope[scope]}
        recognized = 0
        for row in by_scope[scope]:
            if all(row["id"] in texts for _, texts in models):
                recognized += 1
        pivot: dict[str, Any] = {
            "scope": scope,
            "标注总条数": len(scope_ids),
            "样本数(有识别)": recognized,
        }
        for name, _texts in models:
            item = model_stats[name]
            rate = _pct_number(item["correct"], item["n"])
            integer = item["ref_chars"] <= 0
            char_value: Any = item["edits"] if integer else _char_number(item["edits"], item["ref_chars"])
            if name == baseline_name:
                rel_rate = "baseline"
                rel_char: Any = "baseline"
            elif integer and base_integer:
                rel_rate = _signed_pct(rate, base_rate)
                rel_char = _signed_int(item["edits"], base["edits"])
            else:
                rel_rate = _signed_pct(rate, base_rate)
                rel_char = _signed_pct(_char_number(item["edits"], item["ref_chars"]), base_char)
            accuracy_rows.append(
                {
                    "scope": scope,
                    "模型": name,
                    "样本数": item["n"],
                    "正确数(字准=1)": item["correct"],
                    "正确率": _pct_text(rate),
                    rel_rate_col: rel_rate,
                    "总编辑距离": item["edits"],
                    "总基准字数": item["ref_chars"],
                    "总字准": char_value if integer else _pct_text(_char_number(item["edits"], item["ref_chars"])),
                    rel_char_col: rel_char,
                }
            )
            pivot[f"{name}_正确率%"] = rate
            pivot[f"{name}_正确数"] = item["correct"]
            pivot[f"{name}_样本数"] = item["n"]
            pivot[f"{name}_总字准%"] = char_value
            pivot[f"{name}_总编辑距离"] = item["edits"]
            pivot[f"{name}_总基准字数"] = item["ref_chars"]
        pivot_rows.append(pivot)

    detail_rows: list[dict[str, Any]] = []
    for row in gold_rows:
        present = [name for name, texts in models if row["id"] in texts]
        item: dict[str, Any] = {key: row[key] for key in DETAIL_LEAD}
        for name, texts in models:
            scored = stats["overall"][name]["per_id"].get(row["id"])
            if scored is None:
                item[f"{name}_text"] = None
                item[f"{name}_字准率"] = None
                item[f"{name}_exact_correct"] = None
            else:
                hypothesis, accuracy, exact = scored
                item[f"{name}_text"] = hypothesis
                item[f"{name}_字准率"] = accuracy
                item[f"{name}_exact_correct"] = exact
        covered = len(present) == len(models)
        item["in_model_eval"] = int(covered)
        item["eval_scored"] = bool(covered)
        detail_rows.append(item)

    type_counts = {scope: sum(row["type"] == scope for row in gold_rows) for scope in SCOPES[1:]}
    unclassified = not any(_cell(row["type"]) for row in gold_rows)
    if unclassified:
        type_notes = [
            (
                "分类",
                "本金标 label_eval_type_split=false，不划分 auto_gold / hardcase / non_speech，只统计 overall。"
                "label_source_gold_kind=non_speech 或 label_decision=empty 的计分参考仍是空串，不参与总字准率。",
            )
        ]
        count_note = f"total={len(gold_rows)}；不做分类"
    else:
        type_notes = [
            (
                "type non_speech",
                "label_source_gold_kind=non_speech 或 label_decision=empty。"
                "计分金标当空串；只计正确率与编辑距离，不参与总字准率。",
            ),
            ("type auto_gold", "speech，且 plain(qwen_1_text)==plain(label_gold_text)。"),
            ("type hardcase", "speech，且 plain 后预标注与金标不一致。"),
        ]
        count_note = "；".join(f"{scope}={type_counts[scope]}" for scope in type_counts) + f"；total={len(gold_rows)}"
    notes = [
        ("金标", str(gold_path)),
        ("模型", "；".join(f"{name}={model_paths[name]}" for name in names)),
        ("文本列", "各转写 parquet 的 transcripts.<alias>.text，扁平列名为 <alias>_text"),
        ("baseline", f"{baseline_name}。相对列只相对它。"),
        *type_notes,
        (
            "正确率",
            "字准率==1 的条数 / 该 scope 条数。"
            "金标非空：标点差异在 plain 之后不算错；"
            f"计分前从金标和识别中删除语气词表 {SCORE_DROP_YAML}："
            + "、".join(_SCORE_DROP_TOKENS)
            + "；"
            "语气词删完后，同一片段连续出现 3 次及以上收成 1 次；连续 2 次保留，例如「不用不用」不改，「不用不用不用」收成「不用」。"
            "金标为空：不去语气词；识别 plain 后只要还有文字，字准率=0、不算正确。",
        ),
        (
            "总字准",
            "仅金标非空样本：1 - 总编辑距离/总基准字数。"
            "编辑距离和基准字数都按删掉语气词、并收起 3 次及以上连续重复之后的文本计算。"
            "空金标不计入总编辑距离与总基准字数，因此不参与总字准率。",
        ),
        (
            "non_speech / 空金标",
            "只统计正确率与总编辑距离（识别 plain 后的插入字数）；"
            "总基准字数为 0，总字准改写总编辑距离整数；相对 qwen-base总字准改写编辑距离差。"
            "不去语气词，避免「什么」等词被抹成空后误算正确。",
        ),
        ("条数", count_note),
        ("字准实现", "audio_engine.core.transcript_reconcile.levenshtein_ops / plain_transcript_text"),
    ]
    return (
        pd.DataFrame(accuracy_rows),
        pd.DataFrame(pivot_rows),
        pd.DataFrame(detail_rows),
        pd.DataFrame(notes, columns=["字段/口径", "说明"]),
    )


def write_xlsx(
    path: Path, frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
) -> None:
    accuracy, pivot, detail, notes = frames
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("xb")
    try:
        with stream, pd.ExcelWriter(stream, engine="openpyxl") as writer:
            accuracy.to_excel(writer, index=False, sheet_name="正确率")
            pivot.to_excel(writer, index=False, sheet_name="透视对比")
            detail.to_excel(writer, index=False, sheet_name="明细")
            notes.to_excel(writer, index=False, sheet_name="说明")
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def gold_path(root: Path, split: str, version: str) -> Path:
    return root / "datasets" / "stage1" / "derived" / f"gold_fenshen_1015_{split}_{version}.parquet"


def asr_jobs(
    mode: str, split: str, ckpt: str = "all", *, sophon_run_tag: str | None = None
) -> list[tuple[str, str]]:
    """返回 (展示名, asr-run)。qwen-base 始终在第一位，作为相对比较的 baseline。"""
    if ckpt not in ("all", "best", "over", "best-over"):
        raise SystemExit(f"[ERROR] 未知权重: {ckpt}")
    base_run = "qwen-base-full" if split == "train" else "qwen-base-dev"
    jobs: list[tuple[str, str]] = [("qwen-base", base_run)]
    models = ("lora", "full") if mode == "lora-full" else (mode,)
    for model in models:
        if ckpt in ("all", "best", "best-over"):
            suffix = f"-{sophon_run_tag}" if model == "sophon" and sophon_run_tag else ""
            jobs.append((f"qwen-{model}-best", f"qwen-{model}-{split}-best{suffix}"))
        if ckpt in ("all", "over", "best-over"):
            jobs.append((f"qwen-{model}-over", f"qwen-{model}-{split}-over"))
    return jobs


def asr_parquet(root: Path, version: str, split: str, asr_run: str) -> Path:
    eval_name = f"eval_fenshen_1015_{split}_{version}"
    return root / "datasets" / "stage3" / "asr" / eval_name / f"{asr_run}_asr_{eval_name}.parquet"


def create_report_dir(root: Path, version: str) -> Path:
    parent = root / "datasets" / "stage3" / "asr" / f"eval_{version}"
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%m%d_%H%M")
    for attempt in count(1):
        directory = parent / (stamp if attempt == 1 else f"{stamp}_{attempt}")
        try:
            directory.mkdir()
        except FileExistsError:
            continue
        return directory


def evaluate_split(
    root: Path,
    mode: str,
    version: str,
    split: str,
    ckpt: str = "all",
    *,
    report_dir: Path | None = None,
    sophon_run_tag: str | None = None,
) -> Path:
    gold = gold_path(root, split, version)
    if not gold.is_file():
        raise SystemExit(f"[ERROR] 缺少金标: {gold}")
    jobs = asr_jobs(mode, split, ckpt, sophon_run_tag=sophon_run_tag)
    paths = {name: asr_parquet(root, version, split, run) for name, run in jobs}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        joined = "\n".join(missing)
        raise SystemExit(f"[ERROR] 缺少转写 parquet:\n{joined}")
    gold_rows = load_gold_rows(gold)
    models: list[tuple[str, dict[str, str]]] = []
    for name, run in jobs:
        texts = load_asr_text(
            paths[name], run, require_alias=bool(sophon_run_tag) and name == "qwen-sophon-best"
        )
        absent = [row["id"] for row in gold_rows if row["id"] not in texts]
        if absent:
            raise SystemExit(
                f"[ERROR] {paths[name].name} 缺少 {len(absent)} 条金标 id，例如 {absent[0]}"
            )
        models.append((name, texts))
    frames = build_frames(gold_rows, models, gold_path=gold, model_paths=paths)
    if report_dir is None:
        report_dir = create_report_dir(root, version)
    destination = report_dir / f"{version}_{split}_{mode}.xlsx"
    write_xlsx(destination, frames)
    print(f"[OK] {split} {len(gold_rows)} -> {destination}")
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="fenshen_1015 转写 parquet 汇总成字准 xlsx")
    parser.add_argument("mode", choices=("lora", "full", "lora-full", "rl", "sophon"))
    parser.add_argument("version", help="例如 v015")
    parser.add_argument("--split", choices=("train", "dev", "both"))
    parser.add_argument("--ckpt", choices=("all", "best", "over", "best-over"))
    parser.add_argument("--sophon-run-tag", help="本次运行写入 sophon.run.txt 的 run_tag")
    parser.add_argument("--root", type=Path, default=ROOT, help="audio-data-engine 根目录")
    args = parser.parse_args(argv)
    if args.sophon_run_tag:
        from audio_engine.core.source_naming import validate_asr_run

        if args.mode != "sophon":
            parser.error("--sophon-run-tag 仅适用于 sophon")
        try:
            args.sophon_run_tag = validate_asr_run(args.sophon_run_tag)
        except ValueError as exc:
            parser.error(str(exc))
    if not args.version.startswith("v") or not args.version[1:].isdigit():
        raise SystemExit(f"[ERROR] 版本号须为 v 加数字，当前: {args.version}")
    default_split, default_ckpt = ("dev", "best") if args.mode == "lora-full" else ("both", "all")
    split = args.split or default_split
    ckpt = args.ckpt or default_ckpt
    if args.mode == "sophon":
        if ckpt in ("over", "best-over"):
            parser.error("sophon 只提供 best")
        split = args.split or "dev"
        ckpt = "best"
    splits = ("train", "dev") if split == "both" else (split,)
    root = args.root.resolve()
    report_dir = None
    for split in splits:
        destination = evaluate_split(
            root, args.mode, args.version, split, ckpt,
            report_dir=report_dir, sophon_run_tag=args.sophon_run_tag,
        )
        report_dir = destination.parent
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
