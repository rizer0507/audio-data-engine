#!/usr/bin/env python3
"""以外源标注文件 A 为 baseline，对多模型识别结果计算字准率（evaluation CER）。

对齐规则：
  - 模型侧 id 通常为 wav 路径文件名（去后缀），也可用 ``source_path`` 的 stem
  - A 中的 id 至少包含该文件名；优先精确匹配，其次「A.id 包含 model_id」

字准率计算与流水线一致：
  ``normalize_text(zh_asr_v1)`` + ``calculate_cer(reference=baseline, hypothesis=model)``
  输出字段语义与 ``configs/metrics/model_eval.yaml`` 相同（evaluation，非 agreement）。

Example:
  python scripts/eval_external_baseline_cer.py \\
    --baseline 数据集/标注A.xlsx \\
    --baseline-text-col label_text_raw \\
    --model qwen=datasets/manifests/qwen_asr_mt3000.parquet \\
    --model sensevoice=datasets/manifests/sensevoice_asr_mt3000.parquet \\
    --output datasets/exports/external_baseline_cer_mt3000.parquet

  # 或对已聚合的 multi_asr_aggregate 一次算多个模型：
  python scripts/eval_external_baseline_cer.py \\
    --baseline 数据集/标注A.xlsx \\
    --manifest datasets/manifests/multi_asr_aggregate_mt3000.parquet \\
    --models qwen,sensevoice \\
    --output datasets/exports/external_baseline_cer_mt3000.xlsx
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audio_engine.core.manifest import Manifest  # noqa: E402
from audio_engine.metrics.normalization import normalize_text  # noqa: E402
from audio_engine.metrics.runner import run_text_metrics  # noqa: E402

_ID_CANDIDATES = ("id", "sample_id", "audio_id", "utt_id", "文件名", "音频", "音频文件")
_TEXT_CANDIDATES = (
    "baseline_text",
    "gold_text",
    "label_text_raw",
    "label_text",
    "text",
    "标注",
    "标注文本",
    "转写",
)


def _as_text(value: Any) -> str:
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


def _pick_column(columns: list[str], requested: str | None, candidates: tuple[str, ...], label: str) -> str:
    if requested:
        if requested not in columns:
            raise ValueError(f"{label}列不存在: {requested!r}; 可用列: {columns}")
        return requested
    for name in candidates:
        if name in columns:
            return name
    raise ValueError(f"无法自动识别{label}列；请用参数指定。可用列: {columns}")


def load_table(path: Path) -> pd.DataFrame:
    """Load baseline or flat ASR table from common formats."""
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return pd.DataFrame(rows)
    raise ValueError(f"不支持的文件格式: {path.suffix} ({path})")


def load_baseline(
    path: Path,
    *,
    id_col: str | None,
    text_col: str | None,
) -> pd.DataFrame:
    df = load_table(path)
    columns = [str(c) for c in df.columns]
    id_key = _pick_column(columns, id_col, _ID_CANDIDATES, "baseline id")
    text_key = _pick_column(columns, text_col, _TEXT_CANDIDATES, "baseline text")
    out = pd.DataFrame(
        {
            "baseline_id": df[id_key].map(lambda v: _as_text(v)),
            "baseline_text": df[text_key].map(_as_text),
        }
    )
    out = out[out["baseline_id"] != ""].drop_duplicates(subset=["baseline_id"], keep="first")
    return out.reset_index(drop=True)


def audio_stem(path_or_id: str) -> str:
    """Wav 路径取无后缀文件名；纯 id 原样返回。"""
    text = _as_text(path_or_id)
    if not text:
        return ""
    name = Path(text.replace("\\", "/")).name
    if "." in name and not text.endswith(("/", "\\")):
        # Only strip suffix when it looks like a filename with extension.
        suffix = Path(name).suffix.lower()
        if suffix in {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".pcm", ".raw", ".json", ".txt"}:
            return Path(name).stem
    return name or text


def build_baseline_index(baseline: pd.DataFrame) -> dict[str, str]:
    """Map match_key → baseline_id for lookup.

    Keys include full baseline_id and audio stem extracted from it.
    """
    index: dict[str, str] = {}
    for _, row in baseline.iterrows():
        bid = str(row["baseline_id"])
        index[bid] = bid
        stem = audio_stem(bid)
        if stem and stem not in index:
            index[stem] = bid
    return index


def align_baseline_id(model_id: str, baseline_ids: list[str], exact_index: dict[str, str]) -> str | None:
    """Align model sample id to a baseline row id.

    1) exact match on model_id / stem
    2) baseline_id contains model_id (or its stem)
    """
    mid = _as_text(model_id)
    if not mid:
        return None
    stem = audio_stem(mid)
    for key in (mid, stem):
        if key and key in exact_index:
            return exact_index[key]

    needle = stem or mid
    hits = [bid for bid in baseline_ids if needle and needle in bid]
    if not hits:
        return None
    if len(hits) == 1:
        return hits[0]
    # Prefer shortest containing id (most specific), then lexicographic.
    hits.sort(key=lambda x: (len(x), x))
    return hits[0]


def _transcript_from_flat_row(row: dict[str, Any], model: str) -> str:
    for key in (f"{model}_text", model, "text"):
        if key in row and _as_text(row.get(key)):
            return _as_text(row.get(key))
    transcripts = row.get("transcripts")
    if isinstance(transcripts, str):
        try:
            transcripts = json.loads(transcripts)
        except json.JSONDecodeError:
            transcripts = None
    if isinstance(transcripts, dict):
        entry = transcripts.get(model, {})
        if isinstance(entry, dict):
            return _as_text(entry.get("text"))
        return _as_text(entry)
    return ""


def load_model_predictions(path: Path, model: str) -> pd.DataFrame:
    """Load one model's predictions as columns: model_id, source_path, hyp_text."""
    suffix = path.suffix.lower()
    rows: list[dict[str, Any]] = []

    if suffix in {".parquet", ".jsonl"}:
        try:
            manifest = Manifest.load(path)
            for sample in manifest.samples:
                hyp = sample.get_transcript_text(model)
                if not hyp:
                    # Flat parquet may only have {model}_text without nested transcripts.
                    flat = sample.to_flat_dict()
                    hyp = _transcript_from_flat_row(flat, model)
                mid = sample.id or audio_stem(sample.source_path)
                rows.append(
                    {
                        "model_id": mid,
                        "model_id_stem": audio_stem(mid) or audio_stem(sample.source_path),
                        "source_path": sample.source_path,
                        "hyp_text": hyp,
                    }
                )
            if rows:
                return pd.DataFrame(rows)
        except Exception:
            # Fall through to flat table load.
            pass

    df = load_table(path)
    records = df.to_dict(orient="records")
    id_col = None
    for candidate in _ID_CANDIDATES:
        if candidate in df.columns:
            id_col = candidate
            break
    path_col = "source_path" if "source_path" in df.columns else None

    for record in records:
        mid = _as_text(record.get(id_col)) if id_col else ""
        source = _as_text(record.get(path_col)) if path_col else ""
        if not mid:
            mid = audio_stem(source)
        rows.append(
            {
                "model_id": mid,
                "model_id_stem": audio_stem(mid) or audio_stem(source),
                "source_path": source,
                "hyp_text": _transcript_from_flat_row(record, model),
            }
        )
    out = pd.DataFrame(rows)
    out = out[out["model_id"] != ""].drop_duplicates(subset=["model_id"], keep="first")
    return out.reset_index(drop=True)


def load_normalization(path: Path | None) -> dict[str, Any]:
    profile = path or (ROOT / "configs" / "normalization" / "zh_asr_v1.yaml")
    if not profile.is_file():
        return {
            "name": "zh_asr_v1_fallback",
            "unicode": {"normalize": True, "form": "NFKC"},
            "punctuation": {"remove": True},
            "whitespace": {"remove": True},
            "english": {"lowercase": True},
            "filler": {"remove": False},
        }
    return yaml.safe_load(profile.read_text(encoding="utf-8")) or {}


def score_pair(baseline_text: str, hyp_text: str, normalization: dict[str, Any], prefix: str) -> dict[str, Any]:
    """Compute evaluation CER via MetricRunner semantics."""
    record = {"baseline_text": baseline_text, "hyp_text": hyp_text}
    comparison = {
        "purpose": "model_evaluation",
        "reference": {"field": "baseline_text"},
        "hypothesis": {"field": "hyp_text"},
        "metrics": ["cer"],
        "output": {"prefix": prefix},
        "overwrite": True,
    }
    metrics = run_text_metrics(record, comparison, normalization)
    cer = metrics[f"{prefix}_cer"]
    ref_len = metrics[f"{prefix}_reference_length"]
    acc = round(max(0.0, 1.0 - float(cer)), 6)
    return {
        **metrics,
        f"{prefix}_字准率": acc,
        f"{prefix}_ref_norm": normalize_text(baseline_text, normalization),
        f"{prefix}_hyp_norm": normalize_text(hyp_text, normalization),
        "_cer": float(cer),
        "_ref_len": int(ref_len),
        "_acc": acc,
    }


def parse_model_arg(raw: str) -> tuple[str, Path]:
    text = (raw or "").strip()
    if "=" not in text:
        raise ValueError(f"--model 需要 `name=path`，收到: {raw!r}")
    name, _, path = text.partition("=")
    name, path = name.strip(), path.strip()
    if not name or not path:
        raise ValueError(f"--model 需要 `name=path`，收到: {raw!r}")
    return name, Path(path)


def save_output(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        df.to_excel(path, index=False)
    elif suffix == ".csv":
        df.to_csv(path, index=False)
    elif suffix == ".parquet":
        df.to_parquet(path, index=False)
    elif suffix == ".jsonl":
        with path.open("w", encoding="utf-8") as handle:
            for record in df.to_dict(orient="records"):
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    else:
        raise ValueError(f"不支持的输出格式: {path.suffix}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="外源 baseline 标注文件 vs 多模型识别结果 → evaluation CER / 字准率"
    )
    parser.add_argument("--baseline", type=Path, required=True, help="外源标注文件 A")
    parser.add_argument("--baseline-id-col", default=None, help="A 中 id 列名（可自动识别）")
    parser.add_argument("--baseline-text-col", default=None, help="A 中 baseline 文本列名")
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="模型结果 name=path（可重复），path 为 parquet/jsonl/xlsx/csv",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="已聚合的 multi_asr_aggregate parquet/jsonl（配合 --models）",
    )
    parser.add_argument(
        "--models",
        default="",
        help="从 --manifest 读取的模型名，逗号分隔，如 qwen,sensevoice",
    )
    parser.add_argument(
        "--normalization",
        type=Path,
        default=ROOT / "configs" / "normalization" / "zh_asr_v1.yaml",
        help="归一化 profile（默认 zh_asr_v1）",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="输出 parquet/xlsx/csv/jsonl",
    )
    parser.add_argument(
        "--how",
        choices=("inner", "left"),
        default="inner",
        help="对齐方式：inner=只保留匹配到 baseline 的样本；left=保留全部模型样本",
    )
    args = parser.parse_args()

    if not args.baseline.exists():
        print(f"[ERROR] baseline 不存在: {args.baseline}", file=sys.stderr)
        return 1

    model_specs: list[tuple[str, Path]] = [parse_model_arg(item) for item in args.model]
    if args.manifest is not None:
        names = [n.strip() for n in str(args.models).split(",") if n.strip()]
        if not names:
            raise SystemExit("--manifest 需要同时提供 --models qwen,sensevoice")
        if not args.manifest.exists():
            print(f"[ERROR] manifest 不存在: {args.manifest}", file=sys.stderr)
            return 1
        for name in names:
            model_specs.append((name, args.manifest))
    if not model_specs:
        print("[ERROR] 请至少提供 --model name=path 或 --manifest + --models", file=sys.stderr)
        return 1

    for name, path in model_specs:
        if not path.exists():
            print(f"[ERROR] 模型文件不存在: {name}={path}", file=sys.stderr)
            return 1

    normalization = load_normalization(args.normalization)
    baseline = load_baseline(
        args.baseline,
        id_col=args.baseline_id_col,
        text_col=args.baseline_text_col,
    )
    baseline_ids = baseline["baseline_id"].tolist()
    exact_index = build_baseline_index(baseline)
    baseline_text_map = dict(zip(baseline["baseline_id"], baseline["baseline_text"], strict=True))

    print(f"[INFO] baseline: {args.baseline} rows={len(baseline):,}")
    print(f"[INFO] normalizer: {normalization.get('name', args.normalization)}")

    # Union all model ids as the row spine (prefer first model's order).
    predictions: dict[str, pd.DataFrame] = {}
    for name, path in model_specs:
        pred = load_model_predictions(path, name)
        predictions[name] = pred
        print(f"[INFO] model={name} path={path} rows={len(pred):,}")

    # Build aligned rows keyed by model_id from the first model, then union others.
    spine_ids: list[str] = []
    seen: set[str] = set()
    for name, _ in model_specs:
        for mid in predictions[name]["model_id"].tolist():
            if mid not in seen:
                seen.add(mid)
                spine_ids.append(mid)

    rows: list[dict[str, Any]] = []
    matched = unmatched = 0
    corpus: dict[str, dict[str, float | int]] = {
        name: {"edits": 0, "ref_len": 0, "n": 0, "n_skip": 0, "sum_acc": 0.0}
        for name, _ in model_specs
    }

    for mid in spine_ids:
        stem = audio_stem(mid)
        bid = align_baseline_id(mid, baseline_ids, exact_index)
        if bid is None:
            unmatched += 1
            if args.how == "inner":
                continue
            baseline_text = ""
            matched_flag = False
        else:
            matched += 1
            baseline_text = baseline_text_map.get(bid, "")
            matched_flag = True

        row: dict[str, Any] = {
            "id": mid,
            "id_stem": stem,
            "baseline_id": bid or "",
            "baseline_text": baseline_text,
            "baseline_matched": matched_flag,
        }

        for name, _ in model_specs:
            pred = predictions[name]
            hit = pred.loc[pred["model_id"] == mid]
            if hit.empty and stem:
                hit = pred.loc[pred["model_id_stem"] == stem]
            if hit.empty:
                hyp = ""
                source_path = ""
            else:
                hyp = _as_text(hit.iloc[0]["hyp_text"])
                source_path = _as_text(hit.iloc[0]["source_path"])
            row[f"{name}_text"] = hyp
            if source_path and "source_path" not in row:
                row["source_path"] = source_path

            prefix = f"{name}_vs_baseline"
            if not matched_flag or not normalize_text(baseline_text, normalization):
                row[f"{prefix}_cer"] = None
                row[f"{prefix}_字准率"] = None
                row[f"{prefix}_substitutions"] = None
                row[f"{prefix}_deletions"] = None
                row[f"{prefix}_insertions"] = None
                row[f"{prefix}_reference_length"] = None
                corpus[name]["n_skip"] += 1
                continue

            scored = score_pair(baseline_text, hyp, normalization, prefix)
            for key, value in scored.items():
                if key.startswith("_"):
                    continue
                row[key] = value
            corpus[name]["edits"] += int(
                scored[f"{prefix}_substitutions"]
                + scored[f"{prefix}_deletions"]
                + scored[f"{prefix}_insertions"]
            )
            corpus[name]["ref_len"] += int(scored["_ref_len"])
            corpus[name]["n"] += 1
            corpus[name]["sum_acc"] += float(scored["_acc"])

        rows.append(row)

    if not rows:
        print("[ERROR] 无对齐样本，终止", file=sys.stderr)
        return 1

    out = pd.DataFrame(rows)
    save_output(out, args.output)

    print("=" * 60)
    print(f"[INFO] matched={matched:,} unmatched_model_ids={unmatched:,} output_rows={len(out):,}")
    print(f"[INFO] wrote {args.output}")
    print("-" * 60)
    for name, _ in model_specs:
        stats = corpus[name]
        if stats["ref_len"] > 0:
            micro_cer = stats["edits"] / stats["ref_len"]
            micro_acc = max(0.0, 1.0 - micro_cer)
        else:
            micro_cer = micro_acc = float("nan")
        macro_acc = (stats["sum_acc"] / stats["n"]) if stats["n"] else float("nan")
        print(
            f"[SUMMARY] {name}_vs_baseline  "
            f"n={stats['n']:,} skip={stats['n_skip']:,}  "
            f"micro_cer={micro_cer:.6f} micro_字准率={micro_acc:.6f}  "
            f"macro_字准率={macro_acc:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
