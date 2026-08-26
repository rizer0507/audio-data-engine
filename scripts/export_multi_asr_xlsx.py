#!/usr/bin/env python3
"""按 id 合并多个 ASR parquet/jsonl，导出 Excel。

同一 id 可来自多个文件（如 qwen_asr_*.parquet + sensevoice_asr_*.parquet），
合并为一行：id / source_path / qwen_text / sensevoice_text / …

Rules:
  - <= max-rows → one xlsx
  - > max-rows → stem-part-001.xlsx, part-002, ...

Example:
  # 已聚合的单文件（兼容旧用法）
  python scripts/export_multi_asr_xlsx.py \\
    --manifest datasets/manifests/multi_asr_aggregate_mt3000.parquet \\
    --output datasets/exports/multi_asr_aggregate_mt3000.xlsx

  # 多个独立识别结果按 id 对齐
  python scripts/export_multi_asr_xlsx.py \\
    --model qwen=datasets/manifests/qwen_asr_mt3000.parquet \\
    --model sensevoice=datasets/manifests/sensevoice_asr_mt3000.parquet \\
    --output datasets/exports/asr_mt3000.xlsx

  # 多个 manifest + 模型名列表（按文件顺序对应 models）
  python scripts/export_multi_asr_xlsx.py \\
    --manifest datasets/manifests/qwen_asr_mt3000.parquet \\
    --manifest datasets/manifests/sensevoice_asr_mt3000.parquet \\
    --models qwen,sensevoice \\
    -o datasets/exports/asr_mt3000.xlsx
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

DEFAULT_MAX_ROWS = 800_000


def _nonempty_text(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if value != value:  # NaN
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text or None


def _as_id(value: Any) -> str:
    text = _nonempty_text(value)
    return text or ""


def _transcript_text(transcripts: Any, model: str) -> str:
    if not isinstance(transcripts, dict):
        return ""
    entry = transcripts.get(model, {})
    if isinstance(entry, dict):
        return str(entry.get("text") or "")
    return str(entry or "")


def _parse_quality(record: dict[str, Any]) -> dict[str, Any]:
    quality = record.get("quality") or {}
    if isinstance(quality, str):
        try:
            quality = json.loads(quality)
        except json.JSONDecodeError:
            quality = {}
    if not isinstance(quality, dict):
        quality = {}
    return quality


def _parse_transcripts(record: dict[str, Any]) -> dict[str, Any]:
    transcripts = record.get("transcripts") or {}
    if isinstance(transcripts, str):
        try:
            transcripts = json.loads(transcripts)
        except json.JSONDecodeError:
            transcripts = {}
    if not isinstance(transcripts, dict):
        transcripts = {}
    return dict(transcripts)


def _metric(record: dict[str, Any], quality: dict[str, Any], key: str) -> Any:
    for candidate in (f"quality_{key}", key):
        if candidate not in record:
            continue
        value = record[candidate]
        if value is None:
            continue
        try:
            if value != value:  # NaN
                continue
        except (TypeError, ValueError):
            pass
        return value
    return quality.get(key)


def extract_model_text(record: dict[str, Any], model: str) -> str:
    """Prefer nested transcripts[model].text, then flat {model}_text."""
    transcripts = _parse_transcripts(record)
    nested = _nonempty_text(_transcript_text(transcripts, model))
    flat = _nonempty_text(record.get(f"{model}_text"))
    return nested or flat or ""


def ensure_model_transcript(record: dict[str, Any], model: str) -> dict[str, Any]:
    """Bind this file's transcript text to ``model`` (even if source key differs).

    Useful when two Qwen runs both store ``transcripts.qwen`` / ``qwen_text`` but
    should export as ``qwen`` vs ``qwen_sft``.
    """
    out = dict(record)
    transcripts = _parse_transcripts(out)
    text = extract_model_text(out, model)
    if not text:
        text = _nonempty_text(out.get("qwen_text")) or _nonempty_text(out.get("text")) or ""
    if not text and len(transcripts) == 1:
        only = next(iter(transcripts))
        text = extract_model_text(out, only)
    if not text:
        for key, value in out.items():
            if key.endswith("_text") and key not in {"gold_text", "baseline_text"}:
                text = _nonempty_text(value) or ""
                if text:
                    break
    if text:
        entry = transcripts.get(model)
        if isinstance(entry, dict):
            transcripts[model] = {**entry, "text": text}
        else:
            transcripts[model] = {"text": text}
        out["transcripts"] = transcripts
        out[f"{model}_text"] = text
    return out


def merge_two_records(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    """Merge incoming ASR fields into base by id (transcripts / quality / meta)."""
    merged = dict(base)
    for key in ("source_path", "duration", "sample_rate", "channels", "sha256"):
        if _nonempty_text(merged.get(key)) is None and incoming.get(key) is not None:
            merged[key] = incoming.get(key)

    left_t = _parse_transcripts(merged)
    right_t = _parse_transcripts(incoming)
    for model, entry in right_t.items():
        text = ""
        if isinstance(entry, dict):
            text = _nonempty_text(entry.get("text")) or ""
        else:
            text = _nonempty_text(entry) or ""
        if not text:
            text = extract_model_text(incoming, model)
        if not text:
            continue
        existing = left_t.get(model)
        existing_text = ""
        if isinstance(existing, dict):
            existing_text = _nonempty_text(existing.get("text")) or ""
        elif existing is not None:
            existing_text = _nonempty_text(existing) or ""
        if existing_text:
            continue  # keep first non-empty
        if isinstance(entry, dict):
            left_t[model] = {**entry, "text": text}
        else:
            left_t[model] = {"text": text}
        merged[f"{model}_text"] = text
    merged["transcripts"] = left_t

    left_q = _parse_quality(merged)
    right_q = _parse_quality(incoming)
    for key, value in right_q.items():
        if key not in left_q or left_q.get(key) in (None, "", {}, []):
            left_q[key] = value
    if left_q:
        merged["quality"] = left_q

    # Flat quality_* / {model}_text columns from incoming.
    for key, value in incoming.items():
        if key in {"id", "transcripts", "quality", "labels", "lineage", "status", "errors"}:
            continue
        if key.endswith("_text") or key.startswith("quality_"):
            if _nonempty_text(merged.get(key)) is None and _nonempty_text(value) is not None:
                merged[key] = value
    return merged


def merge_by_id(
    sources: list[tuple[str | None, list[dict[str, Any]]]],
    *,
    how: str = "outer",
) -> list[dict[str, Any]]:
    """Join multiple record lists on id.

    ``sources``: list of (forced_model_or_None, records).
    When forced_model is set, that file's transcript is bound to that model key.
    """
    if how not in {"outer", "inner"}:
        raise ValueError(f"unsupported join how={how!r}")

    id_sets: list[set[str]] = []
    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []

    for index, (forced_model, records) in enumerate(sources):
        present: set[str] = set()
        for record in records:
            sample_id = _as_id(record.get("id"))
            if not sample_id:
                continue
            present.add(sample_id)
            payload = dict(record)
            if forced_model:
                payload = ensure_model_transcript(payload, forced_model)
            if sample_id not in merged:
                merged[sample_id] = payload
                order.append(sample_id)
            else:
                merged[sample_id] = merge_two_records(merged[sample_id], payload)
        id_sets.append(present)
        print(
            f"[INFO] source[{index}] model={forced_model or '*'} "
            f"rows={len(records):,} ids={len(present):,}"
        )

    if how == "inner" and id_sets:
        keep = set.intersection(*id_sets) if id_sets else set()
        before = len(order)
        order = [sample_id for sample_id in order if sample_id in keep]
        print(f"[INFO] inner join: {before:,} → {len(order):,} ids")

    return [merged[sample_id] for sample_id in order]


def flatten_row(record: dict[str, Any], models: list[str], baseline: str) -> dict[str, Any]:
    quality = _parse_quality(record)
    transcripts = _parse_transcripts(record)

    resolved_baseline = _metric(record, quality, "asr_edit_baseline") or baseline
    row: dict[str, Any] = {
        "id": record.get("id"),
        "source_path": record.get("source_path"),
        "duration": record.get("duration"),
        "sample_rate": record.get("sample_rate"),
        "channels": record.get("channels"),
        "baseline_model": resolved_baseline,
    }
    for model in models:
        text_col = f"{model}_text"
        nested = _nonempty_text(_transcript_text(transcripts, model))
        flat = _nonempty_text(record.get(text_col))
        row[text_col] = nested or flat or ""
        if model == resolved_baseline:
            continue
        for metric, label in (
            ("total", "total"),
            ("错字", "错字"),
            ("少字", "少字"),
            ("多字", "多字"),
            ("cer", "cer"),
        ):
            row[f"vs_{resolved_baseline}_{model}_{label}"] = _metric(
                record, quality, f"asr_edit_{model}_{metric}"
            )

    # Also export agreement / evaluation CER fields if present in quality.
    for key, value in quality.items():
        if key.endswith("_cer") or key.endswith("_字准率") or key.endswith("_agreement_cer"):
            row[f"quality_{key}"] = value
        elif key.endswith(("_substitutions", "_deletions", "_insertions", "_reference_length")):
            row[f"quality_{key}"] = value

    payload_raw = _metric(record, quality, "asr_edit_json")
    if isinstance(payload_raw, str) and payload_raw:
        try:
            payload = json.loads(payload_raw)
        except json.JSONDecodeError:
            payload = {}
        if isinstance(payload, dict):
            row["baseline_model"] = payload.get("baseline", resolved_baseline)
            for model, ops in (payload.get("models") or {}).items():
                if not isinstance(ops, dict):
                    continue
                for metric, label in (
                    ("total", "total"),
                    ("错字", "错字"),
                    ("少字", "少字"),
                    ("多字", "多字"),
                    ("cer", "cer"),
                ):
                    col = f"vs_{row['baseline_model']}_{model}_{label}"
                    if row.get(col) is None:
                        row[col] = ops.get(metric)

    status = record.get("status") or {}
    if isinstance(status, str):
        row["status_json"] = status
    elif isinstance(status, dict):
        row["status_json"] = json.dumps(status, ensure_ascii=False)
    errors = record.get("errors") or {}
    if isinstance(errors, str) and errors not in ("", "{}", "null"):
        row["errors_json"] = errors
    elif isinstance(errors, dict) and errors:
        row["errors_json"] = json.dumps(errors, ensure_ascii=False)
    return row


def _load_parquet_duckdb(path: Path) -> list[dict[str, Any]]:
    """Fallback when pyarrow hits page-index / histogram incompatibilities."""
    import duckdb

    uri = path.resolve().as_posix()
    frame = duckdb.connect().execute(f"SELECT * FROM read_parquet('{uri}')").df()
    return frame.to_dict(orient="records")


def load_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        try:
            import pandas as pd

            frame = pd.read_parquet(path)
            return frame.to_dict(orient="records")
        except OSError as exc:
            # Newer writers may embed repetition-level histograms that older
            # pyarrow rejects; duckdb is more tolerant.
            print(f"[WARN] pandas/pyarrow failed on {path.name}: {exc}; trying duckdb")
            return _load_parquet_duckdb(path)
    if suffix == ".jsonl":
        records: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no} invalid JSON: {exc}") from exc
        return records
    raise ValueError(f"Unsupported manifest format: {path.suffix} (use .parquet or .jsonl)")


def write_xlsx(rows: list[dict[str, Any]], path: Path) -> None:
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_excel(tmp, index=False, engine="openpyxl")
    tmp.replace(path)


def parse_model_arg(raw: str) -> tuple[str, Path]:
    text = (raw or "").strip()
    if "=" not in text:
        raise ValueError(f"--model 需要 name=path，收到: {raw!r}")
    name, _, path = text.partition("=")
    name, path = name.strip(), path.strip()
    if not name or not path:
        raise ValueError(f"--model 需要 name=path，收到: {raw!r}")
    return name, Path(path)


def infer_models_from_records(records: list[dict[str, Any]], limit: int = 50) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for record in records[:limit]:
        for model in _parse_transcripts(record):
            if model not in seen:
                seen.add(model)
                names.append(model)
        for key, value in record.items():
            if not key.endswith("_text") or key in {"gold_text", "baseline_text"}:
                continue
            if _nonempty_text(value) is None:
                continue
            model = key[: -len("_text")]
            if model and model not in seen:
                seen.add(model)
                names.append(model)
    return names


def main() -> int:
    parser = argparse.ArgumentParser(
        description="按 id 合并多个 ASR parquet/jsonl 并导出 xlsx"
    )
    parser.add_argument(
        "--manifest",
        action="append",
        default=[],
        type=Path,
        help="输入 parquet/jsonl（可重复；与 --model 二选一或混用）",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="绑定模型到文件：name=/path/to.parquet（可重复，推荐）",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("datasets/exports/multi_asr_export.xlsx"),
        help="输出 xlsx（超限时写 stem-part-NNN.xlsx）",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=DEFAULT_MAX_ROWS,
        help=f"单个 xlsx 最大行数（默认 {DEFAULT_MAX_ROWS}）",
    )
    parser.add_argument(
        "--models",
        default="",
        help="导出的模型列（逗号分隔）。省略则自动从数据推断；"
        "与多个 --manifest 一起用时，按顺序绑定到各文件",
    )
    parser.add_argument(
        "--baseline",
        default="qwen",
        help="基线模型名（用于 vs_* 列命名）",
    )
    parser.add_argument(
        "--how",
        choices=("outer", "inner"),
        default="outer",
        help="多文件 join：outer=保留任一文件出现的 id；inner=只保留所有文件共有的 id",
    )
    args = parser.parse_args()

    sources: list[tuple[str | None, Path]] = []
    try:
        for item in args.model:
            name, path = parse_model_arg(item)
            sources.append((name, path))
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    model_list = [item.strip() for item in str(args.models).split(",") if item.strip()]
    for index, path in enumerate(args.manifest):
        forced = model_list[index] if index < len(model_list) and len(args.manifest) > 1 else None
        sources.append((forced, path))

    if not sources:
        # Backward-compatible default single file.
        default = Path("datasets/manifests/multi_asr_aggregate_source_A.parquet")
        sources.append((None, default))

    for _, path in sources:
        if not path.is_file():
            print(f"[ERROR] manifest not found: {path}", file=sys.stderr)
            return 1

    loaded: list[tuple[str | None, list[dict[str, Any]]]] = []
    for forced_model, path in sources:
        print(f"[INFO] loading {path}")
        records = load_records(path)
        loaded.append((forced_model, records))

    merged_records = merge_by_id(loaded, how=args.how)
    if not merged_records:
        print("[ERROR] 无匹配到的 id", file=sys.stderr)
        return 1

    export_models = model_list
    if not export_models:
        # Prefer explicit --model names, then infer.
        export_models = [name for name, _ in sources if name]
    if not export_models:
        export_models = infer_models_from_records(merged_records)
    if not export_models:
        export_models = [args.baseline]
    # Deduplicate while preserving order.
    seen_models: set[str] = set()
    models: list[str] = []
    for name in export_models:
        if name not in seen_models:
            seen_models.add(name)
            models.append(name)

    print(
        f"[INFO] merged_ids={len(merged_records):,} models={models} "
        f"baseline={args.baseline} how={args.how}"
    )
    flat = [flatten_row(record, models, args.baseline) for record in merged_records]

    max_rows = max(1, int(args.max_rows))
    if len(flat) <= max_rows:
        write_xlsx(flat, args.output)
        print(f"[OK] wrote {len(flat):,} rows → {args.output}")
        return 0

    stem = args.output.with_suffix("")
    parts = 0
    for start in range(0, len(flat), max_rows):
        parts += 1
        chunk = flat[start : start + max_rows]
        out = Path(f"{stem}-part-{parts:03d}.xlsx")
        write_xlsx(chunk, out)
        print(f"[OK] part {parts}: {len(chunk):,} rows → {out}")
    print(f"[OK] total {len(flat):,} rows in {parts} xlsx shards (max_rows={max_rows})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
