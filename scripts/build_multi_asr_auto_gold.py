#!/usr/bin/env python3
"""多路转写 parquet 取交集，完全一致的 id 汇总为 auto_gold。

每个 batch 输入 qwen1 / qwen2 / glm1 / glm2，sv1 可选。只保留所有输入
都有的 id，且各路 ``plain_transcript_text`` 后文本相同且非空的样本。
以 qwen1 行为底表，合并各路 transcripts，打上 auto_gold 标签后写出一个 parquet。

比较用项目统一的 ``plain_transcript_text``（去控制标签、标点、空白）。
金标文本取 qwen1 原文（保留标点）。

示例::

    python scripts/build_multi_asr_auto_gold.py ^
        --name 0908-30000 ^
        --qwen1 datasets/stage1/asr/qwen3-asr-1_asr_0908-30000.parquet ^
        --qwen2 datasets/stage1/asr/qwen3-asr-2_asr_0908-30000.parquet ^
        --glm1 datasets/stage1/asr/glm-asr-1_asr_0908-30000.parquet ^
        --glm2 datasets/stage1/asr/glm-asr-2_asr_0908-30000.parquet ^
        --sv1 datasets/stage1/asr/sensevoice-asr-1_asr_0908-30000.parquet ^
        --output-dir tmp/1026/auto_gold

多 batch 用 YAML（每个元素写一份）::

    batches:
      - name: 0908-30000
        qwen1: datasets/stage1/asr/qwen3-asr-1_asr_0908-30000.parquet
        qwen2: datasets/stage1/asr/qwen3-asr-2_asr_0908-30000.parquet
        glm1: datasets/stage1/asr/glm-asr-1_asr_0908-30000.parquet
        glm2: datasets/stage1/asr/glm-asr-2_asr_0908-30000.parquet
        sv1: datasets/stage1/asr/sensevoice-asr-1_asr_0908-30000.parquet
      - name: 0914-mixed-30000
        qwen1: datasets/stage1/asr/qwen-basr-1_asr_0914-mixed-30000.parquet
        qwen2: datasets/stage1/asr/qwen-basr-2_asr_0914-mixed-30000.parquet
        glm1: datasets/stage1/asr/glm-base-1_asr_0914-mixed-30000.parquet
        glm2: datasets/stage1/asr/glm-base-2_asr_0914-mixed-30000.parquet

    python scripts/build_multi_asr_auto_gold.py --config batches.yaml --output-dir tmp/1026/auto_gold
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from audio_engine.core.artifacts import atomic_path
from audio_engine.core.transcript_reconcile import plain_transcript_text

REQUIRED_ROLES = ("qwen1", "qwen2", "glm1", "glm2")
OPTIONAL_ROLES = ("sv1",)
ALL_ROLES = REQUIRED_ROLES + OPTIONAL_ROLES
PRIMARY_ROLE = "qwen1"
BUCKET_AUTO_GOLD = "auto_gold"


@dataclass(frozen=True)
class BatchSpec:
    name: str
    paths: dict[str, Path]


@dataclass(frozen=True)
class RouteData:
    role: str
    path: Path
    raw_by_id: dict[str, str]
    entry_by_id: dict[str, dict[str, Any]]
    source_key: str


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _load_mapping(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return None
    raw = value.strip()
    if not raw or raw[0] not in "{[":
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _entry_text(entry: Any) -> str:
    if entry is None:
        return ""
    if isinstance(entry, dict):
        return _text(entry.get("text"))
    return _text(entry)


def _as_entry(text: str, *, model: str = "") -> dict[str, Any]:
    entry: dict[str, Any] = {"text": text}
    if model:
        entry["model"] = model
    return entry


def _resolve_path(raw: str | Path) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    return path


def _pick_flat_text_column(names: list[str], role: str) -> str | None:
    preferred = (
        f"{role}_text",
        f"{role.replace('1', '_1').replace('2', '_2')}_text",
    )
    for name in preferred:
        if name in names:
            return name
    text_cols = [name for name in names if name.endswith("_text")]
    if len(text_cols) == 1:
        return text_cols[0]
    return None


def load_route(role: str, path: Path) -> RouteData:
    """读一路 parquet：id -> 原文，以及可写入 transcripts 的 entry。"""
    if not path.is_file():
        raise ValueError(f"{role}: 找不到文件 {path}")
    table = pq.read_table(path)
    names = table.schema.names
    if "id" not in names:
        raise ValueError(f"{role}: {path.name} 缺少 id 列，实际列 {names}")

    flat_col = _pick_flat_text_column(names, role)
    has_transcripts = "transcripts" in names
    if flat_col is None and not has_transcripts:
        raise ValueError(
            f"{role}: {path.name} 需要 transcripts 或 *_text 列，实际列 {names}"
        )

    columns = ["id"]
    if has_transcripts:
        columns.append("transcripts")
    if flat_col is not None:
        columns.append(flat_col)
    rows = table.select(columns).to_pylist()

    raw_by_id: dict[str, str] = {}
    entry_by_id: dict[str, dict[str, Any]] = {}
    source_key_counts: dict[str, int] = {}

    for row in rows:
        sample_id = _text(row.get("id")).strip()
        if not sample_id:
            raise ValueError(f"{role}: {path.name} 有空 id")
        if sample_id in raw_by_id:
            raise ValueError(f"{role}: {path.name} id 重复: {sample_id}")

        mapping = _load_mapping(row.get("transcripts")) if has_transcripts else None
        raw = ""
        entry: dict[str, Any] | None = None
        source_key = ""

        if mapping:
            if len(mapping) == 1:
                source_key, value = next(iter(mapping.items()))
                source_key = str(source_key)
                raw = _entry_text(value)
                entry = value if isinstance(value, dict) else _as_entry(raw, model=source_key)
            elif role in mapping:
                source_key = role
                value = mapping[role]
                raw = _entry_text(value)
                entry = value if isinstance(value, dict) else _as_entry(raw, model=role)
            else:
                for key, value in mapping.items():
                    text = _entry_text(value)
                    if text:
                        source_key = str(key)
                        raw = text
                        entry = value if isinstance(value, dict) else _as_entry(raw, model=source_key)
                        break

        if not raw and flat_col is not None:
            raw = _text(row.get(flat_col))
            if entry is None:
                entry = _as_entry(raw, model=role)
            if not source_key:
                source_key = flat_col or role

        if entry is None:
            entry = _as_entry(raw, model=role)
        if not source_key:
            source_key = role

        raw_by_id[sample_id] = raw
        entry_by_id[sample_id] = dict(entry)
        source_key_counts[source_key] = source_key_counts.get(source_key, 0) + 1

    source_key = (
        max(source_key_counts.items(), key=lambda item: item[1])[0]
        if source_key_counts
        else role
    )
    return RouteData(
        role=role,
        path=path,
        raw_by_id=raw_by_id,
        entry_by_id=entry_by_id,
        source_key=source_key,
    )


def find_consensus_ids(routes: list[RouteData]) -> tuple[list[str], dict[str, int]]:
    """交集内 plain 全相同且非空的 id，按 primary 出现顺序。"""
    primary = routes[0]
    id_sets = [set(route.raw_by_id) for route in routes]
    common = set.intersection(*id_sets)
    only_counts = {
        route.role: len(set(route.raw_by_id) - common) for route in routes
    }

    keep: list[str] = []
    for sample_id in primary.raw_by_id:
        if sample_id not in common:
            continue
        plains = [plain_transcript_text(route.raw_by_id[sample_id]) for route in routes]
        if not plains[0]:
            continue
        if all(text == plains[0] for text in plains[1:]):
            keep.append(sample_id)
    return keep, only_counts


def _update_labels(raw_labels: Any, *, gold_text: str, routes: list[str]) -> str:
    labels = _load_mapping(raw_labels) or {}
    labels.update(
        {
            "type": BUCKET_AUTO_GOLD,
            "classification_bucket": BUCKET_AUTO_GOLD,
            "gold_text": gold_text,
            "label": gold_text,
            "gold_source": PRIMARY_ROLE,
            "decision": "auto_accept",
            "annotation_state": "auto_accepted",
            "auto_gold_routes": list(routes),
        }
    )
    return json.dumps(labels, ensure_ascii=False)


def build_auto_gold_table(routes: list[RouteData], keep_ids: list[str]) -> pa.Table:
    primary = routes[0]
    base = pq.read_table(primary.path)
    if "id" not in base.schema.names:
        raise ValueError(f"{primary.path.name} 缺少 id 列")

    id_list = base.column("id").to_pylist()
    id_to_index = { _text(sample_id): index for index, sample_id in enumerate(id_list) }
    missing = [sample_id for sample_id in keep_ids if sample_id not in id_to_index]
    if missing:
        raise ValueError(f"主键表缺少已筛选 id，例如: {missing[:5]}")

    indices = [id_to_index[sample_id] for sample_id in keep_ids]
    selected = base.take(pa.array(indices, type=pa.int64()))
    rows = selected.to_pylist()
    role_names = [route.role for route in routes]

    out_rows: list[dict[str, Any]] = []
    for row in rows:
        sample_id = _text(row.get("id"))
        transcripts = _load_mapping(row.get("transcripts")) or {}
        for route in routes:
            entry = dict(route.entry_by_id[sample_id])
            if "text" not in entry:
                entry["text"] = route.raw_by_id[sample_id]
            transcripts[route.role] = entry

        gold_text = primary.raw_by_id[sample_id]
        row["transcripts"] = json.dumps(transcripts, ensure_ascii=False)
        row["labels"] = _update_labels(row.get("labels"), gold_text=gold_text, routes=role_names)
        row["type"] = BUCKET_AUTO_GOLD
        row["bucket"] = BUCKET_AUTO_GOLD
        row["gold_text"] = gold_text
        row["label_type"] = BUCKET_AUTO_GOLD
        row["label_classification_bucket"] = BUCKET_AUTO_GOLD
        row["label_gold_text"] = gold_text
        row["label_label"] = gold_text
        row["label_gold_source"] = PRIMARY_ROLE
        row["label_decision"] = "auto_accept"
        for route in routes:
            row[f"{route.role}_text"] = route.raw_by_id[sample_id]
        out_rows.append(row)

    if not out_rows:
        # 空表也保留主键 schema，方便下游读
        empty = selected.slice(0, 0)
        extras = {
            "type": pa.array([], type=pa.string()),
            "bucket": pa.array([], type=pa.string()),
            "gold_text": pa.array([], type=pa.string()),
            "label_type": pa.array([], type=pa.string()),
            "label_classification_bucket": pa.array([], type=pa.string()),
            "label_gold_text": pa.array([], type=pa.string()),
            "label_label": pa.array([], type=pa.string()),
            "label_gold_source": pa.array([], type=pa.string()),
            "label_decision": pa.array([], type=pa.string()),
        }
        for route in routes:
            extras[f"{route.role}_text"] = pa.array([], type=pa.string())
        for name, array in extras.items():
            if name in empty.schema.names:
                empty = empty.set_column(empty.schema.get_field_index(name), name, array)
            else:
                empty = empty.append_column(name, array)
        return empty

    return pa.Table.from_pylist(out_rows)


def process_batch(spec: BatchSpec, output_dir: Path) -> dict[str, Any]:
    ordered_roles = [role for role in ALL_ROLES if role in spec.paths]
    missing_required = [role for role in REQUIRED_ROLES if role not in spec.paths]
    if missing_required:
        raise ValueError(f"batch {spec.name}: 缺少必填角色 {missing_required}")

    routes = [load_route(role, spec.paths[role]) for role in ordered_roles]
    keep_ids, only_counts = find_consensus_ids(routes)
    table = build_auto_gold_table(routes, keep_ids)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"auto_gold_{spec.name}.parquet"
    with atomic_path(out_path) as tmp:
        pq.write_table(table, tmp, compression="zstd")

    summary = {
        "name": spec.name,
        "routes": {
            route.role: {
                "path": str(route.path),
                "rows": len(route.raw_by_id),
                "source_key": route.source_key,
                "only": only_counts[route.role],
            }
            for route in routes
        },
        "intersection": len(set.intersection(*[set(route.raw_by_id) for route in routes])),
        "auto_gold": len(keep_ids),
        "compare": "plain_transcript_text",
        "gold_source": PRIMARY_ROLE,
        "output": str(out_path),
    }
    summary_path = output_dir / f"auto_gold_{spec.name}_summary.json"
    with atomic_path(summary_path) as tmp:
        tmp.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary["summary"] = str(summary_path)
    return summary


def load_batches_from_config(path: Path) -> list[BatchSpec]:
    raw = yaml.safe_load(_resolve_path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "batches" not in raw:
        raise ValueError(f"配置需要顶层 batches 列表: {path}")
    items = raw["batches"]
    if not isinstance(items, list) or not items:
        raise ValueError(f"batches 必须是非空列表: {path}")
    specs: list[BatchSpec] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"batches[{index}] 必须是 mapping")
        name = str(item.get("name") or "").strip()
        if not name:
            raise ValueError(f"batches[{index}] 缺少 name")
        paths: dict[str, Path] = {}
        for role in ALL_ROLES:
            if role not in item or item[role] in (None, ""):
                continue
            paths[role] = _resolve_path(item[role])
        specs.append(BatchSpec(name=name, paths=paths))
    return specs


def batch_from_args(args: argparse.Namespace) -> BatchSpec:
    name = str(args.name or "").strip()
    if not name:
        raise ValueError("单 batch 模式需要 --name")
    paths: dict[str, Path] = {}
    for role in ALL_ROLES:
        value = getattr(args, role, None)
        if value is None:
            continue
        paths[role] = _resolve_path(value)
    return BatchSpec(name=name, paths=paths)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="多路转写取交集，完全一致的 id 汇总为 auto_gold parquet",
    )
    parser.add_argument("--config", type=Path, help="多 batch YAML，含 batches 列表")
    parser.add_argument("--name", type=str, help="单 batch 名称，输出 auto_gold_{name}.parquet")
    parser.add_argument("--qwen1", type=Path, help="qwen1 转写 parquet")
    parser.add_argument("--qwen2", type=Path, help="qwen2 转写 parquet")
    parser.add_argument("--glm1", type=Path, help="glm1 转写 parquet")
    parser.add_argument("--glm2", type=Path, help="glm2 转写 parquet")
    parser.add_argument("--sv1", type=Path, help="sensevoice / sv1 转写 parquet，可选")
    parser.add_argument("--output-dir", type=Path, required=True, help="输出目录")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    output_dir = _resolve_path(args.output_dir)

    if args.config:
        specs = load_batches_from_config(args.config)
    else:
        for role in REQUIRED_ROLES:
            if getattr(args, role, None) is None:
                raise SystemExit(f"缺少 --{role}（或改用 --config）")
        specs = [batch_from_args(args)]

    summaries: list[dict[str, Any]] = []
    for spec in specs:
        summary = process_batch(spec, output_dir)
        summaries.append(summary)
        print(
            f"[{spec.name}] auto_gold={summary['auto_gold']} "
            f"intersection={summary['intersection']} -> {summary['output']}"
        )
        for role, info in summary["routes"].items():
            print(f"  {role}: rows={info['rows']} only={info['only']} key={info['source_key']}")

    if len(summaries) > 1:
        all_path = output_dir / "auto_gold_all_summary.json"
        with atomic_path(all_path) as tmp:
            tmp.write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"all_summary {all_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
