#!/usr/bin/env python3
"""036 · 多 batch 分类/标注 parquet → 版本化训练 Manifest 汇总。

从多个 batch 的 classified / *_with_gold parquet 按配方抽取指定类别，
清洗 semantic_risk 中「不做训练 / 不作训练语音信箱」等标记为 hardcase，
写出 datasets/stage1/trainsets/<version>/ 下的完整产物。

清洗先于类别过滤。同 trainset_version 默认不覆盖。

关于自动 voicemail：分类器正则命中的 voicemail 不是人工金标
（见 docs/04-改进需求/进行中/036-标注-训练数据汇总.md §4）；
本脚本只处理标注侧「不做训练*」标记，不改分类器正则。

示例::

  python scripts/assemble_training_parquet.py \\
    --version zh_asr_mix_20260921_v001 \\
    --recipe configs/trainsets/zh_asr_mix_20260921_v001.yaml

  python scripts/assemble_training_parquet.py \\
    --version zh_asr_mix_20260921_v001 \\
    --batch BATCH_A --batch BATCH_B \\
    --categories gold_candidate,semantic_risk,hardcase \\
    --rule-stem classified_five_class_v2_3_min_acoustic
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audio_engine.core.manifest import Manifest, file_sha256
from audio_engine.core.sample import Sample
from audio_engine.core.selection_v3.input_contract import original_audio_sha256

DEFAULT_RULE_STEM = "classified_five_class_v2_3_min_acoustic"
DEFAULT_EXCLUDE_MARKERS = "configs/selection/training_exclude_markers_v1.yaml"
SCRIPT_VERSION = "1.0.0"
AUDIO_KEYS = ("resampled_16k", "pcm_wav", "pcm_to_wav", "raw", "pcm_wav_path")

HELP_EPILOG = """
注意：自动 voicemail（分类器正则）不是金标；本入口只清洗人工「不做训练*」标记。
详见 docs/04-改进需求/进行中/036-标注-训练数据汇总.md §4。
""".strip()


# ──────────────────────────────────────────────
# paths / io
# ──────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def resolve_path(raw: str | Path) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    return path


def default_with_gold_path(batch: str, *, rule_stem: str = DEFAULT_RULE_STEM) -> Path:
    return (
        ROOT
        / "datasets"
        / "stage1"
        / "derived"
        / f"{rule_stem}_{batch}_with_gold.parquet"
    )


def trainset_dir(version: str) -> Path:
    return ROOT / "datasets" / "stage1" / "trainsets" / version


def load_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"YAML 必须是 mapping: {path}")
    return raw


def dump_yaml(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def dump_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# ──────────────────────────────────────────────
# category / markers
# ──────────────────────────────────────────────

def category_before_clean(sample: Sample) -> str:
    """人工纠正优先 reviewed_category → labels.category → type。"""
    labels = sample.labels or {}
    for key in ("reviewed_category", "category", "type"):
        val = labels.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()
    return ""


def _norm_text(text: str) -> str:
    return str(text or "").strip()


def _norm_cmp(text: str) -> str:
    return _norm_text(text).casefold()


@dataclass
class ExcludeMarkersConfig:
    version: str
    target_categories: set[str]
    gold_text_after_clean: str
    markers: list[dict[str, str]]
    near_synonym_warn: list[str]

    @classmethod
    def load(cls, path: Path) -> ExcludeMarkersConfig:
        raw = load_yaml(path)
        markers = []
        for item in raw.get("markers") or []:
            if not isinstance(item, dict):
                continue
            match = str(item.get("match") or "").strip()
            if not match:
                continue
            markers.append(
                {
                    "match": match,
                    "reason": str(item.get("reason") or "exclude_from_training").strip(),
                    "mode": str(item.get("mode") or "exact").strip().lower(),
                }
            )
        targets = raw.get("target_categories_before_clean") or ["semantic_risk"]
        near = [str(x).strip() for x in (raw.get("near_synonym_warn") or []) if str(x).strip()]
        gold_after = raw.get("gold_text_after_clean", "")
        if gold_after is None:
            gold_after = ""
        return cls(
            version=str(raw.get("version") or path.stem),
            target_categories={str(x).strip() for x in targets if str(x).strip()},
            gold_text_after_clean=str(gold_after),
            markers=markers,
            near_synonym_warn=near,
        )


def match_exclude_marker(
    text: str,
    cfg: ExcludeMarkersConfig,
) -> tuple[str, str] | None:
    """命中返回 (reason, matched_marker)；否则 None。"""
    cmp = _norm_cmp(text)
    if not cmp:
        return None
    for item in cfg.markers:
        marker_cmp = _norm_cmp(item["match"])
        mode = item["mode"]
        hit = False
        if mode == "prefix":
            hit = cmp.startswith(marker_cmp)
        else:
            # exact：整串相等
            hit = cmp == marker_cmp
        if hit:
            return item["reason"], item["match"]
    return None


def warn_near_synonym(text: str, cfg: ExcludeMarkersConfig) -> str | None:
    cmp = _norm_cmp(text)
    if not cmp:
        return None
    # 已是正式 marker 则不告警
    if match_exclude_marker(text, cfg) is not None:
        return None
    for phrase in cfg.near_synonym_warn:
        p = _norm_cmp(phrase)
        if p and (cmp == p or p in cmp):
            return phrase
    return None


def apply_exclude_clean(
    sample: Sample,
    *,
    category_pre: str,
    cfg: ExcludeMarkersConfig,
) -> dict[str, Any]:
    """就地清洗；返回明细行字段。"""
    gold = _norm_text(sample.labels.get("gold_text") or "")
    detail: dict[str, Any] = {
        "category_before": category_pre,
        "category_after": category_pre,
        "clean_reason": "",
        "matched_marker": "",
        "has_gold_text": bool(gold),
        "dropped_after_clean": False,
    }
    if category_pre not in cfg.target_categories:
        return detail

    hit = match_exclude_marker(gold, cfg)
    if hit is None:
        return detail

    reason, matched = hit
    sample.labels["gold_text_raw"] = gold
    sample.labels["annotation_raw_text"] = gold
    sample.labels["category_before_clean"] = category_pre
    sample.labels["training_category"] = "hardcase"
    sample.labels["category"] = "hardcase"
    sample.labels["type"] = "hardcase"
    sample.labels["clean_reason"] = reason
    sample.labels["gold_text"] = cfg.gold_text_after_clean
    # 同步清空可能被同步过的 label 训练正文
    if _norm_cmp(str(sample.labels.get("label") or "")) == _norm_cmp(gold):
        sample.labels["label"] = cfg.gold_text_after_clean
    sample.labels["exclude_marker_version"] = cfg.version

    detail["category_after"] = "hardcase"
    detail["clean_reason"] = reason
    detail["matched_marker"] = matched
    detail["has_gold_text"] = bool(_norm_text(cfg.gold_text_after_clean))
    return detail


def has_audio_ref(sample: Sample) -> bool:
    for key in AUDIO_KEYS:
        raw = sample.audio.get(key) if sample.audio else None
        if raw and str(raw).strip():
            return True
    if sample.source_path and str(sample.source_path).strip():
        return True
    return False


# ──────────────────────────────────────────────
# recipe
# ──────────────────────────────────────────────

@dataclass
class BatchSource:
    batch: str
    parquet: Path


@dataclass
class AssembleRecipe:
    trainset_version: str
    categories: list[str]
    batches: list[BatchSource]
    rule_stem: str = DEFAULT_RULE_STEM
    exclude_markers_config: Path = field(
        default_factory=lambda: resolve_path(DEFAULT_EXCLUDE_MARKERS)
    )
    duplicate_audio_policy: str = "fail"  # fail | keep_first

    def to_dict(self) -> dict[str, Any]:
        return {
            "trainset_version": self.trainset_version,
            "rule_stem": self.rule_stem,
            "categories": list(self.categories),
            "batches": [
                {"batch": b.batch, "parquet": _rel_or_abs(b.parquet)} for b in self.batches
            ],
            "exclude_markers_config": _rel_or_abs(self.exclude_markers_config),
            "duplicate_audio_policy": self.duplicate_audio_policy,
        }


def _rel_or_abs(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def parse_categories(raw: str | list[str] | None) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        items = [str(x).strip() for x in raw]
    else:
        items = [x.strip() for x in str(raw).replace(";", ",").split(",")]
    out: list[str] = []
    seen: set[str] = set()
    for c in items:
        if not c or c in seen:
            continue
        seen.add(c)
        out.append(c)
    return out


def load_recipe_from_yaml(path: Path, *, version_override: str | None = None) -> AssembleRecipe:
    raw = load_yaml(path)
    version = str(version_override or raw.get("trainset_version") or "").strip()
    if not version:
        raise ValueError(f"recipe 缺少 trainset_version: {path}")
    categories = parse_categories(raw.get("categories"))
    if not categories:
        raise ValueError("必须显式指定 categories（禁止默认吞全部五类）")
    rule_stem = str(raw.get("rule_stem") or DEFAULT_RULE_STEM).strip()
    batches_raw = raw.get("batches") or []
    if not batches_raw:
        raise ValueError("recipe 必须显式列出 batches")
    batches: list[BatchSource] = []
    for item in batches_raw:
        if not isinstance(item, dict):
            raise ValueError(f"batches 项必须是 mapping: {item!r}")
        batch = str(item.get("batch") or "").strip()
        if not batch:
            raise ValueError(f"batches 项缺少 batch: {item!r}")
        parquet_raw = item.get("parquet")
        if parquet_raw:
            parquet = resolve_path(str(parquet_raw))
        else:
            parquet = default_with_gold_path(batch, rule_stem=rule_stem)
        batches.append(BatchSource(batch=batch, parquet=parquet))
    excl = resolve_path(
        str(raw.get("exclude_markers_config") or DEFAULT_EXCLUDE_MARKERS)
    )
    policy = str(raw.get("duplicate_audio_policy") or "fail").strip().lower()
    if policy not in {"fail", "keep_first"}:
        raise ValueError("duplicate_audio_policy 必须是 fail | keep_first")
    return AssembleRecipe(
        trainset_version=version,
        categories=categories,
        batches=batches,
        rule_stem=rule_stem,
        exclude_markers_config=excl,
        duplicate_audio_policy=policy,
    )


def build_recipe_from_cli(args: argparse.Namespace) -> AssembleRecipe:
    version = str(args.version or "").strip()
    if not version:
        raise ValueError("必须指定 --version")
    categories = parse_categories(args.categories)
    if not categories:
        raise ValueError("必须显式指定 --categories")
    batches_cli = [str(b).strip() for b in (args.batch or []) if str(b).strip()]
    if not batches_cli:
        raise ValueError("无 --recipe 时必须用 --batch 显式列出批次")
    rule_stem = str(args.rule_stem or DEFAULT_RULE_STEM).strip()
    parquet_overrides = list(args.parquet or [])
    batches: list[BatchSource] = []
    for i, batch in enumerate(batches_cli):
        if i < len(parquet_overrides) and str(parquet_overrides[i]).strip():
            parquet = resolve_path(str(parquet_overrides[i]))
        else:
            parquet = default_with_gold_path(batch, rule_stem=rule_stem)
        batches.append(BatchSource(batch=batch, parquet=parquet))
    excl = resolve_path(str(args.exclude_markers_config or DEFAULT_EXCLUDE_MARKERS))
    policy = str(args.duplicate_audio_policy or "fail").strip().lower()
    if policy not in {"fail", "keep_first"}:
        raise ValueError("duplicate_audio_policy 必须是 fail | keep_first")
    return AssembleRecipe(
        trainset_version=version,
        categories=categories,
        batches=batches,
        rule_stem=rule_stem,
        exclude_markers_config=excl,
        duplicate_audio_policy=policy,
    )


# ──────────────────────────────────────────────
# assemble
# ──────────────────────────────────────────────

@dataclass
class AssembleResult:
    samples: list[Sample]
    summary: dict[str, Any]
    details: list[dict[str, Any]]
    lineage: list[dict[str, Any]]
    recipe_snapshot: dict[str, Any]


def assemble(recipe: AssembleRecipe) -> AssembleResult:
    markers_cfg = ExcludeMarkersConfig.load(recipe.exclude_markers_config)
    selected = set(recipe.categories)

    input_table: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    lineage: list[dict[str, Any]] = []
    selected_samples: list[Sample] = []

    # stats
    by_batch_category: dict[str, Counter[str]] = defaultdict(Counter)
    by_batch_duration: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    gold_by_category: dict[str, Counter[str]] = defaultdict(Counter)
    clean_stats = Counter()
    dropped_after_clean = 0
    missing_category = 0
    missing_audio = 0
    near_synonym_warnings: list[dict[str, str]] = []

    # duplicate audio tracking across selected rows
    audio_first: dict[str, tuple[str, str]] = {}  # sha -> (batch, sample_id)
    duplicate_audio: list[dict[str, str]] = []
    skipped_duplicate = 0

    generated_at = _now_iso()

    for src in recipe.batches:
        path = src.parquet
        if not path.is_file():
            raise FileNotFoundError(f"输入 parquet 不存在: {path}")
        sha = file_sha256(path)
        manifest = Manifest.load(path)
        input_table.append(
            {
                "batch": src.batch,
                "parquet": _rel_or_abs(path),
                "sha256": sha,
                "row_count": len(manifest.samples),
            }
        )

        # 稳定：按 sample_id 排序后再处理该 batch
        ordered = sorted(manifest.samples, key=lambda s: str(s.id))
        for sample in ordered:
            s = sample.model_copy(deep=True)
            cat_pre = category_before_clean(s)
            if not cat_pre:
                missing_category += 1

            if not has_audio_ref(s):
                missing_audio += 1

            gold_raw = _norm_text(s.labels.get("gold_text") or "")
            near = warn_near_synonym(gold_raw, markers_cfg)
            if near and cat_pre in markers_cfg.target_categories:
                near_synonym_warnings.append(
                    {
                        "batch": src.batch,
                        "sample_id": s.id,
                        "gold_text": gold_raw,
                        "near_synonym": near,
                    }
                )

            clean_detail = apply_exclude_clean(s, category_pre=cat_pre, cfg=markers_cfg)
            cat_after = str(clean_detail["category_after"] or "")

            if clean_detail["clean_reason"]:
                clean_stats[clean_detail["clean_reason"]] += 1
                clean_stats["to_hardcase_total"] += 1

            # 写入来源元数据
            s.labels["source_batch"] = src.batch
            s.labels["source_parquet"] = _rel_or_abs(path)
            s.labels["trainset_version"] = recipe.trainset_version
            if "training_category" not in s.labels or not str(s.labels.get("training_category") or "").strip():
                s.labels["training_category"] = cat_after
            s.add_lineage(
                "assemble_training_parquet",
                SCRIPT_VERSION,
                {
                    "source_batch": src.batch,
                    "category_before": cat_pre,
                    "category_after": cat_after,
                    "clean_reason": clean_detail["clean_reason"] or None,
                },
            )

            row_detail = {
                "sample_id": s.id,
                "batch": src.batch,
                "category_before": cat_pre,
                "category_after": cat_after,
                "clean_reason": clean_detail["clean_reason"],
                "matched_marker": clean_detail["matched_marker"],
                "has_gold_text": bool(_norm_text(s.labels.get("gold_text") or "")),
                "duration": float(s.duration) if s.duration is not None else None,
                "original_audio_sha256": original_audio_sha256(s),
                "selected": False,
                "dropped_after_clean": False,
                "skip_reason": "",
            }

            if cat_after not in selected:
                if clean_detail["clean_reason"] and cat_pre in selected and cat_after not in selected:
                    dropped_after_clean += 1
                    row_detail["dropped_after_clean"] = True
                    row_detail["skip_reason"] = "dropped_after_clean"
                else:
                    row_detail["skip_reason"] = "category_not_selected"
                details.append(row_detail)
                continue

            ohash = original_audio_sha256(s)
            if ohash:
                if ohash in audio_first:
                    prev_batch, prev_id = audio_first[ohash]
                    dup = {
                        "original_audio_sha256": ohash,
                        "first_batch": prev_batch,
                        "first_sample_id": prev_id,
                        "dup_batch": src.batch,
                        "dup_sample_id": s.id,
                    }
                    duplicate_audio.append(dup)
                    if recipe.duplicate_audio_policy == "fail":
                        raise ValueError(
                            "跨 batch 重复音频（duplicate_audio_policy=fail）: "
                            f"sha={ohash} first=({prev_batch},{prev_id}) "
                            f"dup=({src.batch},{s.id})"
                        )
                    # keep_first
                    skipped_duplicate += 1
                    row_detail["skip_reason"] = "duplicate_audio_keep_first"
                    details.append(row_detail)
                    continue
                audio_first[ohash] = (src.batch, s.id)

            row_detail["selected"] = True
            details.append(row_detail)
            selected_samples.append(s)

            by_batch_category[src.batch][cat_after] += 1
            if s.duration is not None:
                try:
                    by_batch_duration[src.batch][cat_after] += float(s.duration)
                except (TypeError, ValueError):
                    pass
            has_gold = bool(_norm_text(s.labels.get("gold_text") or ""))
            gold_by_category[cat_after]["with_gold" if has_gold else "without_gold"] += 1

            lineage.append(
                {
                    "sample_id": s.id,
                    "source_batch": src.batch,
                    "source_parquet": _rel_or_abs(path),
                    "category_before": cat_pre,
                    "category_after": cat_after,
                    "clean_reason": clean_detail["clean_reason"] or None,
                    "original_audio_sha256": ohash,
                }
            )

    # 全局稳定排序：先 recipe batch 顺序，再 sample_id
    batch_order = {b.batch: i for i, b in enumerate(recipe.batches)}
    selected_samples.sort(
        key=lambda s: (batch_order.get(str(s.labels.get("source_batch") or ""), 10**9), str(s.id))
    )

    counts_table: list[dict[str, Any]] = []
    for src in recipe.batches:
        for cat in recipe.categories:
            n = int(by_batch_category[src.batch].get(cat, 0))
            dur = float(by_batch_duration[src.batch].get(cat, 0.0))
            counts_table.append(
                {
                    "batch": src.batch,
                    "category": cat,
                    "count": n,
                    "duration_sec": round(dur, 3),
                    "duration_hours": round(dur / 3600.0, 4) if dur else 0.0,
                }
            )

    gold_presence = {
        cat: {
            "with_gold_text": int(gold_by_category[cat].get("with_gold", 0)),
            "without_gold_text": int(gold_by_category[cat].get("without_gold", 0)),
        }
        for cat in recipe.categories
    }

    summary: dict[str, Any] = {
        "trainset_version": recipe.trainset_version,
        "generated_at": generated_at,
        "script_version": SCRIPT_VERSION,
        "exclude_markers_version": markers_cfg.version,
        "exclude_markers_config": _rel_or_abs(recipe.exclude_markers_config),
        "rule_stem": recipe.rule_stem,
        "categories": list(recipe.categories),
        "duplicate_audio_policy": recipe.duplicate_audio_policy,
        "total_selected": len(selected_samples),
        "inputs": input_table,
        "counts_by_batch_category": counts_table,
        "clean_stats": {
            "to_hardcase_total": int(clean_stats.get("to_hardcase_total", 0)),
            "exclude_from_training": int(clean_stats.get("exclude_from_training", 0)),
            "exclude_voicemail_marked": int(clean_stats.get("exclude_voicemail_marked", 0)),
            "dropped_after_clean": dropped_after_clean,
        },
        "gold_text_presence": gold_presence,
        "warnings": {
            "duplicate_audio": duplicate_audio,
            "skipped_duplicate_audio": skipped_duplicate,
            "missing_category_field": missing_category,
            "missing_audio_ref": missing_audio,
            "near_synonym_unrecognized": near_synonym_warnings[:200],
            "near_synonym_unrecognized_count": len(near_synonym_warnings),
        },
        "notes": {
            "voicemail_auto_not_gold": (
                "自动 voicemail（分类器正则）不是金标；"
                "本汇总只清洗人工「不做训练/不作训练语音信箱」等标记。"
                "见 036 §4。"
            ),
            "clean_before_filter": True,
            "unique_key": "(source_batch, sample_id)",
        },
    }

    recipe_snapshot = recipe.to_dict()
    recipe_snapshot["generated_at"] = generated_at
    recipe_snapshot["script_version"] = SCRIPT_VERSION
    recipe_snapshot["input_sha256"] = [
        {"batch": x["batch"], "parquet": x["parquet"], "sha256": x["sha256"]}
        for x in input_table
    ]
    recipe_snapshot["exclude_markers_version"] = markers_cfg.version

    return AssembleResult(
        samples=selected_samples,
        summary=summary,
        details=details,
        lineage=lineage,
        recipe_snapshot=recipe_snapshot,
    )


def render_summary_md(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# 训练集汇总 · `{summary['trainset_version']}`")
    lines.append("")
    lines.append(f"- 生成时间（UTC）：`{summary['generated_at']}`")
    lines.append(f"- 脚本版本：`{summary['script_version']}`")
    lines.append(f"- 清洗规则：`{summary['exclude_markers_version']}`（`{summary['exclude_markers_config']}`）")
    lines.append(f"- rule_stem：`{summary['rule_stem']}`")
    lines.append(f"- 入选类别：{', '.join(summary['categories'])}")
    lines.append(f"- 重复音频策略：`{summary['duplicate_audio_policy']}`")
    lines.append(f"- **入选总数**：{summary['total_selected']}")
    lines.append("")
    notes = summary.get("notes") or {}
    if notes.get("voicemail_auto_not_gold"):
        lines.append(f"> 说明：{notes['voicemail_auto_not_gold']}")
        lines.append("")

    lines.append("## 输入表")
    lines.append("")
    lines.append("| batch | parquet | SHA256 | 原始行数 |")
    lines.append("| --- | --- | --- | ---: |")
    for row in summary.get("inputs") or []:
        sha = str(row.get("sha256") or "")
        sha_short = sha[:12] + "…" if len(sha) > 12 else sha
        lines.append(
            f"| `{row['batch']}` | `{row['parquet']}` | `{sha_short}` | {row['row_count']} |"
        )
    lines.append("")

    lines.append("## 按 batch × 清洗后 category")
    lines.append("")
    lines.append("| batch | category | 条数 | 时长(h) |")
    lines.append("| --- | --- | ---: | ---: |")
    for row in summary.get("counts_by_batch_category") or []:
        lines.append(
            f"| `{row['batch']}` | `{row['category']}` | {row['count']} | {row['duration_hours']} |"
        )
    lines.append("")

    cs = summary.get("clean_stats") or {}
    lines.append("## 清洗动作")
    lines.append("")
    lines.append(f"- 不做训练 → hardcase 合计：{cs.get('to_hardcase_total', 0)}")
    lines.append(f"- 其中 `exclude_from_training`：{cs.get('exclude_from_training', 0)}")
    lines.append(f"- 其中语音信箱标记 `exclude_voicemail_marked`：{cs.get('exclude_voicemail_marked', 0)}")
    lines.append(f"- 清洗后因类别不在配方而丢弃 `dropped_after_clean`：{cs.get('dropped_after_clean', 0)}")
    lines.append("")

    lines.append("## 有/无 gold_text（按清洗后类别，入选样本）")
    lines.append("")
    lines.append("| category | 有 gold_text | 无 gold_text |")
    lines.append("| --- | ---: | ---: |")
    for cat, g in (summary.get("gold_text_presence") or {}).items():
        lines.append(
            f"| `{cat}` | {g.get('with_gold_text', 0)} | {g.get('without_gold_text', 0)} |"
        )
    lines.append("")

    warns = summary.get("warnings") or {}
    lines.append("## 告警")
    lines.append("")
    lines.append(f"- 跨 batch 重复音频条数：{len(warns.get('duplicate_audio') or [])}")
    lines.append(f"- keep_first 跳过：{warns.get('skipped_duplicate_audio', 0)}")
    lines.append(f"- 缺类别字段：{warns.get('missing_category_field', 0)}")
    lines.append(f"- 缺音频引用：{warns.get('missing_audio_ref', 0)}")
    lines.append(
        f"- 近义「不做训练」未自动清洗：{warns.get('near_synonym_unrecognized_count', 0)}"
    )
    dups = warns.get("duplicate_audio") or []
    if dups:
        lines.append("")
        lines.append("重复音频样例（最多 20）：")
        for d in dups[:20]:
            lines.append(
                f"  - sha=`{d['original_audio_sha256'][:16]}…` "
                f"first=({d['first_batch']},{d['first_sample_id']}) "
                f"dup=({d['dup_batch']},{d['dup_sample_id']})"
            )
    near = warns.get("near_synonym_unrecognized") or []
    if near:
        lines.append("")
        lines.append("近义告警样例（最多 20）：")
        for n in near[:20]:
            lines.append(
                f"  - `{n['batch']}`/`{n['sample_id']}` gold={n['gold_text']!r} ~ {n['near_synonym']!r}"
            )
    lines.append("")
    lines.append("明细全量见同目录 `details.csv` / `lineage.json`。")
    lines.append("")
    return "\n".join(lines)


def write_details_csv(details: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id",
        "batch",
        "category_before",
        "category_after",
        "clean_reason",
        "matched_marker",
        "has_gold_text",
        "duration",
        "original_audio_sha256",
        "selected",
        "dropped_after_clean",
        "skip_reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in details:
            writer.writerow(row)


def write_outputs(
    result: AssembleResult,
    out_dir: Path,
    *,
    overwrite: bool = False,
) -> Path:
    if out_dir.exists() and any(out_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"trainset 版本目录已存在且非空（默认不覆盖）: {out_dir}；"
                "若确认重写请加 --overwrite"
            )
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.parquet"
    Manifest.from_samples(result.samples).save(manifest_path)

    dump_yaml(result.recipe_snapshot, out_dir / "recipe.yaml")
    dump_json(result.summary, out_dir / "summary.json")
    (out_dir / "summary.md").write_text(render_summary_md(result.summary), encoding="utf-8")
    dump_json(result.lineage, out_dir / "lineage.json")
    write_details_csv(result.details, out_dir / "details.csv")
    return out_dir


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="036 · 多 batch 标注/分类 parquet → 版本化训练 Manifest",
        epilog=HELP_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--version", help="trainset_version（业务版本号，非时间戳冒充）")
    p.add_argument("--recipe", help="配方 YAML（优先）")
    p.add_argument("--batch", action="append", default=[], help="批次名；可重复")
    p.add_argument(
        "--parquet",
        action="append",
        default=[],
        help="与 --batch 顺序对应的显式 parquet 路径；可省略则用默认 with_gold 路径",
    )
    p.add_argument(
        "--categories",
        help="逗号分隔类别，如 gold_candidate,semantic_risk,hardcase",
    )
    p.add_argument(
        "--rule-stem",
        default=DEFAULT_RULE_STEM,
        help=f"默认输入 stem（默认 {DEFAULT_RULE_STEM}）",
    )
    p.add_argument(
        "--exclude-markers-config",
        default=DEFAULT_EXCLUDE_MARKERS,
        help="不做训练标记配置",
    )
    p.add_argument(
        "--duplicate-audio-policy",
        choices=("fail", "keep_first"),
        default="fail",
        help="跨 batch 同 original_audio_sha256 策略（默认 fail）",
    )
    p.add_argument(
        "--output-dir",
        help="输出目录（默认 datasets/stage1/trainsets/<version>/）",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖已存在的同名版本目录",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.recipe:
            recipe = load_recipe_from_yaml(
                resolve_path(args.recipe),
                version_override=str(args.version).strip() if args.version else None,
            )
            # CLI 可覆盖部分字段
            if args.categories:
                recipe.categories = parse_categories(args.categories)
            if args.duplicate_audio_policy:
                recipe.duplicate_audio_policy = args.duplicate_audio_policy
            if args.exclude_markers_config and args.exclude_markers_config != DEFAULT_EXCLUDE_MARKERS:
                recipe.exclude_markers_config = resolve_path(args.exclude_markers_config)
        else:
            recipe = build_recipe_from_cli(args)

        out_dir = (
            resolve_path(args.output_dir)
            if args.output_dir
            else trainset_dir(recipe.trainset_version)
        )

        print(
            f"[INFO] assemble trainset={recipe.trainset_version} "
            f"batches={len(recipe.batches)} categories={recipe.categories}"
        )
        print(
            "[INFO] 自动 voicemail 不是金标；本入口只清洗人工「不做训练*」标记（036 §4）。"
        )
        result = assemble(recipe)
        write_outputs(result, out_dir, overwrite=bool(args.overwrite))

        print(render_summary_md(result.summary))
        print(f"[OK] → {out_dir}")
        print(f"     manifest.parquet  selected={result.summary['total_selected']}")
        print(f"     summary.md / summary.json / recipe.yaml / lineage.json / details.csv")
        return 0
    except FileExistsError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
