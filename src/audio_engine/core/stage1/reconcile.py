"""Delivery reconciliation gates for stage-1 formal success."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from audio_engine.core.manifest import Manifest
from audio_engine.core.selection_v3.input_contract import (
    align_run_manifest,
    original_audio_sha256,
    validate_base_snapshot,
)
from audio_engine.core.stage1.cache_policy import FAMILY_RUN_ALIASES, all_run_aliases
from audio_engine.core.stage1.job import SELECTION_RULE


FIVE_CLASSES = {
    "clear_semantic",
    "environment_noise",
    "human_noise",
    "non_speech",
    "foreign_language",
}


@dataclass
class ReconcileResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
            "stats": self.stats,
        }


def _load_identity(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"identity 不是映射: {path}")
    return raw


def discover_xlsx_parts(output: Path) -> list[Path]:
    if output.is_file():
        return [output]
    stem = output.with_suffix("")
    parts = sorted(stem.parent.glob(f"{stem.name}-part-*.xlsx"))
    return parts


def reconcile_delivery(
    *,
    batch: str,
    cleaned: Path,
    asr_paths: dict[str, Path],
    registered_identities: dict[str, Path],
    classified: Path,
    export_xlsx: Path,
    max_xlsx_rows: int = 20000,
) -> ReconcileResult:
    errors: list[str] = []
    warnings: list[str] = []
    stats: dict[str, Any] = {"batch": batch, "selection_rule": SELECTION_RULE}

    # --- required ASR routes ---
    expected = all_run_aliases()
    missing_asr = [alias for alias in expected if not asr_paths.get(alias) or not asr_paths[alias].is_file()]
    if missing_asr:
        errors.append(f"ASR 路次缺失（不能包装成完成）: {missing_asr}")

    missing_id = [
        alias
        for alias in expected
        if not registered_identities.get(alias) or not registered_identities[alias].is_file()
    ]
    if missing_id:
        errors.append(f"run identity 登记缺失: {missing_id}")

    identities: list[dict[str, Any]] = []
    if not missing_id:
        for alias in expected:
            identities.append(_load_identity(registered_identities[alias]))
        exec_ids = [str(item.get("execution_id") or "") for item in identities]
        art_ids = [str(item.get("artifact_id") or "") for item in identities]
        if len(set(exec_ids)) != 6 or any(not x for x in exec_ids):
            errors.append("execution_id 必须六路齐全且互异")
        if len(set(art_ids)) != 6 or any(not x for x in art_ids):
            errors.append("artifact_id 必须六路齐全且互异；禁止复制登记冒充双跑")
        stats["execution_ids"] = exec_ids
        stats["artifact_ids"] = art_ids

    if not cleaned.is_file():
        errors.append(f"cleaned 不存在: {cleaned}")
        return ReconcileResult(ok=False, errors=errors, warnings=warnings, stats=stats)

    cleaned_samples = list(Manifest.load(cleaned))
    stats["cleaned_count"] = len(cleaned_samples)
    try:
        indexed = validate_base_snapshot(cleaned_samples)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"cleaned 快照校验失败: {exc}")
        indexed = {}

    # --- align each ASR route ---
    asr_missing_hash = 0
    asr_extra_ids = 0
    asr_row_counts: dict[str, int] = {}
    for alias in expected:
        path = asr_paths.get(alias)
        if not path or not path.is_file():
            continue
        incoming = list(Manifest.load(path))
        asr_row_counts[alias] = len(incoming)
        if not indexed:
            continue
        try:
            alignment = align_run_manifest(
                indexed,
                incoming,
                transcript_key=alias,
                path=str(path),
                id_policy="left",
            )
        except ValueError as exc:
            errors.append(f"{alias}: 对齐失败 {exc}")
            continue
        unchecked = int(alignment.get("original_audio_sha256_unchecked") or 0)
        extra = int(alignment.get("extra_ids") or 0)
        hash_mismatch = int(alignment.get("original_audio_sha256_mismatches") or 0)
        asr_missing_hash += unchecked + hash_mismatch
        asr_extra_ids += extra
        if unchecked or extra or hash_mismatch:
            errors.append(
                f"{alias}: 与 cleaned 对齐失败 "
                f"unchecked={unchecked} hash_mismatch={hash_mismatch} extra_ids={extra}"
            )
    stats["asr_row_counts"] = asr_row_counts
    stats["asr_missing_hash"] = asr_missing_hash
    stats["asr_extra_ids"] = asr_extra_ids

    # duplicate keys inside cleaned
    cleaned_keys = [(s.id, original_audio_sha256(s)) for s in cleaned_samples]
    if len(cleaned_keys) != len(set(cleaned_keys)):
        errors.append("cleaned 存在重复 sample_id+original_audio_sha256")

    if not classified.is_file():
        errors.append(f"classified 不存在: {classified}")
        return ReconcileResult(ok=False, errors=errors, warnings=warnings, stats=stats)

    classified_samples = list(Manifest.load(classified))
    stats["classified_count"] = len(classified_samples)
    classified_ids = {s.id for s in classified_samples}
    cleaned_ids = {s.id for s in cleaned_samples}
    missing_from_clean = sorted(cleaned_ids - classified_ids)
    extra_in_class = sorted(classified_ids - cleaned_ids)
    if missing_from_clean:
        errors.append(
            f"cleaned→classified 丢失 {len(missing_from_clean)} 条（示例 {missing_from_clean[:5]}）"
        )
    if extra_in_class:
        errors.append(
            f"classified 出现 cleaned 中不存在的 id {len(extra_in_class)} 条"
        )
    if len(classified_ids) != len(classified_samples):
        errors.append("classified 存在重复 sample_id")

    # rule / five-class / excluded / DNSMOS fallback accounting
    rule_mismatch = 0
    missing_bucket = 0
    classified_no_category = 0
    excluded_no_reason = 0
    five_class_counter: Counter[str] = Counter()
    dnsmos_unavailable = 0
    v2_fallback = 0
    unavailable_family_total = 0
    for sample in classified_samples:
        labels = sample.labels or {}
        rule = str(
            labels.get("rule_version")
            or labels.get("selection_policy_version")
            or ""
        )
        if rule and SELECTION_RULE not in rule:
            rule_mismatch += 1
        bucket = str(labels.get("classification_bucket") or "").strip()
        if not bucket:
            missing_bucket += 1
        outcome = str(labels.get("outcome") or labels.get("classification_outcome") or "").strip()
        category = str(labels.get("category") or "").strip()
        if outcome == "classified" or bucket in {
            "consensus_gold",
            "pseudo_high",
            "classification_bucket",
        } or category:
            if category:
                five_class_counter[category] += 1
                if category not in FIVE_CLASSES:
                    errors.append(f"非法五类 category={category} sample={sample.id}")
            elif outcome == "classified":
                classified_no_category += 1
        if outcome == "excluded" or bucket.startswith("excluded") or labels.get("excluded"):
            reasons = labels.get("classification_reason_codes") or labels.get(
                "exclude_reason"
            ) or labels.get("classification_reason")
            if not reasons:
                excluded_no_reason += 1
        if labels.get("v2_fallback") is True:
            v2_fallback += 1
        dnsmos_status = str(
            labels.get("dnsmos_status")
            or (sample.quality or {}).get("dnsmos_status")
            or ""
        ).lower()
        if dnsmos_status == "unavailable":
            dnsmos_unavailable += 1
        unavailable_family_total += int(labels.get("unavailable_family_count") or 0)

    if rule_mismatch:
        errors.append(f"{rule_mismatch} 条样本 rule_version 非 {SELECTION_RULE}")
    if missing_bucket:
        errors.append(f"{missing_bucket} 条缺少 classification_bucket；导出门禁失败")
    if classified_no_category:
        errors.append(f"{classified_no_category} 条 classified 无 category")
    if excluded_no_reason:
        errors.append(f"{excluded_no_reason} 条 excluded 缺少原因")

    stats["five_class_counts"] = dict(five_class_counter)
    stats["dnsmos_unavailable"] = dnsmos_unavailable
    stats["v2_fallback"] = v2_fallback
    stats["unavailable_family_total"] = unavailable_family_total
    stats["asr_route_missing"] = missing_asr
    # Keep DNSMOS legal fallback separate from ASR route absence.
    stats["accounting"] = {
        "asr_routes_missing": missing_asr,
        "dnsmos_unavailable_or_v2_fallback": {
            "dnsmos_unavailable": dnsmos_unavailable,
            "v2_fallback": v2_fallback,
        },
    }

    # --- XLSX vs Parquet ---
    parts = discover_xlsx_parts(export_xlsx)
    if not parts:
        errors.append(f"导出 XLSX 不存在: {export_xlsx}")
    else:
        try:
            import pandas as pd

            frames = [pd.read_excel(path) for path in parts]
            xlsx_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            xlsx_ids = [str(x) for x in xlsx_df.get("sample_id", []).tolist()]
            stats["xlsx_parts"] = [str(p) for p in parts]
            stats["xlsx_rows"] = len(xlsx_ids)
            if len(xlsx_ids) != len(classified_samples):
                errors.append(
                    f"XLSX 行数 {len(xlsx_ids)} != classified {len(classified_samples)}"
                )
            if len(xlsx_ids) != len(set(xlsx_ids)):
                errors.append("XLSX 存在重复 sample_id")
            if set(xlsx_ids) != classified_ids:
                errors.append("XLSX 与 classified 样本集合不一致")
            for part in parts:
                n = len(pd.read_excel(part))
                if n > max_xlsx_rows:
                    errors.append(f"{part} 行数 {n} 超过 max_rows={max_xlsx_rows}")
            if "category" in xlsx_df.columns and "sample_id" in xlsx_df.columns:
                by_id = {
                    s.id: str((s.labels or {}).get("category") or "")
                    for s in classified_samples
                }
                mismatch = 0
                for _, row in xlsx_df.iterrows():
                    sid = str(row["sample_id"])
                    cat = str(row.get("category") or "")
                    if by_id.get(sid, "") != cat:
                        mismatch += 1
                if mismatch:
                    errors.append(f"XLSX 与 Parquet category 不一致 {mismatch} 条")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"读取 XLSX 对账失败: {exc}")

    # family completeness: cannot drop failed family to fake 3 families
    for family, aliases in FAMILY_RUN_ALIASES.items():
        if any(alias in missing_asr for alias in aliases):
            errors.append(f"家族 {family} 路次未齐，禁止删家族凑数")

    ok = not errors
    return ReconcileResult(ok=ok, errors=errors, warnings=warnings, stats=stats)
