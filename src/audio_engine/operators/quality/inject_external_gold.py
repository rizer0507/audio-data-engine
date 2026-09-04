"""Inject external gold labels from xlsx into a Manifest (by sample id)."""

from __future__ import annotations

import os
from pathlib import Path

from audio_engine.core.artifacts import atomic_write_json
from audio_engine.core.operator import ManifestOperator, OperatorConfig
from audio_engine.core.registry import register_operator
from audio_engine.core.sample import Sample
from audio_engine.operators.quality.external_gold import load_external_gold_table


@register_operator
class InjectExternalGoldOperator(ManifestOperator):
    """Join external xlsx gold onto samples; does not invent consensus gold.

    Params:
      - xlsx_path / EXTERNAL_GOLD_XLSX: path to gold workbook
      - id_col / label_col / type_col: column overrides (auto-detect if omitted)
      - align_policy: ``intersection`` (default) | ``left``
      - missing_gold: ``fail`` (default for left) | ``allow_empty``
    """

    name = "inject_external_gold"
    version = "1.0.0"
    category = "quality"

    def run(self, samples: list[Sample], config: OperatorConfig) -> list[Sample]:
        params = dict(config.params)
        xlsx = str(params.get("xlsx_path") or os.environ.get("EXTERNAL_GOLD_XLSX") or "").strip()
        if not xlsx:
            raise ValueError(
                "quality.inject_external_gold requires params.xlsx_path "
                "or env EXTERNAL_GOLD_XLSX (pass --external-gold on CLI)"
            )
        align_policy = str(params.get("align_policy") or "intersection").strip().lower()
        if align_policy not in {"intersection", "left"}:
            raise ValueError("align_policy must be intersection|left")
        missing_gold = str(params.get("missing_gold") or "fail").strip().lower()
        if missing_gold not in {"fail", "allow_empty"}:
            raise ValueError("missing_gold must be fail|allow_empty")

        table, load_counts = load_external_gold_table(
            xlsx,
            id_col=params.get("id_col"),
            label_col=params.get("label_col"),
            type_col=params.get("type_col"),
        )
        if not table:
            raise ValueError(f"external gold xlsx produced no rows: {xlsx}")

        indexed = {sample.id: sample.model_copy(deep=True) for sample in samples}
        if len(indexed) != len(samples):
            raise ValueError("inject_external_gold input contains duplicate ids")

        report = {
            "xlsx_path": str(Path(xlsx).resolve()),
            "align_policy": align_policy,
            "missing_gold": missing_gold,
            "load": load_counts,
            "manifest_count": len(samples),
            "matched": 0,
            "missing_in_xlsx": [],
            "missing_in_manifest": sorted(set(table) - set(indexed))[:50],
            "missing_in_manifest_count": len(set(table) - set(indexed)),
        }

        selected_ids: list[str]
        if align_policy == "intersection":
            selected_ids = [sample.id for sample in samples if sample.id in table]
            if not selected_ids:
                raise ValueError(
                    "inject_external_gold: no overlapping ids between manifest and xlsx"
                )
        else:
            selected_ids = [sample.id for sample in samples]
            missing = [sample_id for sample_id in selected_ids if sample_id not in table]
            report["missing_in_xlsx"] = missing[:50]
            report["missing_in_xlsx_count"] = len(missing)
            if missing and missing_gold == "fail":
                raise ValueError(
                    f"inject_external_gold: {len(missing)} manifest ids missing from xlsx "
                    f"(examples={missing[:5]})"
                )

        updated: list[Sample] = []
        for sample_id in selected_ids:
            sample = indexed[sample_id]
            entry = table.get(sample_id)
            if entry is None:
                sample.labels.setdefault("gold_text", "")
                sample.labels.setdefault("label", "")
                sample.labels["gold_mode"] = "external"
                sample.labels["gold_source"] = "external"
            else:
                gold = entry["gold_text"]
                sample.labels["gold_text"] = gold
                sample.labels["label"] = gold
                sample.labels["label_text_raw"] = entry["label_text_raw"]
                sample.labels["gold_mode"] = "external"
                sample.labels["gold_source"] = "external"
                if entry.get("type"):
                    sample.labels["external_type"] = entry["type"]
                report["matched"] += 1
            sample.mark_completed(self.full_name)
            sample.add_lineage(
                self.full_name,
                self.version,
                {"xlsx_path": str(Path(xlsx).resolve()), "align_policy": align_policy},
            )
            updated.append(sample)

        if config.run_dir is not None:
            report_path = Path(config.run_dir) / "reports" / f"{config.step_name}_inject.json"
            atomic_write_json(report_path, report)
        return updated
