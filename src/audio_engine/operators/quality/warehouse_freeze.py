"""Freeze batch-unique warehouse after human review (033 segment 2)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from audio_engine.core.annotation_v3.config import AnnotationConfig, default_annotation_config_path
from audio_engine.core.artifacts import atomic_write_json
from audio_engine.core.manifest import Manifest
from audio_engine.core.operator import ManifestOperator, OperatorConfig
from audio_engine.core.registry import register_operator
from audio_engine.core.sample import Sample
from audio_engine.core.source_naming import validate_source_name
from audio_engine.core.warehouse.categories import merge_category_params
from audio_engine.core.warehouse.completion import WarehouseGateError
from audio_engine.core.warehouse.freeze import publish_warehouse


@register_operator
class WarehouseFreezeOperator(ManifestOperator):
    """Manifest-layer freeze: align ↔ complete ↔ atomic warehouse publish.

    Params:
      batch: required
      classified_manifest: required path to classified snapshot
      annotation_config: annotation policy
      categories_config / allowed_categories: extensible category allow-list
      warehouse_output_dir: default datasets/stage1/warehouses
      catalog_dir: default data/catalog
      review_evidence: list of annotation artifact paths that must remain
      rule_version: optional override recorded in warehouse.json
    """

    name = "warehouse_freeze"
    version = "1.0.0"
    category = "quality"

    def run(self, samples: list[Sample], config: OperatorConfig) -> list[Sample]:
        params = dict(config.params or {})
        batch = validate_source_name(str(params.get("batch") or "").strip())
        classified_path = params.get("classified_manifest")
        if not classified_path:
            raise ValueError("warehouse_freeze requires params.classified_manifest")
        classified_path = Path(str(classified_path))
        if not classified_path.is_file():
            raise FileNotFoundError(f"classified_manifest not found: {classified_path}")

        ann_path = Path(
            str(params.get("annotation_config") or default_annotation_config_path())
        )
        ann_cfg = AnnotationConfig.load(ann_path)
        allowed = merge_category_params(params)

        evidence = params.get("review_evidence") or []
        if isinstance(evidence, (str, Path)):
            evidence = [evidence]

        run_dir = Path(config.run_dir) if config.run_dir else Path("runs") / "warehouse_freeze"
        run_dir.mkdir(parents=True, exist_ok=True)

        classified = list(Manifest.load(classified_path))
        try:
            result = publish_warehouse(
                samples,
                batch=batch,
                classified=classified,
                classified_path=classified_path,
                config=ann_cfg,
                allowed_categories=allowed,
                catalog_dir=params.get("catalog_dir") or "data/catalog",
                output_dir=params.get("warehouse_output_dir")
                or params.get("output_dir")
                or "datasets/stage1/warehouses",
                run_dir=run_dir,
                review_evidence=[str(x) for x in evidence],
                rule_version=params.get("rule_version"),
                annotation_config_path=str(ann_path),
            )
        except WarehouseGateError:
            report = {
                "operator": self.full_name,
                "version": self.version,
                "status": "failed",
                "batch": batch,
            }
            atomic_write_json(run_dir / "warehouse_freeze_failed.json", report)
            for sample in samples:
                sample.mark_failed(self.full_name, "warehouse_freeze_refused")
            raise

        report: dict[str, Any] = {
            "operator": self.full_name,
            "version": self.version,
            "status": "ok",
            "warehouse": result.to_dict(),
        }
        report_path = Path(
            str(params.get("report_output") or (run_dir / "warehouse_freeze_report.json"))
        )
        atomic_write_json(report_path, report)

        stamped: list[Sample] = []
        for sample in samples:
            s = sample.model_copy(deep=True)
            s.labels["warehouse_id"] = result.warehouse_id
            s.labels["warehouse_batch"] = batch
            s.labels["warehouse_dir"] = str(result.warehouse_dir)
            s.labels["warehouse_content_fingerprint"] = result.content_fingerprint
            s.labels["warehouse_freeze_report"] = str(report_path)
            s.mark_completed(self.full_name)
            s.add_lineage(
                self.full_name,
                self.version,
                {
                    "warehouse_id": result.warehouse_id,
                    "batch": batch,
                    "idempotent_hit": result.idempotent_hit,
                    "content_fingerprint": result.content_fingerprint,
                },
            )
            stamped.append(s)
        return stamped
