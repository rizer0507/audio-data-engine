"""Export full-batch warehouse annotation packs (033 segment 1)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from audio_engine.core.annotation_v3.config import AnnotationConfig, default_annotation_config_path
from audio_engine.core.annotation_v3.types import VIEWS
from audio_engine.core.artifacts import atomic_write_json
from audio_engine.core.operator import ManifestOperator, OperatorConfig
from audio_engine.core.registry import register_operator
from audio_engine.core.sample import Sample
from audio_engine.core.source_naming import validate_source_name
from audio_engine.core.warehouse.export import export_warehouse_annotation_pack
from audio_engine.core.warehouse.identity import classified_snapshot_digest


@register_operator
class WarehouseExportAnnotationOperator(ManifestOperator):
    """Export whole classified Manifest as warehouse annotation pack(s).

    Params:
      batch: required business batch id
      revision: annotation revision (required)
      output: pack stem path (without requiring suffix)
      view: blind|candidate_check|...
      format: xlsx|jsonl|both
      annotation_config: configs/annotation/zh_asr_v3.yaml
      sample_ids: optional subset for sub-packs (freeze still needs full coverage)
      pack_index / pack_total: optional sub-pack metadata
      classified_manifest: path recorded into binding (default: pipeline input)
    """

    name = "warehouse_export_annotation"
    version = "1.0.0"
    category = "quality"

    def run(self, samples: list[Sample], config: OperatorConfig) -> list[Sample]:
        params = dict(config.params or {})
        batch = validate_source_name(str(params.get("batch") or "").strip())
        revision = str(params.get("revision") or "").strip()
        if not revision:
            raise ValueError("warehouse_export_annotation requires params.revision")

        out_raw = params.get("output") or params.get("pack_output")
        if not out_raw:
            out_raw = f"datasets/stage1/review/warehouse/{batch}/{revision}/pack"
        output = Path(str(out_raw))

        view = str(params.get("view") or "blind")
        if view not in VIEWS:
            raise ValueError(f"invalid view {view!r}; expected one of {sorted(VIEWS)}")
        fmt = str(params.get("format") or "both")
        ann_path = Path(
            str(params.get("annotation_config") or default_annotation_config_path())
        )
        ann_cfg = AnnotationConfig.load(ann_path)

        classified_manifest = str(params.get("classified_manifest") or "")
        sample_ids = params.get("sample_ids")
        if sample_ids is not None:
            sample_ids = [str(x) for x in sample_ids]

        digest = classified_snapshot_digest(samples)
        result = export_warehouse_annotation_pack(
            samples,
            config=ann_cfg,
            dataset_path=classified_manifest or f"classified:{batch}",
            output=output,
            batch=batch,
            revision=revision,
            view=view,
            fmt=fmt,
            sample_ids=sample_ids,
            pack_index=params.get("pack_index"),
            pack_total=params.get("pack_total"),
            classified_digest=digest,
        )

        run_dir = Path(config.run_dir) if config.run_dir else Path("runs") / "warehouse_export"
        run_dir.mkdir(parents=True, exist_ok=True)
        report: dict[str, Any] = {
            "operator": self.full_name,
            "version": self.version,
            "status": "ok",
            "export": result,
        }
        report_path = Path(str(params.get("report_output") or (run_dir / "warehouse_export_report.json")))
        atomic_write_json(report_path, report)

        stamped: list[Sample] = []
        for sample in samples:
            s = sample.model_copy(deep=True)
            s.labels["warehouse_batch"] = batch
            s.labels["warehouse_export_revision"] = revision
            s.labels["warehouse_export_queue_id"] = result["queue_id"]
            s.labels["warehouse_classified_digest"] = digest
            s.labels["warehouse_export_pack"] = str(output)
            s.labels["warehouse_export_report"] = str(report_path)
            s.mark_completed(self.full_name)
            s.add_lineage(
                self.full_name,
                self.version,
                {
                    "batch": batch,
                    "revision": revision,
                    "queue_id": result["queue_id"],
                    "sample_count": result["sample_count"],
                    "classified_digest": digest,
                },
            )
            stamped.append(s)
        return stamped
