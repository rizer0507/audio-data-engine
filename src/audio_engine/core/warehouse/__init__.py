"""Batch-unique data warehouse (033): annotation export + freeze.

Distinct from dataset_v3 Release (train/eval splits). One batch ↔ one formal warehouse.
"""

from __future__ import annotations

from audio_engine.core.warehouse.completion import (
    WarehouseGateError,
    assert_batch_complete,
    final_category,
    is_explicitly_excluded,
)
from audio_engine.core.warehouse.export import (
    build_warehouse_export_rows,
    export_warehouse_annotation_pack,
)
from audio_engine.core.warehouse.freeze import (
    WarehousePublishResult,
    publish_warehouse,
    warehouse_id_for_batch,
)
from audio_engine.core.warehouse.identity import (
    classified_snapshot_digest,
    original_audio_sha,
    resolve_audio_ref,
)
from audio_engine.core.warehouse.import_support import (
    apply_reviewed_category_from_rows,
    validate_warehouse_binding,
)
from audio_engine.core.warehouse.categories import load_allowed_categories, merge_category_params

__all__ = [
    "WarehouseGateError",
    "WarehousePublishResult",
    "apply_reviewed_category_from_rows",
    "assert_batch_complete",
    "build_warehouse_export_rows",
    "classified_snapshot_digest",
    "export_warehouse_annotation_pack",
    "final_category",
    "is_explicitly_excluded",
    "load_allowed_categories",
    "merge_category_params",
    "original_audio_sha",
    "publish_warehouse",
    "resolve_audio_ref",
    "validate_warehouse_binding",
    "warehouse_id_for_batch",
]
