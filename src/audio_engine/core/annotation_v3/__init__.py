"""annotation_v3 — human review contract, blind packs, dual-review import."""

from __future__ import annotations

from audio_engine.core.annotation_v3.config import AnnotationConfig, default_annotation_config_path
from audio_engine.core.annotation_v3.contract import (
    AnnotationDraft,
    GoldTextValue,
    decode_gold_text_from_tabular,
    encode_gold_text_for_tabular,
    may_pass_formal_gold,
    validate_annotation_draft,
)
from audio_engine.core.annotation_v3.import_workflow import (
    ImportResult,
    apply_review_import_v3,
    import_has_blocking_issues,
    load_review_rows,
)
from audio_engine.core.annotation_v3.package import build_export_rows, write_review_package
from audio_engine.core.annotation_v3.queue import (
    queue_id_v3,
    requires_dual_review,
    select_review_batch,
)
from audio_engine.core.annotation_v3.types import ANNOTATION_VERSION

__all__ = [
    "ANNOTATION_VERSION",
    "AnnotationConfig",
    "AnnotationDraft",
    "GoldTextValue",
    "ImportResult",
    "apply_review_import_v3",
    "build_export_rows",
    "decode_gold_text_from_tabular",
    "default_annotation_config_path",
    "encode_gold_text_for_tabular",
    "import_has_blocking_issues",
    "load_review_rows",
    "may_pass_formal_gold",
    "queue_id_v3",
    "requires_dual_review",
    "select_review_batch",
    "validate_annotation_draft",
    "write_review_package",
]
