"""selection_v3.0 — contract, quality consumption, multi-family (≥3) classification.

Stage A: family config, 2N-route status, input alignment.
Stage B: DNSMOS consumption, family evidence, semantic risk, consensus, classifier.
"""

from audio_engine.core.selection_v3.classifier import classify_sample
from audio_engine.core.selection_v3.config import RunIdentity, SelectionV3Config
from audio_engine.core.selection_v3.input_contract import (
    ConservationReport,
    EightRouteAlignmentReport,
    SampleContractResult,
    apply_contract_to_samples,
    classify_run_status,
    evaluate_sample_contract,
    join_key,
    merge_field_by_join_key,
    original_audio_sha256,
)
from audio_engine.core.selection_v3.result import ClassificationResultV3

__all__ = [
    "RunIdentity",
    "SelectionV3Config",
    "ClassificationResultV3",
    "ConservationReport",
    "EightRouteAlignmentReport",
    "SampleContractResult",
    "apply_contract_to_samples",
    "classify_run_status",
    "classify_sample",
    "evaluate_sample_contract",
    "join_key",
    "merge_field_by_join_key",
    "original_audio_sha256",
]
