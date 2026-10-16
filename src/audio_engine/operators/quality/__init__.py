"""Import quality operators so OperatorRegistry is populated.

Modules are imported individually so a broken optional module (e.g. WIP tagging)
does not block registration of probe/filter/select/classify/warehouse.
"""

from __future__ import annotations

import importlib
import logging

_logger = logging.getLogger(__name__)

_MODULES = (
    "aggregate_manifests",
    "asr_anomaly_noise",
    "asr_edit_distance",
    "asr_route_v2_4",
    "attach_asr_v3",
    "audio_energy",
    "build_dataset_v3",
    "cer",
    "classify",
    "copy_transcripts",
    "dnsmos",
    "dnsmos_v2_2_candidates",
    "evidence_join",
    "efficientat",
    "efficientat_candidates",
    "evaluation_report",
    "filter",
    "inject_external_gold",
    "multi_tag",
    "normalize_transcripts",
    "on_demand_acoustic_candidates",
    "prepare_dataset_v3",
    "probe",
    "select",
    "silero_vad",
    "snr",
    "split_dataset",
    "text_metrics",
    "warehouse_export_annotation",
    "warehouse_freeze",
)

_failed: list[str] = []
for _name in _MODULES:
    try:
        importlib.import_module(f"audio_engine.operators.quality.{_name}")
    except Exception as exc:  # noqa: BLE001 — register what we can
        _failed.append(_name)
        _logger.warning("quality operator module %s failed to import: %s", _name, exc)

__all__: list[str] = []
