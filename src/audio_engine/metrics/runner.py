from __future__ import annotations

from typing import Any

from audio_engine.metrics.cer import calculate_cer
from audio_engine.metrics.normalization import normalize_text


class MetricConfigError(ValueError):
    pass


def run_text_metrics(
    record: dict[str, Any], comparison: dict[str, Any], normalization: dict[str, Any]
) -> dict[str, Any]:
    """Run one configured comparison against a flat record."""
    ref_field = comparison.get("reference", {}).get("field")
    hyp_field = comparison.get("hypothesis", {}).get("field")
    for role, field in (("reference", ref_field), ("hypothesis", hyp_field)):
        if not field or field not in record:
            raise MetricConfigError(f"{role} field `{field}` not found in dataset")
    metrics = comparison.get("metrics", ["cer"])
    unsupported = set(metrics) - {"cer"}
    if unsupported:
        raise MetricConfigError(f"unsupported metrics: {sorted(unsupported)}")
    prefix = comparison.get("output", {}).get("prefix")
    if not prefix:
        raise MetricConfigError("comparison.output.prefix is required")
    output = calculate_cer(
        normalize_text(record[ref_field], normalization),
        normalize_text(record[hyp_field], normalization),
    )
    result = {f"{prefix}_cer": output.pop("cer")}
    result.update({f"{prefix}_{key}": value for key, value in output.items()})
    collisions = result.keys() & record.keys()
    if collisions and not comparison.get("overwrite", False):
        raise MetricConfigError(f"output fields already exist: {sorted(collisions)}")
    return result
