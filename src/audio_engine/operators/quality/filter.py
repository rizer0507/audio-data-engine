from __future__ import annotations

from typing import Any

import pandas as pd
from loguru import logger

from audio_engine.core.operator import BaseOperator, ManifestOperator, OperatorConfig
from audio_engine.core.registry import register_operator
from audio_engine.core.sample import Sample


@register_operator
class FilterOperator(ManifestOperator):
    """Apply a pandas query filter and mark samples as kept/dropped.

    Evaluated in chunks over a DataFrame instead of once per sample: a one-row
    query costs ~3.5 ms, which dominates a 100k+ run and cannot be parallelised
    because it is GIL-bound. Chunking keeps the row-wise semantics of the
    per-sample version while bounding peak memory.
    """

    name = "filter"
    version = "2.0.0"
    category = "quality"

    def run(self, samples: list[Sample], config: OperatorConfig) -> list[Sample]:
        expr = config.params.get("expr", "")
        label_key = config.params.get("label_key", "filter_pass")
        chunk_size = int(config.params.get("chunk_size", 20000))

        updated = [s.model_copy(deep=True) for s in samples]
        if not updated:
            return updated

        if not expr:
            for sample in updated:
                sample.labels[label_key] = True
                sample.mark_completed(self.full_name)
            return updated

        passed = 0
        for start in range(0, len(updated), chunk_size):
            chunk = updated[start : start + chunk_size]
            frame = pd.DataFrame([s.to_flat_dict() for s in chunk])
            try:
                kept_positions = set(frame.query(expr, engine="python").index)
                error = ""
            except Exception as exc:
                kept_positions = set()
                error = str(exc)
                logger.warning("quality.filter: expression failed on chunk: {}", error)

            for position, sample in enumerate(chunk):
                sample.labels[label_key] = position in kept_positions
                if error:
                    sample.labels[f"{label_key}_error"] = error
                sample.add_lineage(
                    operator=self.full_name,
                    version=self.version,
                    params={"expr": expr, "label_key": label_key},
                )
                sample.mark_completed(self.full_name)
            passed += len(kept_positions)

        logger.info(
            "quality.filter: {}/{} samples pass '{}' -> labels.{}",
            passed,
            len(updated),
            expr,
            label_key,
        )
        return updated


@register_operator
class TranscriptDiffOperator(BaseOperator):
    name = "transcript_diff"
    version = "1.0.0"
    category = "quality"

    def _execute(self, sample: Sample, config: OperatorConfig) -> dict[str, Any]:
        model_a = config.params.get("model_a", "asr.qwen")
        model_b = config.params.get("model_b", "asr.sensevoice")
        key_a = model_a.split(".")[-1] if "." in model_a else model_a
        key_b = model_b.split(".")[-1] if "." in model_b else model_b

        text_a = sample.get_transcript_text(key_a)
        text_b = sample.get_transcript_text(key_b)
        match = text_a == text_b

        return {
            "quality": {
                "transcript_match": match,
                f"{key_a}_text": text_a,
                f"{key_b}_text": text_b,
            },
            "labels": {"transcript_mismatch": not match},
            "lineage_entry": {
                "operator": self.full_name,
                "version": self.version,
                "params": {"model_a": key_a, "model_b": key_b},
            },
        }
