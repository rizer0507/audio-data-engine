from __future__ import annotations

import pandas as pd
from loguru import logger

from audio_engine.core.operator import ManifestOperator, OperatorConfig
from audio_engine.core.registry import register_operator
from audio_engine.core.sample import Sample


@register_operator
class SelectOperator(ManifestOperator):
    """Keep only samples matching a pandas query expression.

    Used as the cleaning export step: drop samples that failed audio_pass.
    """

    name = "select"
    version = "1.0.0"
    category = "quality"

    def run(self, samples: list[Sample], config: OperatorConfig) -> list[Sample]:
        expr = config.params.get("expr", "")
        if not expr:
            return list(samples)

        rows = [s.to_flat_dict() for s in samples]
        if not rows:
            return []

        df = pd.DataFrame(rows)
        try:
            matched = df.query(expr, engine="python")
        except Exception as exc:
            raise ValueError(f"Invalid select expression: {expr!r} ({exc})") from exc

        keep_ids = set(matched["id"].tolist())
        kept = [s for s in samples if s.id in keep_ids]
        for sample in kept:
            sample.mark_completed(self.full_name)
            sample.add_lineage(
                operator=self.full_name,
                version=self.version,
                params={"expr": expr},
            )

        logger.info(
            "quality.select: kept {}/{} samples ({})",
            len(kept),
            len(samples),
            expr,
        )
        return kept
