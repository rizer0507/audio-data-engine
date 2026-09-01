from __future__ import annotations

import hashlib
from collections import Counter, defaultdict

from audio_engine.core.operator import ManifestOperator, OperatorConfig
from audio_engine.core.registry import register_operator
from audio_engine.core.sample import Sample


@register_operator
class SplitDatasetOperator(ManifestOperator):
    """Deterministic group-aware split that prevents a group crossing data subsets."""

    name = "split_dataset"
    version = "1.0.0"
    category = "quality"

    def run(self, samples: list[Sample], config: OperatorConfig) -> list[Sample]:
        group_key = str(config.params.get("group_key", "speaker_id"))
        stratify_key = config.params.get("stratify_key")
        seed = int(config.params.get("seed", 0))
        ratios = config.params.get("ratios") or {"train": 0.8, "dev": 0.1, "test": 0.1}
        if not ratios or any(float(value) < 0 for value in ratios.values()):
            raise ValueError("split ratios must be non-negative")
        total = sum(float(value) for value in ratios.values())
        if total <= 0:
            raise ValueError("split ratios must have a positive total")
        ordered = list(ratios)
        cumulative: list[tuple[str, float]] = []
        cursor = 0.0
        for name in ordered:
            cursor += float(ratios[name]) / total
            cumulative.append((name, cursor))

        group_strata: dict[str, Counter[str]] = defaultdict(Counter)
        for sample in samples:
            raw_group = sample.labels.get(group_key)
            if raw_group is None:
                raise ValueError(f"sample {sample.id} missing split group label: {group_key}")
            stratum = str(sample.labels.get(stratify_key, "all")) if stratify_key else "all"
            group_strata[str(raw_group)][stratum] += 1
        primary_stratum = {
            group: sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
            for group, counts in group_strata.items()
        }

        updated = []
        group_splits: dict[str, str] = {}
        for source in samples:
            sample = source.model_copy(deep=True)
            raw_group = sample.labels.get(group_key)
            if raw_group is None:
                raise ValueError(f"sample {sample.id} missing split group label: {group_key}")
            group = str(raw_group)
            if group not in group_splits:
                digest = hashlib.sha256(
                    f"{seed}\0{primary_stratum[group]}\0{group}".encode()
                ).digest()
                point = int.from_bytes(digest[:8], "big") / 2**64
                group_splits[group] = next(
                    name for name, boundary in cumulative if point < boundary
                )
            split = group_splits[group]
            sample.labels.update(
                {
                    "split": split,
                    "split_seed": seed,
                    "split_group_key": group_key,
                    "split_stratify_key": stratify_key,
                }
            )
            sample.mark_completed(self.full_name)
            sample.add_lineage(
                self.full_name,
                self.version,
                {
                    "group_key": group_key,
                    "stratify_key": stratify_key,
                    "seed": seed,
                    "ratios": ratios,
                },
            )
            updated.append(sample)
        return updated
