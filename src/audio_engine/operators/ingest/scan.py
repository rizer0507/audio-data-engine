from __future__ import annotations

import shutil
from pathlib import Path

from loguru import logger

from audio_engine.core.manifest import Manifest
from audio_engine.core.operator import ManifestOperator, OperatorConfig
from audio_engine.core.registry import register_operator
from audio_engine.core.sample import Sample


@register_operator
class ScanIngestOperator(ManifestOperator):
    """Scan an external directory (read-only) and append new samples.

    Params:
        source_dir: directory containing raw audio files (required)
        recursive:  scan subdirectories (default True)
        copy_to:    optional directory to copy files into (e.g. data/raw)
    """

    name = "scan"
    version = "1.0.0"
    category = "ingest"

    def run(self, samples: list[Sample], config: OperatorConfig) -> list[Sample]:
        source_dir = config.params.get("source_dir")
        if not source_dir:
            raise ValueError("ingest.scan requires a 'source_dir' param (or input.source_dir in the pipeline YAML)")

        recursive = bool(config.params.get("recursive", True))
        scanned = Manifest.ingest(source_dir, recursive=recursive)

        copy_to = config.params.get("copy_to")
        if copy_to:
            copy_dir = Path(copy_to)
            copy_dir.mkdir(parents=True, exist_ok=True)
            for sample in scanned.samples:
                src = Path(sample.source_path)
                dst = copy_dir / src.name
                if not dst.exists():
                    shutil.copy2(src, dst)
                sample.source_path = str(dst.resolve())
                sample.audio["raw"] = str(dst.resolve())

        existing = {s.sha256 for s in samples if s.sha256}
        new_samples: list[Sample] = []
        for sample in scanned.samples:
            if sample.sha256 in existing:
                continue
            sample.add_lineage(
                operator=self.full_name,
                version=self.version,
                params={"source_dir": str(source_dir)},
                output_key="raw",
                output_path=sample.source_path,
            )
            sample.mark_completed(self.full_name)
            new_samples.append(sample)

        logger.info(
            "ingest.scan: {} files scanned, {} new samples added",
            len(scanned.samples),
            len(new_samples),
        )
        return list(samples) + new_samples
