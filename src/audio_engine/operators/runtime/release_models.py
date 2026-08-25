"""Release cached ASR models between serial multi-model pipeline stages."""

from __future__ import annotations

import gc

from loguru import logger

from audio_engine.core.operator import ManifestOperator, OperatorConfig
from audio_engine.core.registry import register_operator
from audio_engine.core.sample import Sample


@register_operator
class ReleaseAsrModelsOperator(ManifestOperator):
    """Clear in-process ASR model caches and free CUDA memory.

    Used between serial ASR steps in one pipeline so Qwen and SenseVoice
    (or future models) do not occupy GPU memory at the same time.
    """

    name = "release_asr_models"
    version = "1.0.0"
    category = "runtime"

    def run(self, samples: list[Sample], config: OperatorConfig) -> list[Sample]:
        released: dict[str, int] = {}
        try:
            from audio_engine.operators.asr import qwen as qwen_mod

            released["qwen"] = int(qwen_mod.release_cached_models())
        except Exception as exc:  # noqa: BLE001 - best-effort cleanup
            logger.warning("release qwen cache failed: {}", exc)
            released["qwen"] = -1

        try:
            from audio_engine.operators.asr import sensevoice as sv_mod

            released["sensevoice"] = int(sv_mod.release_cached_models())
        except Exception as exc:  # noqa: BLE001 - best-effort cleanup
            logger.warning("release sensevoice cache failed: {}", exc)
            released["sensevoice"] = -1

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception as exc:  # noqa: BLE001 - optional CUDA cleanup
            logger.warning("cuda empty_cache failed: {}", exc)

        logger.info(
            "Step '{}': released ASR model caches {} (samples untouched={})",
            config.step_name or self.full_name,
            released,
            len(samples),
        )
        return samples
