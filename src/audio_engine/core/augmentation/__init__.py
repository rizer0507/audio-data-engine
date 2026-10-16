"""Pitch-preserving tempo augmentation (047)."""

from audio_engine.core.augmentation.params import (
    DEFAULT_SPEED_FACTOR,
    TempoParams,
    factor_tag,
    normalize_speed_factor,
    resolve_tempo_params,
)
from audio_engine.core.augmentation.publish import AugmentBatchIncomplete
from audio_engine.core.augmentation.recipe import (
    STAGE1_AUGMENTED_DIR,
    build_recipe,
    product_root_for,
    recipe_digest,
)

__all__ = [
    "AugmentBatchIncomplete",
    "DEFAULT_SPEED_FACTOR",
    "STAGE1_AUGMENTED_DIR",
    "TempoParams",
    "build_recipe",
    "factor_tag",
    "normalize_speed_factor",
    "product_root_for",
    "recipe_digest",
    "resolve_tempo_params",
]
