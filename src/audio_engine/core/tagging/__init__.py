"""Composable multi-tag rule engine (040)."""

from audio_engine.core.tagging.config import TaggingConfig, load_tagging_config
from audio_engine.core.tagging.engine import MultiTagEngine
from audio_engine.core.tagging.service import TaggingRunRequest, TaggingService
from audio_engine.core.tagging.types import POLICY_VERSION, Ternary

__all__ = [
    "POLICY_VERSION",
    "MultiTagEngine",
    "TaggingConfig",
    "TaggingRunRequest",
    "TaggingService",
    "Ternary",
    "load_tagging_config",
]
