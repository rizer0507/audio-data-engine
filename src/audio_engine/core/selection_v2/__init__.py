"""selection_v2.0 / consensus_v2 — modular transcript consensus engine.

Keeps ``selection_v1.1`` / ``consensus_v1`` untouched. Entry point:
``classify_sample`` in ``engine``.
"""

from audio_engine.core.selection_v2.engine import classify_sample
from audio_engine.core.selection_v2.config import SelectionV2Config
from audio_engine.core.selection_v2.result import ClassificationResultV2

__all__ = [
    "SelectionV2Config",
    "ClassificationResultV2",
    "classify_sample",
]
