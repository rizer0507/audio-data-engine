"""Compatibility alias: shared Chinese-only classify-text pre-layer.

044: implementation moved to ``audio_engine.core.evidence.classify_text``.
This module aliases the neutral implementation so historical imports (including
private helpers) keep working.
"""

import sys

from audio_engine.core.evidence import classify_text as _impl

sys.modules[__name__] = _impl
