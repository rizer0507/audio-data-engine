"""Compatibility alias: versioned text layers and safe tolerance.

044: implementation moved to ``audio_engine.core.evidence.text_tolerance``.
This module aliases the neutral implementation so historical imports (including
private helpers) keep working.
"""

import sys

from audio_engine.core.evidence import text_tolerance as _impl

sys.modules[__name__] = _impl
