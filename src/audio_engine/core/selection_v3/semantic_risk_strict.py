"""Compatibility alias: strict semantic-risk predicates.

044: implementation moved to ``audio_engine.core.evidence.semantic_strict``.
This module aliases the neutral implementation so historical imports (including
private helpers) keep working.
"""

import sys

from audio_engine.core.evidence import semantic_strict as _impl

sys.modules[__name__] = _impl
