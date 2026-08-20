"""Import all operators to register them with OperatorRegistry."""

from audio_engine.operators import asr, audio, augmentation, quality

__all__ = ["asr", "audio", "augmentation", "quality"]
