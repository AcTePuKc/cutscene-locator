"""Forced-alignment interfaces and validators."""

from .base import AlignmentBackend, AlignmentMeta, AlignmentResult, AlignmentSpan
from .validation import validate_alignment_result

__all__ = [
    "AlignmentBackend",
    "AlignmentMeta",
    "AlignmentResult",
    "AlignmentSpan",
    "validate_alignment_result",
]
