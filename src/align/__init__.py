"""Forced-alignment interfaces and validators."""

from .base import AlignmentBackend, AlignmentMeta, AlignmentResult, AlignmentSpan, ReferenceSpan
from .qwen3_forced_aligner import Qwen3ForcedAligner
from .reference_spans import script_table_to_reference_spans
from .validation import validate_alignment_result

__all__ = [
    "AlignmentBackend",
    "AlignmentMeta",
    "AlignmentResult",
    "AlignmentSpan",
    "ReferenceSpan",
    "Qwen3ForcedAligner",
    "script_table_to_reference_spans",
    "validate_alignment_result",
]
