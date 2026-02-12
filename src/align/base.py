"""Alignment backend interface and data contracts."""

from __future__ import annotations

from typing import Protocol, TypedDict


class AlignmentSpan(TypedDict):
    """Required keys present in every validated alignment span."""

    span_id: str
    start: float | int
    end: float | int
    text: str
    confidence: float | int


class ReferenceSpan(TypedDict):
    """Known text span supplied by caller for forced alignment."""

    ref_id: str
    text: str


class AlignmentMeta(TypedDict):
    """Alignment metadata structure."""

    backend: str
    version: str
    device: str


class AlignmentResult(TypedDict):
    """Internal normalized alignment structure."""

    transcript_text: str
    spans: list[AlignmentSpan]
    meta: AlignmentMeta


class AlignmentBackend(Protocol):
    """Contract for forced-alignment backend implementations."""

    def align(self, audio_path: str, reference_spans: list[ReferenceSpan]) -> AlignmentResult:
        """Return aligned spans for caller-provided known text against audio input."""
        ...
