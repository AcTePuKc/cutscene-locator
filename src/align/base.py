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

    def align(self, audio_path: str, transcript_text: str) -> AlignmentResult:
        """Return aligned spans for known transcript text against audio input."""
        ...
