"""Authoritative ASR backend interface and data contracts."""

from __future__ import annotations

from typing import Protocol, TypedDict

from .config import ASRConfig


class ASRSegmentRequired(TypedDict):
    """Required keys present in every validated ASR segment."""

    segment_id: str
    start: float | int
    end: float | int
    text: str


class ASRSegment(ASRSegmentRequired, total=False):
    """Validated ASR segment structure."""

    speaker: str


class ASRMeta(TypedDict):
    """ASR metadata structure."""

    backend: str
    model: str
    version: str
    device: str


class ASRResult(TypedDict):
    """Internal normalized ASR structure."""

    segments: list[ASRSegment]
    meta: ASRMeta


class ASRBackend(Protocol):
    """Contract for ASR backend implementations."""

    def transcribe(self, audio_path: str, config: ASRConfig) -> ASRResult:
        """Return ASR output in normalized internal JSON structure."""
        ...
