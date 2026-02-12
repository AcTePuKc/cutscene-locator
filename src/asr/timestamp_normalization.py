"""Deterministic ASR timestamp normalization helpers."""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from typing import Any

from .base import ASRSegment

_TIMESTAMP_SCALE_DECIMALS = 6
_TIMESTAMP_QUANTIZER = Decimal("1").scaleb(-_TIMESTAMP_SCALE_DECIMALS)


class TimestampNormalizationError(ValueError):
    """Raised when backend timestamp output cannot be normalized safely."""


def _as_decimal_seconds(value: Any, *, path: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TimestampNormalizationError(f"{path} must be numeric seconds")

    decimal_value = Decimal(str(value))
    if decimal_value.is_nan() or decimal_value.is_infinite():
        raise TimestampNormalizationError(f"{path} must be finite numeric seconds")
    if decimal_value < 0:
        raise TimestampNormalizationError(f"{path} must be non-negative")

    return decimal_value.quantize(_TIMESTAMP_QUANTIZER, rounding=ROUND_HALF_UP)


def normalize_asr_segments_for_contract(
    segments: list[ASRSegment],
    *,
    source: str,
) -> list[ASRSegment]:
    """Normalize backend segments deterministically for contract validation.

    Rules:
    - timestamps are normalized as numeric seconds with deterministic half-up rounding
    - negative, NaN, and infinite timestamps are rejected
    - end must be greater than or equal to start; zero-length entries are dropped
    - final ordering is stable and deterministic by (start, end, original_index)
    - text payload is preserved exactly as provided by backend output
    """

    normalized: list[tuple[Decimal, Decimal, int, ASRSegment]] = []

    for index, segment in enumerate(segments):
        segment_path = f"{source}: segments[{index}]"

        segment_id = segment.get("segment_id")
        text = segment.get("text")
        if not isinstance(segment_id, str) or not segment_id.strip():
            raise TimestampNormalizationError(f"{segment_path}.segment_id must be a non-empty string")
        if not isinstance(text, str) or not text.strip():
            raise TimestampNormalizationError(f"{segment_path}.text must be a non-empty string")

        start = _as_decimal_seconds(segment.get("start"), path=f"{segment_path}.start")
        end = _as_decimal_seconds(segment.get("end"), path=f"{segment_path}.end")

        if end < start:
            raise TimestampNormalizationError(
                f"{segment_path} (segment_id={segment_id}): end must be greater than or equal to start"
            )
        if end == start:
            continue

        normalized_segment: ASRSegment = {
            "segment_id": segment_id,
            "start": float(start),
            "end": float(end),
            "text": text,
        }

        speaker = segment.get("speaker")
        if speaker is not None:
            if not isinstance(speaker, str) or not speaker.strip():
                raise TimestampNormalizationError(f"{segment_path}.speaker must be a non-empty string")
            normalized_segment["speaker"] = speaker

        normalized.append((start, end, index, normalized_segment))

    normalized.sort(key=lambda item: (item[0], item[1], item[2]))
    return [item[3] for item in normalized]
