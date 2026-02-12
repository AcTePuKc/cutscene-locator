"""Validation helpers for forced-alignment backend output."""

from __future__ import annotations

from typing import Any

from .base import AlignmentResult, AlignmentSpan


def _require_non_empty_string(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value


def _require_numeric(value: Any, *, path: str) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be numeric")
    return value


def validate_alignment_result(raw_data: Any, *, source: str = "alignment data") -> AlignmentResult:
    """Validate and return forced-alignment JSON matching the internal contract."""

    if not isinstance(raw_data, dict):
        raise ValueError(f"{source}: root must be an object")

    transcript_text = _require_non_empty_string(
        raw_data.get("transcript_text"), path=f"{source}: transcript_text"
    )

    spans_raw = raw_data.get("spans")
    meta_raw = raw_data.get("meta")

    if not isinstance(spans_raw, list):
        raise ValueError(f"{source}: 'spans' must be a list")
    if not isinstance(meta_raw, dict):
        raise ValueError(f"{source}: 'meta' must be an object")

    backend = _require_non_empty_string(meta_raw.get("backend"), path=f"{source}: meta.backend")
    version = _require_non_empty_string(meta_raw.get("version"), path=f"{source}: meta.version")
    device = _require_non_empty_string(meta_raw.get("device"), path=f"{source}: meta.device")

    validated_spans: list[AlignmentSpan] = []
    for idx, span_raw in enumerate(spans_raw):
        span_path = f"{source}: spans[{idx}]"
        if not isinstance(span_raw, dict):
            raise ValueError(f"{span_path} must be an object")

        span_id = _require_non_empty_string(span_raw.get("span_id"), path=f"{span_path}.span_id")
        start = _require_numeric(span_raw.get("start"), path=f"{span_path}.start")
        end = _require_numeric(span_raw.get("end"), path=f"{span_path}.end")
        if start >= end:
            raise ValueError(f"{span_path} (span_id={span_id}): start must be less than end")

        text = _require_non_empty_string(span_raw.get("text"), path=f"{span_path}.text")
        confidence = _require_numeric(span_raw.get("confidence"), path=f"{span_path}.confidence")
        if confidence < 0 or confidence > 1:
            raise ValueError(f"{span_path}.confidence must be within [0, 1]")

        validated_spans.append(
            {
                "span_id": span_id,
                "start": start,
                "end": end,
                "text": text,
                "confidence": confidence,
            }
        )

    return {
        "transcript_text": transcript_text,
        "spans": validated_spans,
        "meta": {
            "backend": backend,
            "version": version,
            "device": device,
        },
    }
