"""ASR backend implementations and validation helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import ASRResult, ASRSegment
from .config import ASRConfig
from .device import resolve_device


@dataclass(frozen=True)
class MockASRBackend:
    """Load ASR segments from a local JSON file for deterministic testing."""

    mock_json_path: Path

    def transcribe(self, audio_path: str, config: ASRConfig) -> ASRResult:
        del audio_path  # Unused by mock backend.
        if not self.mock_json_path.exists():
            raise ValueError(f"Mock ASR file does not exist: {self.mock_json_path}")

        try:
            raw_data = json.loads(self.mock_json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Mock ASR JSON parse error in '{self.mock_json_path}': {exc.msg}") from exc

        if not isinstance(raw_data, dict):
            raise ValueError(f"{self.mock_json_path}: root must be an object")

        raw_meta = raw_data.get("meta") if isinstance(raw_data.get("meta"), dict) else {}
        resolved_device = resolve_device(config.device)
        resolved_model = "unknown"
        if config.model_path is not None:
            resolved_model = config.model_path.name or str(config.model_path)
        elif isinstance(raw_meta.get("model"), str) and raw_meta["model"].strip():
            resolved_model = raw_meta["model"].strip()

        resolved_version = "unknown"
        if isinstance(raw_meta.get("version"), str) and raw_meta["version"].strip():
            resolved_version = raw_meta["version"].strip()

        normalized: dict[str, Any] = {
            "segments": raw_data.get("segments"),
            "meta": {
                "backend": "mock",
                "model": resolved_model,
                "version": resolved_version,
                "device": resolved_device,
            },
        }
        return validate_asr_result(normalized, source=str(self.mock_json_path))


def _require_non_empty_string(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value


def _require_numeric_timestamp(value: Any, *, path: str) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be numeric")
    return value


def validate_asr_result(raw_data: Any, *, source: str = "ASR data") -> ASRResult:
    """Validate and return ASR JSON that matches the internal data contract."""

    if not isinstance(raw_data, dict):
        raise ValueError(f"{source}: root must be an object")

    segments_raw = raw_data.get("segments")
    meta_raw = raw_data.get("meta")

    if not isinstance(segments_raw, list):
        raise ValueError(f"{source}: 'segments' must be a list")
    if not isinstance(meta_raw, dict):
        raise ValueError(f"{source}: 'meta' must be an object")

    backend = _require_non_empty_string(meta_raw.get("backend"), path=f"{source}: meta.backend")
    model = _require_non_empty_string(meta_raw.get("model"), path=f"{source}: meta.model")
    version = _require_non_empty_string(meta_raw.get("version"), path=f"{source}: meta.version")
    device = _require_non_empty_string(meta_raw.get("device"), path=f"{source}: meta.device")

    validated_segments: list[ASRSegment] = []
    for idx, segment_raw in enumerate(segments_raw):
        segment_path = f"{source}: segments[{idx}]"
        if not isinstance(segment_raw, dict):
            raise ValueError(f"{segment_path} must be an object")

        segment_id = _require_non_empty_string(segment_raw.get("segment_id"), path=f"{segment_path}.segment_id")
        start = _require_numeric_timestamp(segment_raw.get("start"), path=f"{segment_path}.start")
        end = _require_numeric_timestamp(segment_raw.get("end"), path=f"{segment_path}.end")

        if start >= end:
            raise ValueError(
                f"{segment_path} (segment_id={segment_id}): start must be less than end"
            )

        text = _require_non_empty_string(segment_raw.get("text"), path=f"{segment_path}.text")

        validated_segment: ASRSegment = {
            "segment_id": segment_id,
            "start": start,
            "end": end,
            "text": text,
        }

        speaker = segment_raw.get("speaker")
        if speaker is not None:
            validated_segment["speaker"] = _require_non_empty_string(
                speaker,
                path=f"{segment_path}.speaker",
            )

        validated_segments.append(validated_segment)

    return {
        "segments": validated_segments,
        "meta": {
            "backend": backend,
            "model": model,
            "version": version,
            "device": device,
        },
    }
