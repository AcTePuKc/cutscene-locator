"""VibeVoice ASR backend implementation."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from .backends import validate_asr_result
from .base import ASRResult
from .config import ASRConfig
from .device import resolve_device_with_details
from .timestamp_normalization import normalize_asr_segments_for_contract


class VibeVoiceBackend:
    """ASR backend powered by VibeVoice runtime."""

    def transcribe(self, audio_path: str, config: ASRConfig) -> ASRResult:
        if config.model_path is None:
            raise ValueError(
                "vibevoice backend requires a resolved model path. "
                "Provide --model-path or --model-id."
            )

        resolution = resolve_device_with_details(config.device)
        resolved_device = resolution.resolved

        try:
            vibevoice_module = import_module("vibevoice")
            torch_module = import_module("torch")
        except ModuleNotFoundError as exc:
            raise ValueError(
                "vibevoice backend requires optional dependencies. "
                "Install them with: pip install 'cutscene-locator[asr_vibevoice]'"
            ) from exc

        if resolved_device == "cuda" and not bool(torch_module.cuda.is_available()):
            raise ValueError(
                "Requested --device cuda, but CUDA is not available in this environment. "
                "Use --device cpu or follow docs/CUDA.md."
            )

        transcribe_file = getattr(vibevoice_module, "transcribe_file", None)
        if transcribe_file is None:
            raise ValueError(
                "Installed vibevoice package is missing required transcribe_file() API."
            )

        try:
            raw_result = transcribe_file(
                audio_path=audio_path,
                model_path=str(config.model_path),
                device=resolved_device,
                language=config.language,
            )
        except Exception as exc:  # pragma: no cover - runtime specific
            raise ValueError(
                f"vibevoice transcription failed for '{audio_path}'. "
                "Verify input audio and model compatibility."
            ) from exc

        segments = normalize_asr_segments_for_contract(
            _normalize_vibevoice_segments(raw_result),
            source="vibevoice",
        )

        backend_version = "unknown"
        try:
            backend_version = version("vibevoice")
        except PackageNotFoundError:
            backend_version = "unknown"

        model_name = Path(config.model_path).name or str(config.model_path)
        return validate_asr_result(
            {
                "segments": segments,
                "meta": {
                    "backend": "vibevoice",
                    "model": model_name,
                    "version": backend_version,
                    "device": resolved_device,
                },
            },
            source="vibevoice",
        )


def _normalize_vibevoice_segments(raw_result: Any) -> list[dict[str, float | str]]:
    segments = raw_result.get("segments") if isinstance(raw_result, dict) else None
    if not isinstance(segments, list) or not segments:
        raise ValueError(
            "vibevoice backend did not return timestamped segments. "
            "Expected a non-empty 'segments' list."
        )

    normalized_segments: list[dict[str, float | str]] = []
    for index, segment in enumerate(segments, start=1):
        if not isinstance(segment, dict):
            raise ValueError(f"vibevoice segment at index {index - 1} must be an object.")

        text = segment.get("text")
        start = segment.get("start")
        end = segment.get("end")

        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"vibevoice segment at index {index - 1} is missing non-empty text.")
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, (float, int))
            or not isinstance(end, (float, int))
        ):
            raise ValueError(f"vibevoice segment at index {index - 1} has non-numeric timestamps.")

        normalized_segments.append(
            {
                "segment_id": f"seg_{index:04d}",
                "start": float(start),
                "end": float(end),
                "text": text,
            }
        )

    return normalized_segments

