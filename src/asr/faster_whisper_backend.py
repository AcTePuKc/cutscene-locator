"""Pilot faster-whisper ASR backend implementation."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .base import ASRResult
from .backends import validate_asr_result
from .config import ASRConfig
from .device import resolve_device


class FasterWhisperBackend:
    """ASR backend powered by faster-whisper."""

    def transcribe(self, audio_path: str, config: ASRConfig) -> ASRResult:
        if config.model_path is None:
            raise ValueError(
                "faster-whisper backend requires a resolved model path. "
                "Provide --model-path or use models/faster-whisper, cache, or --auto-download tiny."
            )

        resolved_device = resolve_device(config.device)

        try:
            faster_whisper_module = import_module("faster_whisper")
        except ModuleNotFoundError as exc:
            raise ValueError(
                "faster-whisper backend requires the optional dependency. "
                "Install it with: pip install 'cutscene-locator[faster-whisper]'"
            ) from exc

        whisper_model_class = getattr(faster_whisper_module, "WhisperModel", None)
        if whisper_model_class is None:
            raise ValueError("Installed faster-whisper package is missing WhisperModel.")

        model = whisper_model_class(str(config.model_path), device=resolved_device)
        raw_segments, _info = model.transcribe(audio_path)

        normalized_segments: list[dict[str, str | float]] = []
        for index, segment in enumerate(raw_segments, start=1):
            text = str(getattr(segment, "text", "")).strip()
            normalized_segments.append(
                {
                    "segment_id": f"seg_{index:04d}",
                    "start": float(getattr(segment, "start")),
                    "end": float(getattr(segment, "end")),
                    "text": text,
                }
            )

        model_name = Path(config.model_path).name or str(config.model_path)
        backend_version = "unknown"
        try:
            backend_version = version("faster-whisper")
        except PackageNotFoundError:
            backend_version = "unknown"

        return validate_asr_result(
            {
                "segments": normalized_segments,
                "meta": {
                    "backend": "faster-whisper",
                    "model": model_name,
                    "version": backend_version,
                    "device": resolved_device,
                },
            },
            source="faster-whisper",
        )
