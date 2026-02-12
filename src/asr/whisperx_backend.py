"""WhisperX ASR backend implementation."""

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


class WhisperXBackend:
    """ASR backend powered by WhisperX runtime."""

    def transcribe(self, audio_path: str, config: ASRConfig) -> ASRResult:
        if config.model_path is None:
            raise ValueError(
                "whisperx backend requires a resolved model path. "
                "Provide --model-path or --model-id."
            )

        resolution = resolve_device_with_details(config.device)
        resolved_device = resolution.resolved

        try:
            whisperx_module = import_module("whisperx")
            torch_module = import_module("torch")
        except ModuleNotFoundError as exc:
            raise ValueError(
                "whisperx backend requires optional dependencies. "
                "Install them with: pip install 'cutscene-locator[asr_whisperx]'"
            ) from exc

        if resolved_device == "cuda" and not bool(torch_module.cuda.is_available()):
            raise ValueError(
                "Requested --device cuda, but CUDA is not available in this environment. "
                "Use --device cpu or follow docs/CUDA.md."
            )

        load_model = getattr(whisperx_module, "load_model", None)
        load_audio = getattr(whisperx_module, "load_audio", None)
        if load_model is None or load_audio is None:
            raise ValueError(
                "Installed whisperx package is missing required runtime APIs "
                "(load_model/load_audio)."
            )

        compute_type = config.compute_type if config.compute_type != "auto" else "float32"

        try:
            model = load_model(
                str(config.model_path),
                resolved_device,
                compute_type=compute_type,
                language=config.language,
                download_root=str(config.model_path.parent),
            )
        except Exception as exc:  # pragma: no cover - runtime specific
            raise ValueError(
                "Failed to initialize whisperx model from resolved path "
                f"'{config.model_path}'. Ensure this directory is a compatible local snapshot."
            ) from exc

        try:
            audio = load_audio(audio_path)
            raw_result = model.transcribe(audio, batch_size=1)
        except Exception as exc:  # pragma: no cover - runtime specific
            raise ValueError(
                f"whisperx transcription failed for '{audio_path}'. "
                "Verify input audio and model compatibility."
            ) from exc

        segments = normalize_asr_segments_for_contract(
            _normalize_whisperx_segments(raw_result),
            source="whisperx",
        )

        backend_version = "unknown"
        try:
            backend_version = version("whisperx")
        except PackageNotFoundError:
            backend_version = "unknown"

        model_name = Path(config.model_path).name or str(config.model_path)
        return validate_asr_result(
            {
                "segments": segments,
                "meta": {
                    "backend": "whisperx",
                    "model": model_name,
                    "version": backend_version,
                    "device": resolved_device,
                },
            },
            source="whisperx",
        )


def _normalize_whisperx_segments(raw_result: Any) -> list[dict[str, float | str]]:
    segments = raw_result.get("segments") if isinstance(raw_result, dict) else None
    if not isinstance(segments, list) or not segments:
        raise ValueError(
            "whisperx backend did not return timestamped segments. "
            "Expected a non-empty 'segments' list."
        )

    normalized_segments: list[dict[str, float | str]] = []
    for index, segment in enumerate(segments, start=1):
        if not isinstance(segment, dict):
            raise ValueError(f"whisperx segment at index {index - 1} must be an object.")

        text = segment.get("text")
        start = segment.get("start")
        end = segment.get("end")

        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"whisperx segment at index {index - 1} is missing non-empty text.")
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, (float, int))
            or not isinstance(end, (float, int))
        ):
            raise ValueError(f"whisperx segment at index {index - 1} has non-numeric timestamps.")

        normalized_segment: dict[str, float | str] = {
            "segment_id": f"seg_{index:04d}",
            "start": float(start),
            "end": float(end),
            "text": text,
        }
        speaker = segment.get("speaker")
        if speaker is not None:
            normalized_segment["speaker"] = str(speaker)
        normalized_segments.append(normalized_segment)

    return normalized_segments

