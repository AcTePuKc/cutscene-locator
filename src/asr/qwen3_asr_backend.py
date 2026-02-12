"""Qwen3 ASR backend implementation."""

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


class Qwen3ASRBackend:
    """ASR backend powered by Hugging Face Qwen3-ASR checkpoints."""

    def transcribe(self, audio_path: str, config: ASRConfig) -> ASRResult:
        if config.model_path is None:
            raise ValueError(
                "qwen3-asr backend requires a resolved model path. "
                "Provide --model-id or --model-path."
            )

        resolution = resolve_device_with_details(config.device)
        resolved_device = resolution.resolved

        try:
            torch_module = import_module("torch")
            transformers_module = import_module("transformers")
        except ModuleNotFoundError as exc:
            raise ValueError(
                "qwen3-asr backend requires optional dependencies. "
                "Install them with: pip install 'cutscene-locator[asr_qwen3]'"
            ) from exc

        if resolved_device == "cuda" and not bool(torch_module.cuda.is_available()):
            raise ValueError(
                "Requested --device cuda, but CUDA is not available in this environment. "
                "Use --device cpu or follow docs/CUDA.md."
            )

        pipeline_factory = getattr(transformers_module, "pipeline", None)
        if pipeline_factory is None:
            raise ValueError(
                "Installed transformers package is missing pipeline(). "
                "Install a supported transformers version for qwen3-asr."
            )

        device = 0 if resolved_device == "cuda" else -1
        try:
            asr_pipeline = pipeline_factory(
                task="automatic-speech-recognition",
                model=str(config.model_path),
                device=device,
                trust_remote_code=True,
            )
        except Exception as exc:  # pragma: no cover - backend/runtime specific
            raise ValueError(
                "Failed to initialize qwen3-asr model from resolved path "
                f"'{config.model_path}'. Ensure the model artifacts are present and compatible."
            ) from exc

        try:
            raw_result = asr_pipeline(audio_path, return_timestamps=True)
        except Exception as exc:  # pragma: no cover - backend/runtime specific
            raise ValueError(
                f"qwen3-asr transcription failed for '{audio_path}'. "
                "Verify input audio and model compatibility."
            ) from exc

        segments = normalize_asr_segments_for_contract(
            _normalize_qwen_segments(raw_result),
            source="qwen3-asr",
        )

        backend_version = "unknown"
        try:
            backend_version = version("transformers")
        except PackageNotFoundError:
            backend_version = "unknown"

        model_name = Path(config.model_path).name or str(config.model_path)
        return validate_asr_result(
            {
                "segments": segments,
                "meta": {
                    "backend": "qwen3-asr",
                    "model": model_name,
                    "version": backend_version,
                    "device": resolved_device,
                },
            },
            source="qwen3-asr",
        )


def _normalize_qwen_segments(raw_result: Any) -> list[dict[str, float | str]]:
    chunks = raw_result.get("chunks") if isinstance(raw_result, dict) else None
    if not isinstance(chunks, list) or not chunks:
        raise ValueError(
            "qwen3-asr backend did not return timestamped chunks. "
            "Expected a non-empty 'chunks' list when return_timestamps=True."
        )

    normalized_segments: list[dict[str, float | str]] = []
    for index, chunk in enumerate(chunks, start=1):
        if not isinstance(chunk, dict):
            raise ValueError(f"qwen3-asr chunk at index {index - 1} must be an object.")

        text = chunk.get("text")
        timestamps = chunk.get("timestamp")

        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"qwen3-asr chunk at index {index - 1} is missing non-empty text.")
        if not isinstance(timestamps, tuple) or len(timestamps) != 2:
            raise ValueError(
                "qwen3-asr chunk at index "
                f"{index - 1} has invalid timestamp format. Expected (start, end)."
            )

        start, end = timestamps
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, (float, int))
            or not isinstance(end, (float, int))
        ):
            raise ValueError(
                f"qwen3-asr chunk at index {index - 1} has non-numeric timestamps."
            )

        normalized_segments.append(
            {
                "segment_id": f"seg_{index:04d}",
                "start": float(start),
                "end": float(end),
                "text": text,
            }
        )

    return normalized_segments
