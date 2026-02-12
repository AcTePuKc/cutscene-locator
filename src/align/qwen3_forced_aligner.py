"""Qwen3 forced alignment backend implementation."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Protocol

from .base import AlignmentResult, ReferenceSpan
from .validation import validate_alignment_result


class AlignerConfig(Protocol):
    """Minimal config contract required by forced aligner backends."""

    model_path: Path | None
    device: str


class Qwen3ForcedAligner:
    """Forced-aligner backend powered by Hugging Face Qwen3 checkpoints."""

    def __init__(self, config: AlignerConfig):
        self._config = config

    def align(self, audio_path: str, reference_spans: list[ReferenceSpan]) -> AlignmentResult:
        config = self._config
        if config.model_path is None:
            raise ValueError(
                "qwen3-forced-aligner backend requires a resolved model path. "
                "Provide --model-id or --model-path."
            )

        if not reference_spans:
            raise ValueError("qwen3-forced-aligner requires at least one reference span.")

        try:
            torch_module = import_module("torch")
            transformers_module = import_module("transformers")
        except ModuleNotFoundError as exc:
            raise ValueError(
                "qwen3-forced-aligner backend requires optional dependencies. "
                "Install them with: pip install 'cutscene-locator[asr_qwen3]'"
            ) from exc

        resolved_device = "cuda" if config.device == "cuda" else "cpu"
        if config.device == "auto":
            resolved_device = "cuda" if bool(torch_module.cuda.is_available()) else "cpu"

        if resolved_device == "cuda" and not bool(torch_module.cuda.is_available()):
            raise ValueError(
                "Requested --device cuda, but CUDA is not available in this environment. "
                "Use --device cpu or follow docs/CUDA.md."
            )

        pipeline_factory = getattr(transformers_module, "pipeline", None)
        if pipeline_factory is None:
            raise ValueError(
                "Installed transformers package is missing pipeline(). "
                "Install a supported transformers version for qwen3-forced-aligner."
            )

        device = 0 if resolved_device == "cuda" else -1
        transcript_text = " ".join(span["text"].strip() for span in reference_spans)
        try:
            align_pipeline = pipeline_factory(
                task="automatic-speech-recognition",
                model=str(config.model_path),
                device=device,
                trust_remote_code=True,
            )
        except Exception as exc:  # pragma: no cover - runtime specific
            raise ValueError(
                "Failed to initialize qwen3-forced-aligner model from resolved path "
                f"'{config.model_path}'. Ensure model artifacts are present and compatible."
            ) from exc

        try:
            raw_result = align_pipeline(audio_path, return_timestamps="word")
        except Exception as exc:  # pragma: no cover - runtime specific
            raise ValueError(
                f"qwen3-forced-aligner failed for '{audio_path}'. "
                "Verify input audio and model compatibility."
            ) from exc

        spans = _normalize_alignment_spans(raw_result, reference_spans)

        backend_version = "unknown"
        try:
            backend_version = version("transformers")
        except PackageNotFoundError:
            backend_version = "unknown"

        model_name = Path(config.model_path).name or str(config.model_path)
        return validate_alignment_result(
            {
                "transcript_text": transcript_text,
                "spans": spans,
                "meta": {
                    "backend": "qwen3-forced-aligner",
                    "version": backend_version,
                    "device": resolved_device,
                },
            },
            source=f"qwen3-forced-aligner:{model_name}",
        )


def _normalize_alignment_spans(raw_result: object, reference_spans: list[ReferenceSpan]) -> list[dict[str, object]]:
    chunks = raw_result.get("chunks") if isinstance(raw_result, dict) else None
    if not isinstance(chunks, list) or not chunks:
        raise ValueError(
            "qwen3-forced-aligner backend did not return timestamped chunks. "
            "Expected a non-empty 'chunks' list when return_timestamps='word'."
        )

    spans: list[dict[str, object]] = []
    for index, span in enumerate(reference_spans, start=1):
        chunk = chunks[min(index - 1, len(chunks) - 1)]
        if not isinstance(chunk, dict):
            raise ValueError(f"qwen3-forced-aligner chunk at index {index - 1} must be an object.")

        timestamps = chunk.get("timestamp")
        confidence = chunk.get("confidence", 1.0)
        if not isinstance(timestamps, tuple) or len(timestamps) != 2:
            raise ValueError(
                "qwen3-forced-aligner chunk at index "
                f"{index - 1} has invalid timestamp format. Expected (start, end)."
            )

        start, end = timestamps
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, (int, float))
            or not isinstance(end, (int, float))
        ):
            raise ValueError(f"qwen3-forced-aligner chunk at index {index - 1} has non-numeric timestamps.")

        confidence_value: float
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            confidence_value = 1.0
        else:
            confidence_value = float(confidence)

        spans.append(
            {
                "span_id": span["ref_id"],
                "start": float(start),
                "end": float(end),
                "text": span["text"],
                "confidence": max(0.0, min(1.0, confidence_value)),
            }
        )

    return spans
