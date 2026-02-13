"""Qwen3 ASR backend implementation."""

from __future__ import annotations

import traceback
from inspect import Parameter, signature
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable

from .backends import validate_asr_result
from .base import ASRResult, ASRSegment
from .config import ASRConfig
from .timestamp_normalization import normalize_asr_segments_for_contract

_SUPPORTED_BACKEND_CONTROLS = ("language", "device", "dtype", "batch_size", "chunk_length_s")


class Qwen3ASRBackend:
    """ASR backend powered by official qwen_asr runtime."""

    def transcribe(self, audio_path: str, config: ASRConfig) -> ASRResult:
        if config.model_path is None:
            raise ValueError(
                "qwen3-asr backend requires a resolved model path. "
                "Provide --model-id or --model-path."
            )

        resolved_device = config.device if config.device in {"cpu", "cuda"} else "cpu"
        resolved_model_path = str(config.model_path)

        try:
            qwen_asr_module = import_module("qwen_asr")
        except ModuleNotFoundError as exc:
            raise ValueError(
                "qwen3-asr backend requires optional dependencies and the `qwen_asr` import target. "
                "Install them with: pip install 'cutscene-locator[asr_qwen3]' and verify `import qwen_asr` works."
            ) from exc

        qwen_model_class = getattr(qwen_asr_module, "Qwen3ASRModel", None)
        if qwen_model_class is None:
            raise ValueError(
                "Installed qwen_asr package is missing required Qwen3ASRModel API."
            )

        _validate_supported_options(config)

        model_init_kwargs: dict[str, object] = {
            "dtype": _resolve_dtype(config.compute_type),
        }

        try:
            model = qwen_model_class.from_pretrained(
                resolved_model_path,
                **model_init_kwargs,
            )
        except Exception as exc:  # pragma: no cover - backend/runtime specific
            if config.log_callback is not None:
                config.log_callback(traceback.format_exc())
            raise ValueError(
                "Failed to initialize qwen3-asr model from resolved path "
                f"'{config.model_path}'. Ensure this directory is a compatible qwen_asr "
                "snapshot with required artifacts: config.json, tokenizer assets "
                "(tokenizer.json/tokenizer.model/vocab.json), tokenizer_config.json, "
                "and model weights (model.safetensors/pytorch_model.bin or sharded index json). "
                "processor_config.json / preprocessor_config.json are optional. "
                "Runtime hints: verify qwen_asr/transformers/torch version compatibility; "
                "confirm this model repo supports qwen_asr Qwen3ASRModel loading; "
                "and check optional runtime dependencies are installed despite extras installation."
            ) from exc

        _move_model_to_device_or_raise(model, resolved_device)

        inference_kwargs = _build_qwen_transcribe_candidate_kwargs(config)

        inference_kwargs = _filter_supported_transcribe_kwargs(
            model.transcribe,
            inference_kwargs,
            config.log_callback,
        )

        try:
            raw_result = model.transcribe(audio_path, **inference_kwargs)
        except Exception as exc:  # pragma: no cover - backend/runtime specific
            if config.log_callback is not None:
                config.log_callback(traceback.format_exc())
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
            backend_version = version("qwen-asr")
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


def _filter_supported_transcribe_kwargs(
    transcribe_callable: object,
    candidate_kwargs: dict[str, object],
    log_callback: Callable[[str], None] | None,
) -> dict[str, object]:
    if not candidate_kwargs:
        return {}

    candidate_without_forbidden = {
        key: value
        for key, value in candidate_kwargs.items()
        if key != "return_timestamps"
    }

    if not candidate_without_forbidden:
        return {}

    try:
        transcribe_signature = signature(transcribe_callable)
    except (TypeError, ValueError):
        return dict(candidate_without_forbidden)

    supported_names = {
        parameter.name
        for parameter in transcribe_signature.parameters.values()
        if parameter.kind in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY)
    }
    filtered_kwargs = {
        key: value
        for key, value in candidate_without_forbidden.items()
        if key in supported_names
    }

    dropped = sorted(set(candidate_without_forbidden).difference(filtered_kwargs))
    if dropped and callable(log_callback):
        log_callback("asr: filtered unsupported qwen3 transcribe kwargs: " + ", ".join(dropped))

    return filtered_kwargs


def _build_qwen_transcribe_candidate_kwargs(config: ASRConfig) -> dict[str, object]:
    candidate_kwargs: dict[str, object] = {}
    if config.language is not None:
        candidate_kwargs["language"] = config.language
    if config.temperature != 0.0:
        candidate_kwargs["temperature"] = config.temperature
    if config.qwen3_batch_size is not None:
        candidate_kwargs["batch_size"] = config.qwen3_batch_size
    if config.qwen3_chunk_length_s is not None:
        candidate_kwargs["chunk_length_s"] = config.qwen3_chunk_length_s

    return candidate_kwargs


def _resolve_dtype(compute_type: str) -> str:
    return compute_type


def _move_model_to_device_or_raise(model: object, device: str) -> None:
    candidate_names = ("model",)
    transfer_candidates: list[tuple[str, object]] = []

    for candidate_name in candidate_names:
        candidate = getattr(model, candidate_name, None)
        if candidate is not None:
            transfer_candidates.append((f"Qwen3ASRModel.{candidate_name}", candidate))
    transfer_candidates.append(("Qwen3ASRModel", model))

    for candidate_label, candidate in transfer_candidates:
        to_method = getattr(candidate, "to", None)
        if not callable(to_method):
            continue

        try:
            to_method(device)
        except Exception as exc:  # pragma: no cover - runtime/API specific
            raise ValueError(
                "qwen3-asr model loaded but device transfer failed. "
                f"Attempted `{candidate_label}.to('{device}')`. "
                "This indicates a loader/API mismatch between cutscene-locator and the installed qwen_asr runtime."
            ) from exc
        return

    raise ValueError(
        "qwen3-asr model loaded but device transfer is unsupported by installed runtime object. "
        "Expected `Qwen3ASRModel.model.to(device)` or `Qwen3ASRModel.to(device)` after from_pretrained(). "
        "This indicates a loader/API mismatch between cutscene-locator and the installed qwen_asr runtime."
    )


def _validate_supported_options(config: ASRConfig) -> None:
    unsupported_options: list[str] = []

    if config.beam_size != 1:
        unsupported_options.append("beam_size")
    if config.best_of != 1:
        unsupported_options.append("best_of")
    if config.no_speech_threshold is not None:
        unsupported_options.append("no_speech_threshold")
    if config.log_prob_threshold is not None:
        unsupported_options.append("log_prob_threshold")
    if config.vad_filter:
        unsupported_options.append("vad_filter")

    if unsupported_options:
        raise ValueError(
            "qwen3-asr backend does not support options: "
            f"{', '.join(sorted(unsupported_options))}. "
            "Supported backend controls are: "
            f"{', '.join(_SUPPORTED_BACKEND_CONTROLS)}."
        )


def _normalize_qwen_segments(raw_result: Any) -> list[ASRSegment]:
    chunks = raw_result.get("chunks") if isinstance(raw_result, dict) else None
    if not isinstance(chunks, list) or not chunks:
        raise ValueError(
            "qwen3-asr backend did not return timestamped chunks. "
            "Expected a non-empty 'chunks' list with per-segment timestamps; qwen3-asr may emit text-only output depending on runtime/model behavior."
        )

    normalized_segments: list[ASRSegment] = []
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
