"""Pilot faster-whisper ASR backend implementation."""

from __future__ import annotations

from inspect import Parameter, signature
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Callable

from .base import ASRResult, ASRSegment
from .backends import validate_asr_result
from .config import ASRConfig
from .device import resolve_device_with_details
from .timestamp_normalization import normalize_asr_segments_for_contract


_FORBIDDEN_TRANSCRIBE_KWARGS = frozenset({"progress"})


def _merge_short_segments(
    segments: list[ASRSegment],
    *,
    min_duration_seconds: float,
) -> list[ASRSegment]:
    if min_duration_seconds <= 0:
        return list(segments)

    merged: list[ASRSegment] = []
    for segment in segments:
        duration = float(segment["end"]) - float(segment["start"])
        if merged and duration < min_duration_seconds:
            previous = merged[-1]
            previous["end"] = float(segment["end"])
            previous["text"] = f"{str(previous['text']).strip()} {str(segment['text']).strip()}".strip()
            continue
        merged.append(dict(segment))

    for idx, segment in enumerate(merged, start=1):
        segment["segment_id"] = f"seg_{idx:04d}"
    return merged


class FasterWhisperBackend:
    """ASR backend powered by faster-whisper."""

    @staticmethod
    def _filter_supported_transcribe_kwargs(
        transcribe_callable: Callable[..., object],
        candidate_kwargs: dict[str, object],
        log_callback: Callable[[str], None] | None,
    ) -> dict[str, object]:
        """Keep only kwargs accepted by the transcribe() signature."""

        if not candidate_kwargs:
            return {}

        try:
            transcribe_signature = signature(transcribe_callable)
        except (TypeError, ValueError):
            return dict(candidate_kwargs)

        supports_var_kwargs = any(
            parameter.kind == Parameter.VAR_KEYWORD
            for parameter in transcribe_signature.parameters.values()
        )
        if supports_var_kwargs:
            return dict(candidate_kwargs)

        supported_names = set(transcribe_signature.parameters)
        filtered_kwargs = {
            key: value for key, value in candidate_kwargs.items() if key in supported_names
        }

        dropped = sorted(set(candidate_kwargs).difference(filtered_kwargs))
        if dropped and log_callback is not None:
            log_callback(
                "asr: filtered unsupported transcribe kwargs: " + ", ".join(dropped)
            )

        return filtered_kwargs

    def transcribe(self, audio_path: str, config: ASRConfig) -> ASRResult:
        if config.model_path is None:
            raise ValueError(
                "faster-whisper backend requires a resolved model path. "
                "Provide --model-path or use models/faster-whisper, cache, or --auto-download tiny."
            )

        resolved_device = resolve_device_with_details(config.device).resolved
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

        if config.log_callback is not None:
            config.log_callback("asr: model init start")
        model = whisper_model_class(
            str(config.model_path),
            device=resolved_device,
            compute_type=config.compute_type,
        )
        if config.log_callback is not None:
            config.log_callback("asr: model init end")

        if config.log_callback is not None:
            config.log_callback("asr: transcribe start")

        transcribe_kwargs: dict[str, object] = {
            "vad_filter": config.vad_filter,
            "temperature": config.temperature,
        }
        if config.language is not None:
            transcribe_kwargs["language"] = config.language
        if config.beam_size != 1:
            transcribe_kwargs["beam_size"] = config.beam_size
        if config.temperature > 0.0 and config.best_of != 1:
            transcribe_kwargs["best_of"] = config.best_of
        if config.no_speech_threshold is not None:
            transcribe_kwargs["no_speech_threshold"] = config.no_speech_threshold
        if config.log_prob_threshold is not None:
            transcribe_kwargs["log_prob_threshold"] = config.log_prob_threshold
        transcribe_kwargs = self._filter_supported_transcribe_kwargs(
            model.transcribe,
            transcribe_kwargs,
            config.log_callback,
        )
        forbidden = sorted(_FORBIDDEN_TRANSCRIBE_KWARGS.intersection(transcribe_kwargs))
        if forbidden:
            raise ValueError(
                "faster-whisper transcribe kwargs include unsupported options: "
                + ", ".join(forbidden)
            )
        if config.log_callback is not None:
            config.log_callback(f"asr: transcribe kwargs={transcribe_kwargs}")

        raw_segments, _info = model.transcribe(audio_path, **transcribe_kwargs)
        if config.log_callback is not None:
            config.log_callback("asr: transcribe end")

        normalized_segments: list[ASRSegment] = []
        segments_iterable = raw_segments
        if resolved_device == "cuda":
            if config.log_callback is not None:
                config.log_callback("asr: segments consume start")
            segments_list = list(raw_segments)
            if config.log_callback is not None:
                config.log_callback(f"asr: segments consume end; n={len(segments_list)}")
            segments_iterable = segments_list

        for index, segment in enumerate(segments_iterable, start=1):
            text = str(getattr(segment, "text", ""))
            normalized_segments.append(
                {
                    "segment_id": f"seg_{index:04d}",
                    "start": float(getattr(segment, "start")),
                    "end": float(getattr(segment, "end")),
                    "text": text,
                }
            )


        if config.merge_short_segments_seconds > 0:
            normalized_segments = _merge_short_segments(
                normalized_segments,
                min_duration_seconds=config.merge_short_segments_seconds,
            )

        normalized_segments = normalize_asr_segments_for_contract(
            normalized_segments,
            source="faster-whisper",
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
