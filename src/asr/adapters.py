"""ASR adapter layer that standardizes backend execution plumbing."""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol, TypeAlias

from .backends import MockASRBackend
from .base import ASRMeta, ASRResult, ASRSegment
from .config import ASRConfig
from .device import resolve_device_with_details, select_cuda_probe
from .faster_whisper_backend import FasterWhisperBackend
from .qwen3_asr_backend import Qwen3ASRBackend
from .whisperx_backend import WhisperXBackend
from .vibevoice_backend import VibeVoiceBackend
from .registry import get_backend, validate_backend_capabilities

_CONTINUITY_MAX_GAP_SECONDS = 0.12
_CONTINUITY_TINY_FRAGMENT_SECONDS = 0.35
_CONTINUITY_EDGE_DUPLICATE_WINDOW_SECONDS = 0.35


def _normalize_continuity_text(text: str) -> str:
    lowered = text.lower().strip()
    lowered = re.sub(r"\s+", " ", lowered)
    lowered = re.sub(r"[^\w\s]", "", lowered)
    return lowered


def _merge_segment_text(left_text: str, right_text: str) -> str:
    left_trimmed = left_text.strip()
    right_trimmed = right_text.strip()
    if not left_trimmed:
        return right_trimmed
    if not right_trimmed:
        return left_trimmed

    left_tokens = left_trimmed.split()
    right_tokens = right_trimmed.split()
    max_overlap = min(len(left_tokens), len(right_tokens))
    for overlap in range(max_overlap, 0, -1):
        if left_tokens[-overlap:] == right_tokens[:overlap]:
            merged_tokens = left_tokens + right_tokens[overlap:]
            return " ".join(merged_tokens)

    return f"{left_trimmed} {right_trimmed}"


def _segment_chunk_index(segment: dict[str, Any]) -> int | None:
    raw_value = segment.get("chunk_index")
    if isinstance(raw_value, bool) or not isinstance(raw_value, int):
        return None
    return raw_value


def _should_merge_boundary_segments(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_chunk = _segment_chunk_index(left)
    right_chunk = _segment_chunk_index(right)
    if left_chunk is None or right_chunk is None:
        return False
    if right_chunk - left_chunk != 1:
        return False

    left_start = float(left["start"])
    left_end = float(left["end"])
    right_start = float(right["start"])
    right_end = float(right["end"])
    gap = right_start - left_end
    overlap_seconds = min(left_end, right_end) - max(left_start, right_start)
    left_duration = left_end - left_start
    right_duration = right_end - right_start

    left_normalized_text = _normalize_continuity_text(str(left["text"]))
    right_normalized_text = _normalize_continuity_text(str(right["text"]))

    if left_normalized_text == right_normalized_text and (
        overlap_seconds >= 0.0 or abs(gap) <= _CONTINUITY_EDGE_DUPLICATE_WINDOW_SECONDS
    ):
        return True

    if gap <= _CONTINUITY_MAX_GAP_SECONDS and (
        left_duration <= _CONTINUITY_TINY_FRAGMENT_SECONDS
        or right_duration <= _CONTINUITY_TINY_FRAGMENT_SECONDS
    ):
        return True

    if gap <= _CONTINUITY_MAX_GAP_SECONDS and left_normalized_text and right_normalized_text:
        return True

    return False


def _merge_boundary_segments(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {
        "segment_id": left["segment_id"],
        "start": min(float(left["start"]), float(right["start"])),
        "end": max(float(left["end"]), float(right["end"])),
        "text": _merge_segment_text(str(left["text"]), str(right["text"])),
    }
    if "speaker" in left:
        merged["speaker"] = left["speaker"]
    elif "speaker" in right:
        merged["speaker"] = right["speaker"]

    left_chunk = _segment_chunk_index(left)
    right_chunk = _segment_chunk_index(right)
    if left_chunk is not None:
        merged["chunk_index"] = left_chunk
    if left_chunk is not None and right_chunk is not None:
        merged["merged_chunk_indexes"] = [left_chunk, right_chunk]

    return merged


def apply_cross_chunk_continuity(
    *,
    asr_result: ASRResult,
    chunk_offsets_by_index: dict[int, float],
) -> ASRResult:
    """Convert chunk-local ASR timestamps to absolute timeline + merge deterministic boundary continuity."""

    absolute_segments: list[tuple[float, float, int, dict[str, Any]]] = []
    for index, segment in enumerate(asr_result["segments"]):
        segment_copy: dict[str, Any] = dict(segment)
        chunk_index = _segment_chunk_index(segment_copy)
        if chunk_index is not None:
            offset_seconds = float(chunk_offsets_by_index.get(chunk_index, 0.0))
            segment_copy["start"] = float(segment_copy["start"]) + offset_seconds
            segment_copy["end"] = float(segment_copy["end"]) + offset_seconds

        absolute_segments.append(
            (float(segment_copy["start"]), float(segment_copy["end"]), index, segment_copy)
        )

    absolute_segments.sort(key=lambda item: (item[0], item[1], item[2]))

    merged_segments: list[dict[str, Any]] = []
    for _, _, _, segment in absolute_segments:
        if not merged_segments:
            merged_segments.append(segment)
            continue

        previous = merged_segments[-1]
        if _should_merge_boundary_segments(previous, segment):
            merged_segments[-1] = _merge_boundary_segments(previous, segment)
        else:
            merged_segments.append(segment)

    typed_segments: list[ASRSegment] = []
    for merged_segment in merged_segments:
        typed_segment: ASRSegment = {
            "segment_id": str(merged_segment["segment_id"]),
            "start": float(merged_segment["start"]),
            "end": float(merged_segment["end"]),
            "text": str(merged_segment["text"]),
        }
        speaker = merged_segment.get("speaker")
        if speaker is not None:
            typed_segment["speaker"] = str(speaker)
        typed_segments.append(typed_segment)

    raw_meta = asr_result["meta"]
    typed_meta: ASRMeta = {
        "backend": raw_meta["backend"],
        "model": raw_meta["model"],
        "version": raw_meta["version"],
        "device": raw_meta["device"],
    }

    result: ASRResult = {
        "segments": typed_segments,
        "meta": typed_meta,
    }
    return result


class FasterWhisperSubprocessRunner(Protocol):
    """Callable contract for faster-whisper worker subprocess execution."""

    def __call__(
        self,
        *,
        audio_path: Path,
        resolved_model_path: Path,
        asr_config: ASRConfig,
        verbose: bool,
    ) -> ASRResult: ...


class FasterWhisperPreflightPrinter(Protocol):
    """Callable contract for faster-whisper CUDA preflight diagnostics."""

    def __call__(self, *, device: str, compute_type: str) -> None: ...


FasterWhisperSubprocessRunnerType: TypeAlias = FasterWhisperSubprocessRunner
FasterWhisperPreflightPrinterType: TypeAlias = FasterWhisperPreflightPrinter


@dataclass(frozen=True)
class CapabilityRequirements:
    """Pipeline requirements checked against backend capability metadata."""

    requires_segment_timestamps: bool = True
    allows_alignment_backends: bool = False


@dataclass(frozen=True)
class ASRExecutionContext:
    """Execution context shared by all adapters."""

    resolved_model_path: Path | None
    verbose: bool
    mock_asr_path: str | None = None
    run_faster_whisper_subprocess: FasterWhisperSubprocessRunnerType | None = None
    faster_whisper_preflight: FasterWhisperPreflightPrinterType | None = None


class ASRAdapter(Protocol):
    """Unified adapter contract for backend execution."""

    backend_name: str

    def load_model(self, config: ASRConfig, context: ASRExecutionContext) -> object | None:
        ...

    def build_backend_kwargs(self, config: ASRConfig) -> dict[str, object]:
        ...

    def filter_backend_kwargs(
        self,
        candidate_kwargs: dict[str, object],
        *,
        allowed_keys: set[str],
    ) -> dict[str, object]:
        ...

    def normalize_output(self, raw_result: ASRResult) -> ASRResult:
        ...

    def transcribe(self, audio_path: str, config: ASRConfig, context: ASRExecutionContext) -> ASRResult:
        ...


class _BaseASRAdapter(ABC):
    backend_name: str

    def load_model(self, config: ASRConfig, context: ASRExecutionContext) -> object | None:
        del config, context
        return None

    def build_backend_kwargs(self, config: ASRConfig) -> dict[str, object]:
        return {
            "language": config.language,
            "beam_size": config.beam_size,
            "temperature": config.temperature,
            "best_of": config.best_of,
            "no_speech_threshold": config.no_speech_threshold,
            "log_prob_threshold": config.log_prob_threshold,
            "vad_filter": config.vad_filter,
        }

    def filter_backend_kwargs(
        self,
        candidate_kwargs: dict[str, object],
        *,
        allowed_keys: set[str],
    ) -> dict[str, object]:
        return {
            key: candidate_kwargs[key]
            for key in sorted(candidate_kwargs)
            if key in allowed_keys and candidate_kwargs[key] is not None
        }

    def normalize_output(self, raw_result: ASRResult) -> ASRResult:
        return raw_result

    @abstractmethod
    def transcribe(self, audio_path: str, config: ASRConfig, context: ASRExecutionContext) -> ASRResult:
        """Execute transcription using the backend adapter."""


class MockASRAdapter(_BaseASRAdapter):
    backend_name = "mock"

    def transcribe(self, audio_path: str, config: ASRConfig, context: ASRExecutionContext) -> ASRResult:
        if context.mock_asr_path is None:
            raise ValueError("--mock-asr is required when --asr-backend mock is used.")
        backend = MockASRBackend(Path(context.mock_asr_path))
        return self.normalize_output(backend.transcribe(audio_path, config))


class FasterWhisperASRAdapter(_BaseASRAdapter):
    backend_name = "faster-whisper"

    def transcribe(self, audio_path: str, config: ASRConfig, context: ASRExecutionContext) -> ASRResult:
        backend = FasterWhisperBackend()
        effective_config = replace(
            config,
            model_path=context.resolved_model_path,
            log_callback=print if context.verbose else config.log_callback,
        )

        cuda_checker, cuda_probe_label = select_cuda_probe(self.backend_name)
        resolution = resolve_device_with_details(
            effective_config.device,
            cuda_available_checker=cuda_checker,
            cuda_probe_reason_label=cuda_probe_label,
        )
        if context.faster_whisper_preflight is not None:
            context.faster_whisper_preflight(device=resolution.resolved, compute_type=effective_config.compute_type)

        if (
            effective_config.device == "cuda"
            and os.name == "nt"
            and context.run_faster_whisper_subprocess is not None
        ):
            if effective_config.model_path is None:
                raise ValueError("faster-whisper backend requires a resolved model path.")
            return context.run_faster_whisper_subprocess(
                audio_path=Path(audio_path),
                resolved_model_path=effective_config.model_path,
                asr_config=effective_config,
                verbose=context.verbose,
            )

        return self.normalize_output(backend.transcribe(audio_path, effective_config))


class Qwen3ASRAdapter(_BaseASRAdapter):
    backend_name = "qwen3-asr"

    def transcribe(self, audio_path: str, config: ASRConfig, context: ASRExecutionContext) -> ASRResult:
        backend = Qwen3ASRBackend()
        effective_config = replace(
            config,
            model_path=context.resolved_model_path,
            log_callback=print if context.verbose else config.log_callback,
        )
        cuda_checker, cuda_probe_label = select_cuda_probe(self.backend_name)
        resolution = resolve_device_with_details(
            effective_config.device,
            cuda_available_checker=cuda_checker,
            cuda_probe_reason_label=cuda_probe_label,
        )
        effective_config = replace(effective_config, device=resolution.resolved)
        return self.normalize_output(backend.transcribe(audio_path, effective_config))


class WhisperXASRAdapter(_BaseASRAdapter):
    backend_name = "whisperx"

    def transcribe(self, audio_path: str, config: ASRConfig, context: ASRExecutionContext) -> ASRResult:
        backend = WhisperXBackend()
        effective_config = replace(
            config,
            model_path=context.resolved_model_path,
            log_callback=print if context.verbose else config.log_callback,
        )
        cuda_checker, cuda_probe_label = select_cuda_probe(self.backend_name)
        resolution = resolve_device_with_details(
            effective_config.device,
            cuda_available_checker=cuda_checker,
            cuda_probe_reason_label=cuda_probe_label,
        )
        effective_config = replace(effective_config, device=resolution.resolved)
        return self.normalize_output(backend.transcribe(audio_path, effective_config))


class VibeVoiceASRAdapter(_BaseASRAdapter):
    backend_name = "vibevoice"

    def transcribe(self, audio_path: str, config: ASRConfig, context: ASRExecutionContext) -> ASRResult:
        backend = VibeVoiceBackend()
        effective_config = replace(
            config,
            model_path=context.resolved_model_path,
            log_callback=print if context.verbose else config.log_callback,
        )
        cuda_checker, cuda_probe_label = select_cuda_probe(self.backend_name)
        resolution = resolve_device_with_details(
            effective_config.device,
            cuda_available_checker=cuda_checker,
            cuda_probe_reason_label=cuda_probe_label,
        )
        effective_config = replace(effective_config, device=resolution.resolved)
        return self.normalize_output(backend.transcribe(audio_path, effective_config))


_ADAPTER_REGISTRY: dict[str, type[_BaseASRAdapter]] = {
    "mock": MockASRAdapter,
    "faster-whisper": FasterWhisperASRAdapter,
    "qwen3-asr": Qwen3ASRAdapter,
    "whisperx": WhisperXASRAdapter,
    "vibevoice": VibeVoiceASRAdapter,
}


def list_asr_adapters() -> list[str]:
    """Return registered adapter names in deterministic order."""

    return sorted(_ADAPTER_REGISTRY)


def get_asr_adapter(name: str) -> _BaseASRAdapter:
    """Return adapter instance by backend name."""

    adapter_class = _ADAPTER_REGISTRY.get(name)
    if adapter_class is None:
        available = ", ".join(list_asr_adapters())
        raise ValueError(f"No ASR adapter registered for backend '{name}'. Available adapters: {available}")
    return adapter_class()


def dispatch_asr_transcription(
    *,
    audio_path: str,
    config: ASRConfig,
    context: ASRExecutionContext,
    requirements: CapabilityRequirements | None = None,
) -> ASRResult:
    """Dispatch ASR transcription through registry preflight + adapter execution."""

    effective_requirements = requirements or CapabilityRequirements()
    registration = get_backend(config.backend_name)
    validate_backend_capabilities(
        registration,
        requires_segment_timestamps=effective_requirements.requires_segment_timestamps,
        allows_alignment_backends=effective_requirements.allows_alignment_backends,
    )
    adapter = get_asr_adapter(registration.name)
    return adapter.transcribe(audio_path, config, context)
