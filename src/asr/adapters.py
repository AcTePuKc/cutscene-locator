"""ASR adapter layer that standardizes backend execution plumbing."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol, TypeAlias

from .backends import MockASRBackend
from .base import ASRResult
from .config import ASRConfig
from .device import resolve_device_with_details
from .faster_whisper_backend import FasterWhisperBackend
from .qwen3_asr_backend import Qwen3ASRBackend
from .whisperx_backend import WhisperXBackend
from .vibevoice_backend import VibeVoiceBackend
from .registry import get_backend, validate_backend_capabilities


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

        resolution = resolve_device_with_details(effective_config.device)
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
