"""ASR backend registry and capability metadata."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any

from .backends import MockASRBackend
from .faster_whisper_backend import FasterWhisperBackend
from .qwen3_asr_backend import Qwen3ASRBackend
from .whisperx_backend import WhisperXBackend
from .vibevoice_backend import VibeVoiceBackend
from src.align.qwen3_forced_aligner import Qwen3ForcedAligner


@dataclass(frozen=True)
class BackendCapabilities:
    """Capability metadata exposed for each ASR backend."""

    supports_segment_timestamps: bool
    supports_word_timestamps: bool
    supports_alignment: bool
    supports_diarization: bool
    max_audio_duration: int | None


@dataclass(frozen=True)
class BackendRegistration:
    """Registered backend metadata and implementation class."""

    name: str
    backend_class: type[Any]
    capabilities: BackendCapabilities


@dataclass(frozen=True)
class DeclaredBackend:
    """Backend declaration including optional dependency metadata."""

    registration: BackendRegistration
    required_dependencies: tuple[str, ...]
    install_extra: str | None = None


@dataclass(frozen=True)
class BackendStatus:
    """Backend availability status used by CLI validation/help output."""

    name: str
    enabled: bool
    missing_dependencies: tuple[str, ...]
    reason: str
    install_extra: str | None = None


def _missing_dependencies(required_dependencies: tuple[str, ...]) -> tuple[str, ...]:
    """Return required dependency import names that are unavailable."""

    return tuple(dep for dep in required_dependencies if find_spec(dep) is None)


def _build_declared_registry() -> dict[str, DeclaredBackend]:
    default_capabilities = BackendCapabilities(
        supports_segment_timestamps=True,
        supports_word_timestamps=False,
        supports_alignment=False,
        supports_diarization=False,
        max_audio_duration=None,
    )
    alignment_capabilities = BackendCapabilities(
        supports_segment_timestamps=True,
        supports_word_timestamps=False,
        supports_alignment=True,
        supports_diarization=False,
        max_audio_duration=None,
    )
    return {
        "faster-whisper": DeclaredBackend(
            registration=BackendRegistration(
                name="faster-whisper",
                backend_class=FasterWhisperBackend,
                capabilities=default_capabilities,
            ),
            required_dependencies=(),
        ),
        "mock": DeclaredBackend(
            registration=BackendRegistration(
                name="mock",
                backend_class=MockASRBackend,
                capabilities=default_capabilities,
            ),
            required_dependencies=(),
        ),
        "whisperx": DeclaredBackend(
            registration=BackendRegistration(
                name="whisperx",
                backend_class=WhisperXBackend,
                capabilities=BackendCapabilities(
                    supports_segment_timestamps=True,
                    supports_word_timestamps=False,
                    supports_alignment=False,
                    supports_diarization=True,
                    max_audio_duration=None,
                ),
            ),
            required_dependencies=("whisperx", "torch"),
            install_extra="asr_whisperx",
        ),
        "qwen3-asr": DeclaredBackend(
            registration=BackendRegistration(
                name="qwen3-asr",
                backend_class=Qwen3ASRBackend,
                capabilities=default_capabilities,
            ),
            required_dependencies=("torch", "qwen_asr"),
            install_extra="asr_qwen3",
        ),
        "qwen3-forced-aligner": DeclaredBackend(
            registration=BackendRegistration(
                name="qwen3-forced-aligner",
                backend_class=Qwen3ForcedAligner,
                capabilities=alignment_capabilities,
            ),
            required_dependencies=("torch", "qwen_asr"),
            install_extra="asr_qwen3",
        ),
        "vibevoice": DeclaredBackend(
            registration=BackendRegistration(
                name="vibevoice",
                backend_class=VibeVoiceBackend,
                capabilities=default_capabilities,
            ),
            required_dependencies=("vibevoice", "torch"),
            install_extra="asr_vibevoice",
        ),
    }


_DECLARED_REGISTRY = _build_declared_registry()


def list_backend_status() -> list[BackendStatus]:
    """Return declared backend status in deterministic order."""

    statuses: list[BackendStatus] = []
    for name in sorted(_DECLARED_REGISTRY):
        declared_backend = _DECLARED_REGISTRY[name]
        missing_dependencies = _missing_dependencies(declared_backend.required_dependencies)
        enabled = not missing_dependencies
        reason = "enabled" if enabled else f"missing optional dependencies: {', '.join(missing_dependencies)}"
        statuses.append(
            BackendStatus(
                name=name,
                enabled=enabled,
                missing_dependencies=missing_dependencies,
                reason=reason,
                install_extra=declared_backend.install_extra,
            )
        )
    return statuses


def list_declared_backends() -> list[str]:
    """Return all declared backend names in deterministic order."""

    return sorted(_DECLARED_REGISTRY.keys())



def validate_backend_capabilities(
    registration: BackendRegistration,
    *,
    requires_segment_timestamps: bool,
    allows_alignment_backends: bool,
) -> None:
    """Validate backend capabilities against pipeline requirements."""

    supports_segment_timestamps = getattr(registration.capabilities, "supports_segment_timestamps", True)
    supports_alignment = getattr(registration.capabilities, "supports_alignment", False)

    if requires_segment_timestamps and not supports_segment_timestamps:
        raise ValueError(
            f"ASR backend '{registration.name}' does not provide required segment timestamps."
        )

    if not allows_alignment_backends and supports_alignment:
        raise ValueError(
            f"'{registration.name}' is an alignment backend and cannot be used with --asr-backend. "
            "Use the alignment pipeline path instead of ASR-only transcription mode."
        )

def _build_enabled_registry() -> dict[str, BackendRegistration]:
    enabled_registry: dict[str, BackendRegistration] = {}
    for status in list_backend_status():
        if not status.enabled:
            continue
        enabled_registry[status.name] = _DECLARED_REGISTRY[status.name].registration
    return enabled_registry


_REGISTRY = _build_enabled_registry()


def list_backends() -> list[str]:
    """Return enabled backend names in deterministic order."""

    return sorted(_REGISTRY.keys())


def get_backend(name: str) -> BackendRegistration:
    """Return enabled backend registration by name or raise ValueError."""

    registration = _REGISTRY.get(name)
    if registration is None:
        available = ", ".join(list_backends())
        raise ValueError(f"Unknown ASR backend '{name}'. Available backends: {available}")
    return registration
