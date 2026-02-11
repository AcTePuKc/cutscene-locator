"""ASR backend registry and capability metadata."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any

from .backends import MockASRBackend
from .faster_whisper_backend import FasterWhisperBackend
from .qwen3_asr_backend import Qwen3ASRBackend


@dataclass(frozen=True)
class BackendCapabilities:
    """Capability metadata exposed for each ASR backend."""

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


def _qwen3_dependencies_available() -> bool:
    return find_spec("torch") is not None and find_spec("transformers") is not None


def _build_registry() -> dict[str, BackendRegistration]:
    registry: dict[str, BackendRegistration] = {
        "faster-whisper": BackendRegistration(
            name="faster-whisper",
            backend_class=FasterWhisperBackend,
            capabilities=BackendCapabilities(
                supports_word_timestamps=False,
                supports_alignment=False,
                supports_diarization=False,
                max_audio_duration=None,
            ),
        ),
        "mock": BackendRegistration(
            name="mock",
            backend_class=MockASRBackend,
            capabilities=BackendCapabilities(
                supports_word_timestamps=False,
                supports_alignment=False,
                supports_diarization=False,
                max_audio_duration=None,
            ),
        ),
    }

    if _qwen3_dependencies_available():
        registry["qwen3-asr"] = BackendRegistration(
            name="qwen3-asr",
            backend_class=Qwen3ASRBackend,
            capabilities=BackendCapabilities(
                supports_word_timestamps=False,
                supports_alignment=False,
                supports_diarization=False,
                max_audio_duration=None,
            ),
        )

    return registry


_REGISTRY = _build_registry()


def list_backends() -> list[str]:
    """Return available backend names in deterministic order."""

    return sorted(_REGISTRY.keys())


def get_backend(name: str) -> BackendRegistration:
    """Return backend registration by name or raise ValueError."""

    registration = _REGISTRY.get(name)
    if registration is None:
        available = ", ".join(list_backends())
        raise ValueError(f"Unknown ASR backend '{name}'. Available backends: {available}")
    return registration
