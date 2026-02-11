"""ASR backend registry and capability metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .backends import MockASRBackend


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


_REGISTRY: dict[str, BackendRegistration] = {
    "mock": BackendRegistration(
        name="mock",
        backend_class=MockASRBackend,
        capabilities=BackendCapabilities(
            supports_word_timestamps=False,
            supports_alignment=False,
            supports_diarization=False,
            max_audio_duration=None,
        ),
    )
}


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
