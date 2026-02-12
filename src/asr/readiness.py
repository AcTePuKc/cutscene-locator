"""Deterministic backend readiness checks for install/runtime preconditions."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path

from .device import resolve_device_with_details, select_cuda_probe
from .model_resolution import ModelResolutionError, validate_model_artifact_layout
from .registry import list_backend_status


@dataclass(frozen=True)
class BackendReadiness:
    """Readiness summary for a single backend."""

    backend: str
    install_extra: str
    required_dependencies: tuple[str, ...]
    missing_dependencies: tuple[str, ...]
    registry_enabled: bool
    model_layout_valid: bool
    model_layout_error: str | None
    cuda_probe_label: str
    cuda_preflight_reason: str


_BACKEND_RUNTIME_PRECONDITIONS: dict[str, tuple[str, tuple[str, ...], str]] = {
    "qwen3-asr": (
        "asr_qwen3",
        ("torch", "qwen_asr"),
        "Transformers ASR snapshot: config + tokenizer + tokenizer_config + processor/preprocessor + weights.",
    ),
    "whisperx": (
        "asr_whisperx",
        ("whisperx", "torch"),
        "CTranslate2 Whisper snapshot: config.json + model.bin + tokenizer/vocabulary asset.",
    ),
    "vibevoice": (
        "asr_vibevoice",
        ("vibevoice", "torch"),
        "Local runtime-compatible VibeVoice model path (--model-path or resolved --model-id).",
    ),
}


def supported_readiness_backends() -> tuple[str, ...]:
    """Return backends covered by readiness matrix checks."""

    return tuple(sorted(_BACKEND_RUNTIME_PRECONDITIONS))


def backend_runtime_preconditions() -> dict[str, str]:
    """Return deterministic runtime precondition summary by backend name."""

    return {
        backend: runtime_preconditions
        for backend, (_, _, runtime_preconditions) in _BACKEND_RUNTIME_PRECONDITIONS.items()
    }


def collect_backend_readiness(*, backend: str, model_dir: Path | None) -> BackendReadiness:
    """Collect deterministic install/runtime readiness diagnostics for one backend."""

    if backend not in _BACKEND_RUNTIME_PRECONDITIONS:
        raise ValueError(
            f"Unsupported backend '{backend}' for readiness checks. "
            f"Expected one of: {', '.join(supported_readiness_backends())}"
        )

    install_extra, required_dependencies, _ = _BACKEND_RUNTIME_PRECONDITIONS[backend]
    missing_dependencies = tuple(dep for dep in required_dependencies if find_spec(dep) is None)

    status_by_name = {status.name: status for status in list_backend_status()}
    status = status_by_name.get(backend)
    registry_enabled = bool(status and status.enabled)

    if model_dir is None:
        model_layout_valid = False
        model_layout_error = "model path not provided"
    else:
        try:
            validate_model_artifact_layout(backend_name=backend, model_dir=model_dir)
        except ModelResolutionError as exc:
            model_layout_valid = False
            model_layout_error = str(exc)
        else:
            model_layout_valid = True
            model_layout_error = None

    cuda_checker, cuda_probe_label = select_cuda_probe(backend)
    device_details = resolve_device_with_details(
        "auto",
        cuda_available_checker=cuda_checker,
        cuda_probe_reason_label=cuda_probe_label,
    )

    return BackendReadiness(
        backend=backend,
        install_extra=install_extra,
        required_dependencies=required_dependencies,
        missing_dependencies=missing_dependencies,
        registry_enabled=registry_enabled,
        model_layout_valid=model_layout_valid,
        model_layout_error=model_layout_error,
        cuda_probe_label=cuda_probe_label,
        cuda_preflight_reason=device_details.reason,
    )
