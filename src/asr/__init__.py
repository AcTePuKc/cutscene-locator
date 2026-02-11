"""ASR backend interfaces and implementations."""

from .backends import MockASRBackend, validate_asr_result
from .base import ASRBackend, ASRResult
from .config import ASRConfig, DeviceType
from .device import resolve_device
from .model_resolution import ModelResolutionError, resolve_model_cache_dir, resolve_model_path
from .registry import BackendCapabilities, BackendRegistration, get_backend, list_backends

__all__ = [
    "ASRBackend",
    "ASRResult",
    "MockASRBackend",
    "validate_asr_result",
    "ASRConfig",
    "DeviceType",
    "resolve_device",
    "ModelResolutionError",
    "resolve_model_cache_dir",
    "resolve_model_path",
    "BackendCapabilities",
    "BackendRegistration",
    "get_backend",
    "list_backends",
]
