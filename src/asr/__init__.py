"""ASR backend interfaces and implementations."""

from .backends import ASRBackend, ASRResult, MockASRBackend, validate_asr_result
from .config import ASRConfig, DeviceType
from .model_resolution import ModelResolutionError, resolve_model_cache_dir, resolve_model_path
from .registry import BackendCapabilities, BackendRegistration, get_backend, list_backends

__all__ = [
    "ASRBackend",
    "ASRResult",
    "MockASRBackend",
    "validate_asr_result",
    "ASRConfig",
    "DeviceType",
    "ModelResolutionError",
    "resolve_model_cache_dir",
    "resolve_model_path",
    "BackendCapabilities",
    "BackendRegistration",
    "get_backend",
    "list_backends",
]
