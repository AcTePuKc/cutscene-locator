"""ASR backend interfaces and implementations."""

from .backends import ASRBackend, ASRResult, MockASRBackend, validate_asr_result
from .config import ASRConfig, DeviceType
from .registry import BackendCapabilities, BackendRegistration, get_backend, list_backends

__all__ = [
    "ASRBackend",
    "ASRResult",
    "MockASRBackend",
    "validate_asr_result",
    "ASRConfig",
    "DeviceType",
    "BackendCapabilities",
    "BackendRegistration",
    "get_backend",
    "list_backends",
]
