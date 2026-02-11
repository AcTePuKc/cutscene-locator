"""ASR backend interfaces and implementations."""

from .backends import MockASRBackend, validate_asr_result
from .faster_whisper_backend import FasterWhisperBackend
from .qwen3_asr_backend import Qwen3ASRBackend
from .base import ASRBackend, ASRResult
from .config import ASRConfig, ComputeType, DeviceType
from .device import DeviceResolution, resolve_device, resolve_device_with_details
from .model_resolution import ModelResolutionError, resolve_model_cache_dir, resolve_model_path
from .registry import BackendCapabilities, BackendRegistration, get_backend, list_backends

__all__ = [
    "ASRBackend",
    "ASRResult",
    "MockASRBackend",
    "FasterWhisperBackend",
    "Qwen3ASRBackend",
    "validate_asr_result",
    "ASRConfig",
    "ComputeType",
    "DeviceType",
    "DeviceResolution",
    "resolve_device",
    "resolve_device_with_details",
    "ModelResolutionError",
    "resolve_model_cache_dir",
    "resolve_model_path",
    "BackendCapabilities",
    "BackendRegistration",
    "get_backend",
    "list_backends",
]
