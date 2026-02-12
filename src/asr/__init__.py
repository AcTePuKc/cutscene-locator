"""ASR backend interfaces and implementations."""

from .backends import MockASRBackend, parse_asr_result, validate_asr_result
from .faster_whisper_backend import FasterWhisperBackend
from .qwen3_asr_backend import Qwen3ASRBackend
from .whisperx_backend import WhisperXBackend
from .base import ASRBackend, ASRResult
from .adapters import (
    ASRAdapter,
    ASRExecutionContext,
    CapabilityRequirements,
    dispatch_asr_transcription,
    get_asr_adapter,
    list_asr_adapters,
)
from .config import ASRConfig, ComputeType, DeviceType
from .device import DeviceResolution, resolve_device, resolve_device_with_details
from .model_resolution import ModelResolutionError, resolve_model_cache_dir, resolve_model_path
from .registry import (
    BackendCapabilities,
    BackendRegistration,
    BackendStatus,
    get_backend,
    list_backends,
    list_backend_status,
    list_declared_backends,
    validate_backend_capabilities,
)

__all__ = [
    "ASRBackend",
    "ASRResult",
    "ASRAdapter",
    "ASRExecutionContext",
    "CapabilityRequirements",
    "dispatch_asr_transcription",
    "get_asr_adapter",
    "list_asr_adapters",
    "MockASRBackend",
    "FasterWhisperBackend",
    "Qwen3ASRBackend",
    "WhisperXBackend",
    "validate_asr_result",
    "parse_asr_result",
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
    "BackendStatus",
    "get_backend",
    "list_backends",
    "list_backend_status",
    "list_declared_backends",
    "validate_backend_capabilities",
]
