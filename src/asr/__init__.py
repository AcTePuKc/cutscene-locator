"""ASR backend interfaces and implementations."""

from .backends import MockASRBackend, parse_asr_result, validate_asr_result
from .faster_whisper_backend import FasterWhisperBackend
from .qwen3_asr_backend import Qwen3ASRBackend
from .whisperx_backend import WhisperXBackend
from .vibevoice_backend import VibeVoiceBackend
from .base import ASRBackend, ASRResult
from .adapters import (
    ASRAdapter,
    ASRExecutionContext,
    CapabilityRequirements,
    dispatch_asr_transcription,
    apply_cross_chunk_continuity,
    get_asr_adapter,
    list_asr_adapters,
)
from .config import ASRConfig, ComputeType, DeviceType
from .device import DeviceResolution, resolve_device, resolve_device_with_details
from .model_resolution import ModelResolutionError, resolve_model_cache_dir, resolve_model_path
from .readiness import (
    BackendReadiness,
    backend_runtime_preconditions,
    collect_backend_readiness,
    supported_readiness_backends,
)
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
    "apply_cross_chunk_continuity",
    "get_asr_adapter",
    "list_asr_adapters",
    "MockASRBackend",
    "FasterWhisperBackend",
    "Qwen3ASRBackend",
    "WhisperXBackend",
    "VibeVoiceBackend",
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
    "supported_readiness_backends",
    "collect_backend_readiness",
    "backend_runtime_preconditions",
    "BackendReadiness",
    "get_backend",
    "list_backends",
    "list_backend_status",
    "list_declared_backends",
    "validate_backend_capabilities",
]
