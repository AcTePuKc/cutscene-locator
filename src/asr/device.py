"""Deterministic ASR device resolution helpers."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Callable, Literal

from .config import DeviceType

ResolvedDevice = Literal["cpu", "cuda"]


@dataclass(frozen=True)
class DeviceResolution:
    """Resolved device plus a deterministic explanation."""

    requested: DeviceType
    resolved: ResolvedDevice
    reason: str


def _cuda_probe_ctranslate2() -> tuple[bool, str]:
    """Probe CUDA support via ctranslate2 without requiring torch."""

    try:
        ctranslate2 = import_module("ctranslate2")
    except Exception:
        return False, "ctranslate2 is not installed"

    cuda_count_getter = getattr(ctranslate2, "get_cuda_device_count", None)
    if not callable(cuda_count_getter):
        return False, "installed ctranslate2 build does not expose CUDA runtime support"

    try:
        raw_cuda_device_count: object = cuda_count_getter()
    except Exception as exc:
        return False, f"ctranslate2 CUDA check failed: {exc}"

    if isinstance(raw_cuda_device_count, bool):
        cuda_device_count = int(raw_cuda_device_count)
    elif isinstance(raw_cuda_device_count, int):
        cuda_device_count = raw_cuda_device_count
    elif isinstance(raw_cuda_device_count, float | str):
        try:
            cuda_device_count = int(raw_cuda_device_count)
        except (TypeError, ValueError):
            return (
                False,
                "ctranslate2 CUDA check returned a non-numeric device count: "
                f"{raw_cuda_device_count!r}",
            )
    else:
        return (
            False,
            "ctranslate2 CUDA check returned a non-numeric device count: "
            f"{raw_cuda_device_count!r}",
        )

    if cuda_device_count > 0:
        return True, f"ctranslate2 detected {cuda_device_count} CUDA device(s)"
    return False, "ctranslate2 detected 0 CUDA devices"


def _is_cuda_available() -> bool:
    """Return CUDA availability without requiring torch as a hard dependency."""

    try:
        torch = import_module("torch")
    except Exception:
        return False

    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def resolve_device(
    requested_device: DeviceType,
    *,
    cuda_available_checker: Callable[[], bool] | None = None,
) -> ResolvedDevice:
    """Resolve configured ASR device deterministically."""

    return resolve_device_with_details(
        requested_device,
        cuda_available_checker=cuda_available_checker,
    ).resolved


def resolve_device_with_details(
    requested_device: DeviceType,
    *,
    cuda_available_checker: Callable[[], bool] | None = None,
) -> DeviceResolution:
    """Resolve configured ASR device and return deterministic reasoning."""

    if requested_device == "cpu":
        return DeviceResolution(
            requested=requested_device,
            resolved="cpu",
            reason="explicit --device cpu request",
        )

    if cuda_available_checker is None:
        cuda_available, availability_reason = _cuda_probe_ctranslate2()
    else:
        cuda_available = cuda_available_checker()
        availability_reason = "custom CUDA availability probe"

    if requested_device == "cuda":
        if not cuda_available:
            raise ValueError(
                "Requested --device cuda, but CUDA is unavailable. "
                f"Reason: {availability_reason}. "
                "For faster-whisper, install a CUDA-enabled ctranslate2 build, verify NVIDIA driver/CUDA runtime, "
                "or rerun with --device cpu. See docs/CUDA.md for checks and Windows notes."
            )
        return DeviceResolution(
            requested=requested_device,
            resolved="cuda",
            reason=f"explicit --device cuda request; {availability_reason}",
        )

    if requested_device == "auto":
        if cuda_available:
            return DeviceResolution(
                requested=requested_device,
                resolved="cuda",
                reason=f"--device auto selected cuda because {availability_reason}",
            )
        return DeviceResolution(
            requested=requested_device,
            resolved="cpu",
            reason=f"--device auto selected cpu because {availability_reason}",
        )

    raise ValueError(f"Unsupported device '{requested_device}'. Expected cpu, cuda, or auto.")
