"""Deterministic ASR device resolution helpers."""

from __future__ import annotations

from importlib import import_module
from typing import Callable, Literal

from .config import DeviceType

ResolvedDevice = Literal["cpu", "cuda"]


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
    cuda_available_checker: Callable[[], bool] = _is_cuda_available,
) -> ResolvedDevice:
    """Resolve configured ASR device deterministically."""

    if requested_device == "cpu":
        return "cpu"

    cuda_available = cuda_available_checker()
    if requested_device == "cuda":
        if not cuda_available:
            raise ValueError(
                "Requested --device cuda, but CUDA is unavailable. "
                "Install a CUDA-enabled runtime or rerun with --device cpu."
            )
        return "cuda"

    if requested_device == "auto":
        return "cuda" if cuda_available else "cpu"

    raise ValueError(f"Unsupported device '{requested_device}'. Expected cpu, cuda, or auto.")
