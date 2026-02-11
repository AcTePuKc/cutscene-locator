"""ASR configuration contract used by CLI/backend plumbing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

DeviceType = Literal["cpu", "cuda", "auto"]


@dataclass(frozen=True)
class ASRConfig:
    """ASR backend selection and runtime options."""

    backend_name: str
    model_path: Path | None = None
    model_id: str | None = None
    revision: str | None = None
    auto_download: str | None = None
    device: DeviceType = "auto"
    language: str | None = None
    ffmpeg_path: str | None = None
    progress_callback: Callable[[float], None] | None = None
    cancel_check: Callable[[], bool] | None = None
