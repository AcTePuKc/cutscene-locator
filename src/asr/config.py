"""ASR configuration contract used by CLI/backend plumbing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

DeviceType = Literal["cpu", "cuda", "auto"]


@dataclass(frozen=True)
class ASRConfig:
    """ASR backend selection and runtime options."""

    backend_name: str
    model_path: Path | None = None
    auto_download: str | None = None
    device: DeviceType = "auto"
    language: str | None = None
    ffmpeg_path: str | None = None
