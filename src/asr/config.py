"""ASR configuration contract used by CLI/backend plumbing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

DeviceType = Literal["cpu", "cuda", "auto"]
ComputeType = Literal["float16", "float32", "auto"]


@dataclass(frozen=True)
class ASRConfig:
    """ASR backend selection and runtime options."""

    backend_name: str
    model_path: Path | None = None
    model_id: str | None = None
    revision: str | None = None
    auto_download: str | None = None
    device: DeviceType = "auto"
    compute_type: ComputeType = "auto"
    language: str | None = None
    beam_size: int = 1
    temperature: float = 0.0
    best_of: int = 1
    no_speech_threshold: float | None = None
    log_prob_threshold: float | None = None
    qwen3_batch_size: int | None = None
    qwen3_chunk_length_s: float | None = None
    vad_filter: bool = False
    merge_short_segments_seconds: float = 0.0
    ffmpeg_path: str | None = None
    download_progress: bool | None = None
    progress_callback: Callable[[float], None] | None = None
    cancel_check: Callable[[], bool] | None = None
    log_callback: Callable[[str], None] | None = None
