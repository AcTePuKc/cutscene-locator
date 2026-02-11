"""ffmpeg-based preprocessing for canonical WAV conversion and chunking."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class ChunkMetadata:
    source_filename: str
    chunk_index: int
    absolute_offset_seconds: float
    chunk_wav_path: Path


@dataclass(frozen=True)
class PreprocessResult:
    canonical_wav_path: Path
    chunk_metadata: list[ChunkMetadata]
    tmp_dir: Path
    audio_tmp_dir: Path
    chunks_tmp_dir: Path


def _sanitize_stem(path: Path) -> str:
    stem = path.stem.strip().lower()
    stem = re.sub(r"[^a-z0-9._-]+", "_", stem)
    return stem or "input"


def prepare_tmp_workspace(out_dir: Path) -> tuple[Path, Path, Path]:
    tmp_dir = out_dir / "_tmp"
    audio_tmp_dir = tmp_dir / "audio"
    chunks_tmp_dir = tmp_dir / "chunks"
    audio_tmp_dir.mkdir(parents=True, exist_ok=True)
    chunks_tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir, audio_tmp_dir, chunks_tmp_dir


def _run_ffmpeg_command(
    command: list[str],
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> None:
    try:
        runner(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise ValueError("ffmpeg executable not found while running preprocessing.") from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        details = f" Details: {stderr}" if stderr else ""
        raise ValueError(f"ffmpeg preprocessing failed.{details}") from exc


def convert_to_canonical_wav(
    *,
    ffmpeg_binary: str,
    input_path: Path,
    audio_tmp_dir: Path,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> Path:
    canonical_wav_path = audio_tmp_dir / f"{_sanitize_stem(input_path)}_canonical.wav"
    command = [
        ffmpeg_binary,
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(canonical_wav_path),
    ]
    _run_ffmpeg_command(command, runner=runner)
    return canonical_wav_path


def chunk_canonical_wav(
    *,
    ffmpeg_binary: str,
    canonical_wav_path: Path,
    chunks_tmp_dir: Path,
    chunk_seconds: int,
    source_filename: str,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> list[ChunkMetadata]:
    if chunk_seconds == 0:
        return [
            ChunkMetadata(
                source_filename=source_filename,
                chunk_index=0,
                absolute_offset_seconds=0.0,
                chunk_wav_path=canonical_wav_path,
            )
        ]

    chunk_pattern = chunks_tmp_dir / f"{canonical_wav_path.stem}_chunk_%06d.wav"
    command = [
        ffmpeg_binary,
        "-y",
        "-i",
        str(canonical_wav_path),
        "-f",
        "segment",
        "-segment_time",
        str(chunk_seconds),
        "-c",
        "copy",
        str(chunk_pattern),
    ]
    _run_ffmpeg_command(command, runner=runner)

    chunk_paths = sorted(chunks_tmp_dir.glob(f"{canonical_wav_path.stem}_chunk_*.wav"))
    if not chunk_paths:
        raise ValueError("ffmpeg chunking produced no output files.")

    return [
        ChunkMetadata(
            source_filename=source_filename,
            chunk_index=index,
            absolute_offset_seconds=float(index * chunk_seconds),
            chunk_wav_path=chunk_path,
        )
        for index, chunk_path in enumerate(chunk_paths)
    ]


def preprocess_media(
    *,
    ffmpeg_binary: str,
    input_path: Path,
    out_dir: Path,
    chunk_seconds: int,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> PreprocessResult:
    if chunk_seconds < 0:
        raise ValueError("--chunk must be >= 0.")

    tmp_dir, audio_tmp_dir, chunks_tmp_dir = prepare_tmp_workspace(out_dir)
    canonical_wav_path = convert_to_canonical_wav(
        ffmpeg_binary=ffmpeg_binary,
        input_path=input_path,
        audio_tmp_dir=audio_tmp_dir,
        runner=runner,
    )
    chunk_metadata = chunk_canonical_wav(
        ffmpeg_binary=ffmpeg_binary,
        canonical_wav_path=canonical_wav_path,
        chunks_tmp_dir=chunks_tmp_dir,
        chunk_seconds=chunk_seconds,
        source_filename=input_path.name,
        runner=runner,
    )

    return PreprocessResult(
        canonical_wav_path=canonical_wav_path,
        chunk_metadata=chunk_metadata,
        tmp_dir=tmp_dir,
        audio_tmp_dir=audio_tmp_dir,
        chunks_tmp_dir=chunks_tmp_dir,
    )
