"""Subprocess worker entrypoint for isolated ASR execution."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .backends import parse_asr_result
from .base import ASRBackend, ASRResult
from .config import ASRConfig
from .faster_whisper_backend import FasterWhisperBackend
from .qwen3_asr_backend import Qwen3ASRBackend
from .vibevoice_backend import VibeVoiceBackend
from .whisperx_backend import WhisperXBackend


_WORKER_RUNTIME_BACKENDS: tuple[str, ...] = (
    "faster-whisper",
    "qwen3-asr",
    "whisperx",
    "vibevoice",
)


def _build_runtime_backend(backend_name: str) -> ASRBackend:
    if backend_name == "faster-whisper":
        return FasterWhisperBackend()
    if backend_name == "qwen3-asr":
        return Qwen3ASRBackend()
    if backend_name == "whisperx":
        return WhisperXBackend()
    if backend_name == "vibevoice":
        return VibeVoiceBackend()

    supported = ", ".join(_WORKER_RUNTIME_BACKENDS)
    raise ValueError(
        f"Unsupported --asr-backend '{backend_name}' for ASR worker. "
        f"Expected one of: {supported}."
    )


def _configure_runtime_environment(*, device: str, verbose: bool = False) -> None:
    """Set worker-local progress env guards before ASR backend imports."""

    del device, verbose
    os.environ["TQDM_DISABLE"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.asr.asr_worker")
    parser.add_argument("--asr-backend", required=True, choices=_WORKER_RUNTIME_BACKENDS)
    parser.add_argument("--audio-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", required=True, choices=("cpu", "cuda", "auto"))
    parser.add_argument("--compute-type", required=True, choices=("float16", "float32", "auto"))
    parser.add_argument("--result-path", required=True)
    parser.add_argument("--asr-language", default=None)
    parser.add_argument("--asr-beam-size", type=int, default=1)
    parser.add_argument("--asr-temperature", type=float, default=0.0)
    parser.add_argument("--asr-best-of", type=int, default=1)
    parser.add_argument("--qwen3-batch-size", type=int, default=None)
    parser.add_argument("--qwen3-chunk-length-s", type=float, default=None)
    parser.add_argument("--asr-no-speech-threshold", type=float, default=None)
    parser.add_argument("--asr-logprob-threshold", type=float, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser


def _build_runtime_asr_config(args: argparse.Namespace) -> ASRConfig:
    beam_size = args.asr_beam_size if args.asr_beam_size is not None else 1
    temperature = args.asr_temperature if args.asr_temperature is not None else 0.0
    best_of = args.asr_best_of if args.asr_best_of is not None else 1

    common_kwargs = {
        "backend_name": args.asr_backend,
        "model_path": Path(args.model_path),
        "device": args.device,
        "compute_type": args.compute_type,
        "language": args.asr_language,
        "log_callback": print if args.verbose else None,
    }
    if args.asr_backend == "faster-whisper":
        return ASRConfig(
            **common_kwargs,
            beam_size=beam_size,
            temperature=temperature,
            best_of=best_of,
            no_speech_threshold=args.asr_no_speech_threshold,
            log_prob_threshold=args.asr_logprob_threshold,
        )
    if args.asr_backend == "qwen3-asr":
        return ASRConfig(
            **common_kwargs,
            beam_size=beam_size,
            temperature=temperature,
            best_of=best_of,
            qwen3_batch_size=args.qwen3_batch_size,
            qwen3_chunk_length_s=args.qwen3_chunk_length_s,
        )
    if args.asr_backend in {"whisperx", "vibevoice"}:
        return ASRConfig(
            **common_kwargs,
            beam_size=beam_size,
            temperature=temperature,
            best_of=best_of,
        )

    supported = ", ".join(_WORKER_RUNTIME_BACKENDS)
    raise ValueError(
        f"Unsupported --asr-backend '{args.asr_backend}' for ASR worker. "
        f"Expected one of: {supported}."
    )


def _print_verbose_environment_dump(*, backend_name: str) -> None:
    """Print diagnostic runtime details for worker environment parity checks."""

    print(f"asr-worker: sys.executable={sys.executable}")
    print(f"asr-worker: sys.path[0:3]={sys.path[0:3]}")
    if backend_name == "faster-whisper":
        import ctranslate2

        print(f"asr-worker: ctranslate2.__file__={getattr(ctranslate2, '__file__', 'unknown')}")
    else:
        print(f"asr-worker: backend_runtime={backend_name}")
    env_subset = {
        "PATH": os.environ.get("PATH"),
        "CUDA_PATH": os.environ.get("CUDA_PATH"),
        "CUDNN_PATH": os.environ.get("CUDNN_PATH"),
    }
    print(f"asr-worker: env={env_subset}")


def _run_minimal_whisper_preflight(*, audio_path: str, model_path: Path, device: str, compute_type: str) -> None:
    """Run a minimal direct faster-whisper call in-worker before backend execution."""

    from faster_whisper import WhisperModel

    model = WhisperModel(str(model_path), device=device, compute_type=compute_type)
    print("asr-worker: minimal preflight transcribe start")
    raw_segments, _info = model.transcribe(audio_path, vad_filter=False)
    it = iter(raw_segments)
    first_segment = next(it, None)


    if first_segment is not None:
        print("asr-worker: minimal preflight first segment observed")
    print("asr-worker: minimal preflight transcribe end")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.asr_backend not in _WORKER_RUNTIME_BACKENDS:
        supported = ", ".join(_WORKER_RUNTIME_BACKENDS)
        raise ValueError(
            f"Unsupported --asr-backend '{args.asr_backend}' for ASR worker. "
            f"Expected one of: {supported}."
        )

    _configure_runtime_environment(device=args.device, verbose=args.verbose)
    backend = _build_runtime_backend(args.asr_backend)

    if args.verbose:
        _print_verbose_environment_dump(backend_name=args.asr_backend)
        if args.asr_backend != "faster-whisper":
            print(f"asr-worker: minimal preflight skipped for backend={args.asr_backend}")
        elif args.device == "cuda":
            print("asr-worker: minimal preflight skipped on cuda")
        else:
            try:
                _run_minimal_whisper_preflight(
                    audio_path=args.audio_path,
                    model_path=Path(args.model_path),
                    device=args.device,
                    compute_type=args.compute_type,
                )
            except Exception as exc:
                print(f"asr-worker: warning: minimal preflight failed; continuing: {exc}")

    print("asr-worker: backend.transcribe begin", flush=True)
    runtime_config = _build_runtime_asr_config(args)

    try:
        result: ASRResult = backend.transcribe(
            audio_path=args.audio_path,
            config=runtime_config,
        )
    except Exception:
        print("worker step failed: backend.transcribe", flush=True)
        raise
    print("asr-worker: backend.transcribe end", flush=True)

    try:
        serialized_result = parse_asr_result(result, source=f"{args.asr_backend} worker")
    except Exception:
        print("worker step failed: parse_asr_result", flush=True)
        raise
    print("asr-worker: parse_asr_result end", flush=True)

    result_path = Path(args.result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        result_path.write_text(json.dumps(serialized_result, ensure_ascii=False), encoding="utf-8")
    except Exception:
        print("worker step failed: write_result_json", flush=True)
        raise
    print("asr-worker: write_result_json end", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
