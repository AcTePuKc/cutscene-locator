"""Subprocess worker entrypoint for isolated ASR execution."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .backends import parse_asr_result
from .config import ASRConfig
from .faster_whisper_backend import FasterWhisperBackend


def _configure_runtime_environment(*, device: str, verbose: bool = False) -> None:
    """Set worker-local progress env guards before ASR backend imports."""

    del device, verbose
    os.environ["TQDM_DISABLE"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.asr.asr_worker")
    parser.add_argument("--audio-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", required=True, choices=("cpu", "cuda", "auto"))
    parser.add_argument("--compute-type", required=True, choices=("float16", "float32", "auto"))
    parser.add_argument("--result-path", required=True)
    parser.add_argument("--verbose", action="store_true")
    return parser


def _print_verbose_environment_dump() -> None:
    """Print diagnostic runtime details for worker environment parity checks."""

    import ctranslate2

    print(f"asr-worker: sys.executable={sys.executable}")
    print(f"asr-worker: sys.path[0:3]={sys.path[0:3]}")
    print(f"asr-worker: ctranslate2.__file__={getattr(ctranslate2, '__file__', 'unknown')}")
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
    first_segment = next(raw_segments, None)
    if first_segment is not None:
        print("asr-worker: minimal preflight first segment observed")
    print("asr-worker: minimal preflight transcribe end")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_runtime_environment(device=args.device, verbose=args.verbose)

    backend = FasterWhisperBackend()

    if args.verbose:
        _print_verbose_environment_dump()
        if args.device == "cuda":
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
    try:
        result = backend.transcribe(
            audio_path=args.audio_path,
            config=ASRConfig(
                backend_name="faster-whisper",
                model_path=Path(args.model_path),
                device=args.device,
                compute_type=args.compute_type,
                log_callback=print if args.verbose else None,
            ),
        )
    except Exception:
        print("worker step failed: backend.transcribe", flush=True)
        raise
    print("asr-worker: backend.transcribe end", flush=True)

    try:
        serialized_result = parse_asr_result(result, source="faster-whisper worker")
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
