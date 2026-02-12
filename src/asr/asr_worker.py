"""Subprocess worker entrypoint for isolated ASR execution."""

from __future__ import annotations

import argparse
import json
import os
import sys
from importlib import import_module
from pathlib import Path


def _configure_runtime_environment(*, device: str) -> None:
    """Set worker-local progress env guards before ASR backend imports."""

    if device != "cuda":
        return
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

    ctranslate2_module = import_module("ctranslate2")
    print(f"asr-worker: sys.executable={sys.executable}")
    print(f"asr-worker: sys.path[0:3]={sys.path[0:3]}")
    print(f"asr-worker: ctranslate2.__file__={getattr(ctranslate2_module, '__file__', 'unknown')}")
    env_subset = {
        "PATH": os.environ.get("PATH"),
        "CUDA_PATH": os.environ.get("CUDA_PATH"),
        "CUDNN_PATH": os.environ.get("CUDNN_PATH"),
    }
    print(f"asr-worker: env={env_subset}")


def _run_minimal_whisper_preflight(*, audio_path: str, model_path: Path, device: str, compute_type: str) -> None:
    """Run a minimal direct faster-whisper call in-worker before backend execution."""

    faster_whisper_module = import_module("faster_whisper")
    whisper_model_class = getattr(faster_whisper_module, "WhisperModel", None)
    if whisper_model_class is None:
        raise ValueError("Installed faster-whisper package is missing WhisperModel.")

    model = whisper_model_class(str(model_path), device=device, compute_type=compute_type)
    print("asr-worker: minimal preflight transcribe start")
    raw_segments, _info = model.transcribe(audio_path)
    segment_count = sum(1 for _ in raw_segments)
    print(f"asr-worker: minimal preflight transcribe end; segments={segment_count}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_runtime_environment(device=args.device)

    asr_module = import_module("src.asr")
    asr_config_class = asr_module.ASRConfig
    backend_class = asr_module.FasterWhisperBackend
    parse_asr_result = asr_module.parse_asr_result

    backend = backend_class()

    if args.verbose:
        _print_verbose_environment_dump()
        _run_minimal_whisper_preflight(
            audio_path=args.audio_path,
            model_path=Path(args.model_path),
            device=args.device,
            compute_type=args.compute_type,
        )

    result = backend.transcribe(
        audio_path=args.audio_path,
        config=asr_config_class(
            backend_name="faster-whisper",
            model_path=Path(args.model_path),
            device=args.device,
            compute_type=args.compute_type,
            log_callback=print if args.verbose else None,
        ),
    )
    serialized_result = parse_asr_result(result, source="faster-whisper worker")

    result_path = Path(args.result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(serialized_result, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
