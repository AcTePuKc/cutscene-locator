"""Subprocess worker entrypoint for isolated ASR execution."""

from __future__ import annotations

import argparse
import json
import os
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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_runtime_environment(device=args.device)

    asr_module = import_module("src.asr")
    asr_config_class = asr_module.ASRConfig
    backend_class = asr_module.FasterWhisperBackend
    parse_asr_result = asr_module.parse_asr_result

    backend = backend_class()

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
