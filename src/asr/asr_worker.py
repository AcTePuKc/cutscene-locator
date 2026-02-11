"""Subprocess worker entrypoint for isolated ASR execution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import ASRConfig, FasterWhisperBackend


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.asr.asr_worker")
    parser.add_argument("--audio-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", required=True, choices=("cpu", "cuda", "auto"))
    parser.add_argument("--compute-type", required=True, choices=("float16", "float32", "auto"))
    parser.add_argument("--progress", required=True, choices=("on", "off"))
    parser.add_argument("--result-path", required=True)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    backend = FasterWhisperBackend()

    result = backend.transcribe(
        audio_path=args.audio_path,
        config=ASRConfig(
            backend_name="faster-whisper",
            model_path=Path(args.model_path),
            device=args.device,
            compute_type=args.compute_type,
            download_progress=(args.progress == "on"),
            log_callback=print if args.verbose else None,
        ),
    )

    result_path = Path(args.result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
