"""CLI entrypoint for cutscene-locator (Milestone 1 / Phase 1)."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from src.asr import MockASRBackend
from src.ingest.script_parser import load_script_table
from src.match.engine import match_segments_to_script
from src.scene import reconstruct_scenes

VERSION = "0.0.0"


@dataclass
class CliError(Exception):
    """Represents a fatal, user-facing CLI validation/runtime error."""

    message: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cutscene-locator")
    parser.add_argument("--input", dest="input_path")
    parser.add_argument("--script", dest="script_path")
    parser.add_argument("--out", dest="out_dir")
    parser.add_argument("--asr-backend", default="mock")
    parser.add_argument("--mock-asr", dest="mock_asr_path")
    parser.add_argument("--chunk", type=int, default=300)
    parser.add_argument("--scene-gap", type=int, default=10)
    parser.add_argument("--ffmpeg-path")
    parser.add_argument("--keep-wav", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--version", action="store_true")
    parser.add_argument("--match-threshold", type=float, default=0.85)
    return parser


def _validate_required_args(args: argparse.Namespace) -> None:
    missing: list[str] = []
    if not args.input_path:
        missing.append("--input")
    if not args.script_path:
        missing.append("--script")
    if not args.out_dir:
        missing.append("--out")

    if missing:
        raise CliError(f"Missing required arguments: {', '.join(missing)}")


def _validate_backend(args: argparse.Namespace) -> None:
    if args.asr_backend != "mock":
        raise CliError("Invalid ASR backend. Only 'mock' is supported in this phase.")
    if not args.mock_asr_path:
        raise CliError("--mock-asr is required when --asr-backend mock is used.")


def resolve_ffmpeg_binary(
    ffmpeg_path: str | None,
    which: Callable[[str], str | None] = shutil.which,
) -> str:
    if ffmpeg_path:
        return ffmpeg_path

    resolved = which("ffmpeg")
    if not resolved:
        raise CliError(
            "ffmpeg not found. Install ffmpeg or provide an explicit path with --ffmpeg-path."
        )
    return resolved


def run_ffmpeg_preflight(
    ffmpeg_binary: str,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> None:
    try:
        runner(
            [ffmpeg_binary, "-version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise CliError(
            f"ffmpeg executable not found at '{ffmpeg_binary}'. Provide a valid --ffmpeg-path."
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        details = f" Details: {stderr}" if stderr else ""
        raise CliError(f"ffmpeg preflight failed for '{ffmpeg_binary}'.{details}") from exc


def main(
    argv: Sequence[str] | None = None,
    *,
    which: Callable[[str], str | None] = shutil.which,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> int:
    parser = build_parser()

    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        # argparse uses exit code 2 for parse failures; normalize to 1 for fatal CLI errors.
        return 0 if exc.code == 0 else 1

    if args.version:
        print(f"cutscene-locator {VERSION}")
        return 0

    try:
        _validate_required_args(args)
        _validate_backend(args)
        ffmpeg_binary = resolve_ffmpeg_binary(args.ffmpeg_path, which=which)
        run_ffmpeg_preflight(ffmpeg_binary, runner=runner)
        script_table = load_script_table(Path(args.script_path))
        asr_backend = MockASRBackend(Path(args.mock_asr_path))
        asr_result = asr_backend.run()
        matching_output = match_segments_to_script(
            asr_result=asr_result,
            script_table=script_table,
            low_confidence_threshold=args.match_threshold,
        )
        scene_output = reconstruct_scenes(
            matching_output=matching_output,
            scene_gap_seconds=float(args.scene_gap),
        )
    except (CliError, ValueError) as exc:
        message = exc.message if isinstance(exc, CliError) else str(exc)
        print(f"Error: {message}", file=sys.stderr)
        return 1

    if args.verbose:
        print("Verbose: CLI validation, ffmpeg preflight, and script ingestion completed.")
        print(f"Verbose: ffmpeg binary: {ffmpeg_binary}")
        print(f"Verbose: input={Path(args.input_path)} script={Path(args.script_path)} out={Path(args.out_dir)}")
        print(f"Verbose: script rows loaded={len(script_table.rows)} delimiter={repr(script_table.delimiter)}")
        low_confidence_count = sum(1 for match in matching_output.matches if match.low_confidence)
        print(f"Verbose: asr backend={asr_result['meta']['backend']} segments={len(asr_result['segments'])}")
        print(
            "Verbose: matches computed="
            f"{len(matching_output.matches)} low_confidence={low_confidence_count} threshold={args.match_threshold}"
        )
        print(f"Verbose: scenes reconstructed={len(scene_output['scenes'])} gap_seconds={float(args.scene_gap)}")

    print("Preflight checks passed. Script ingestion, ASR validation, matching, and scene reconstruction completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
