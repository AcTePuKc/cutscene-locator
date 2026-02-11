"""CLI entrypoint for cutscene-locator (Milestone 1 / Phase 1)."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import json
import tempfile
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from src.asr import (
    ASRConfig,
    ASRResult,
    FasterWhisperBackend,
    MockASRBackend,
    Qwen3ASRBackend,
    get_backend,
    parse_asr_result,
    resolve_device_with_details,
)
from src.asr.model_resolution import ModelResolutionError, resolve_model_path
from src.export import (
    write_matches_csv,
    write_scenes_json,
    write_subs_qa_srt,
    write_subs_target_srt,
)
from src.ingest import load_script_table, preprocess_media
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
    parser.add_argument("--model-path")
    parser.add_argument("--model-id")
    parser.add_argument("--revision")
    parser.add_argument("--auto-download")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--compute-type", default="auto")
    parser.add_argument("--chunk", type=int, default=300)
    parser.add_argument("--scene-gap", type=int, default=10)
    parser.add_argument("--ffmpeg-path")
    parser.add_argument("--keep-wav", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--version", action="store_true")
    parser.add_argument("--match-threshold", type=float, default=0.85)
    parser.add_argument("--progress", choices=("on", "off"), default=None)
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
    try:
        registration = get_backend(args.asr_backend)
    except ValueError as exc:
        raise CliError(str(exc)) from exc

    if registration.name == "mock" and not args.mock_asr_path:
        raise CliError("--mock-asr is required when --asr-backend mock is used.")




def _resolve_progress_mode(progress: str | None) -> str:
    if progress is not None:
        return progress
    return "off" if os.name == "nt" else "on"


def _apply_windows_progress_guard() -> None:
    """Disable progress-bar monitor threads by default on Windows."""

    if os.name != "nt":
        return
    os.environ["TQDM_DISABLE"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

def _validate_asr_options(args: argparse.Namespace) -> None:
    allowed_devices = {"cpu", "cuda", "auto"}
    if args.device not in allowed_devices:
        raise CliError("Invalid --device value. Expected one of: cpu, cuda, auto.")

    allowed_compute_types = {"float16", "float32", "auto"}
    if args.compute_type not in allowed_compute_types:
        raise CliError("Invalid --compute-type value. Expected one of: float16, float32, auto.")

    if args.auto_download is not None:
        allowed_model_sizes = {"tiny", "base", "small"}
        if args.auto_download not in allowed_model_sizes:
            raise CliError("Invalid --auto-download value. Expected one of: tiny, base, small.")

    if args.model_id is not None and not str(args.model_id).strip():
        raise CliError("Invalid --model-id value. Expected a non-empty Hugging Face repo id.")

    if args.revision is not None and args.model_id is None:
        raise CliError("--revision requires --model-id.")

    selected_model_source_flags = [
        args.model_path is not None,
        args.model_id is not None,
        args.auto_download is not None,
    ]
    if sum(1 for selected in selected_model_source_flags if selected) > 1:
        raise CliError("Use only one of: --model-path, --model-id, or --auto-download.")


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


def _run_faster_whisper_subprocess(
    *,
    audio_path: Path,
    resolved_model_path: Path,
    asr_config: ASRConfig,
    verbose: bool,
) -> ASRResult:
    with tempfile.TemporaryDirectory(prefix="cutscene_locator_asr_") as temp_dir:
        result_path = Path(temp_dir) / "asr_result.json"
        cmd = [
            sys.executable,
            "-m",
            "src.asr.asr_worker",
            "--audio-path",
            str(audio_path),
            "--model-path",
            str(resolved_model_path),
            "--device",
            asr_config.device,
            "--compute-type",
            asr_config.compute_type,
            "--result-path",
            str(result_path),
        ]
        if verbose:
            cmd.append("--verbose")

        completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if completed.stdout:
            print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, file=sys.stderr, end="")

        if completed.returncode != 0:
            model_ref = str(asr_config.model_id or resolved_model_path)
            context = (
                f"device={asr_config.device} compute_type={asr_config.compute_type} "
                f"backend={asr_config.backend_name} model={model_ref}"
            )
            if completed.returncode in {-1073740791, 3221226505}:
                raise CliError(
                    "GPU backend aborted with a native CUDA crash in ASR worker. "
                    "Try --compute-type float32; check ctranslate2 CUDA wheel compatibility. "
                    f"Context: {context}."
                )
            raise CliError(
                f"ASR worker failed with exit code {completed.returncode}. Context: {context}."
            )

        if not result_path.exists():
            raise CliError("ASR worker did not produce result output.")

        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise CliError("ASR worker result must be a JSON object.")
        return parse_asr_result(payload, source="faster-whisper worker")


def _read_ctranslate2_cuda_device_count() -> int | None:
    try:
        ctranslate2_module = importlib.import_module("ctranslate2")
    except ModuleNotFoundError:
        return None

    get_device_count = getattr(ctranslate2_module, "get_cuda_device_count", None)
    if get_device_count is None:
        return None

    try:
        return int(get_device_count())
    except (TypeError, ValueError, RuntimeError):
        return None


def _read_ctranslate2_version() -> str:
    try:
        ctranslate2_module = importlib.import_module("ctranslate2")
    except ModuleNotFoundError:
        return "not-installed"

    raw_version = getattr(ctranslate2_module, "__version__", None)
    if raw_version is None:
        return "unknown"
    return str(raw_version)


def _print_faster_whisper_cuda_preflight(*, device: str, compute_type: str) -> None:
    cuda_device_count = _read_ctranslate2_cuda_device_count()
    cuda_count_text = str(cuda_device_count) if cuda_device_count is not None else "unknown"
    print(
        "ASR preflight (faster-whisper): "
        f"ctranslate2={_read_ctranslate2_version()} "
        f"cuda_device_count={cuda_count_text} "
        f"device={device} "
        f"compute_type={compute_type}"
    )
    print(
        "ASR preflight guidance: if CUDA ASR aborts, retry with --compute-type float32 first; "
        "then verify torch/ctranslate2 CUDA wheel compatibility (see faster-whisper issue #1086)."
    )


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

    timings: dict[str, float] = {}
    runtime_started = time.perf_counter()
    device_resolution_reason: str | None = None
    model_resolution_logs: list[str] = []

    try:
        _validate_required_args(args)
        _validate_backend(args)
        _apply_windows_progress_guard()
        args.progress = _resolve_progress_mode(args.progress)
        _validate_asr_options(args)
        ffmpeg_binary = resolve_ffmpeg_binary(args.ffmpeg_path, which=which)
        asr_config = ASRConfig(
            backend_name=args.asr_backend,
            model_path=Path(args.model_path) if args.model_path else None,
            model_id=args.model_id,
            revision=args.revision,
            auto_download=args.auto_download,
            device=args.device,
            compute_type=args.compute_type,
            language=None,
            ffmpeg_path=ffmpeg_binary,
            download_progress=(args.progress == "on"),
            log_callback=model_resolution_logs.append,
        )
        run_ffmpeg_preflight(ffmpeg_binary, runner=runner)
        input_path = Path(args.input_path)
        out_dir = Path(args.out_dir)
        preprocess_started = time.perf_counter()
        if args.verbose:
            print("stage: preprocess start")
        preprocessing_output = preprocess_media(
            ffmpeg_binary=ffmpeg_binary,
            input_path=input_path,
            out_dir=out_dir,
            chunk_seconds=args.chunk,
            runner=runner,
        )
        timings["preprocess"] = time.perf_counter() - preprocess_started
        if args.verbose:
            print("stage: preprocess end")
        script_table = load_script_table(Path(args.script_path))

        resolved_model_path: Path | None = None
        should_resolve_model = (
            asr_config.backend_name != "mock"
            or asr_config.model_path is not None
            or asr_config.model_id is not None
            or asr_config.auto_download is not None
        )
        if should_resolve_model:
            try:
                resolved_model_path = resolve_model_path(asr_config)
            except ModelResolutionError as exc:
                raise CliError(str(exc)) from exc

        backend_registration = get_backend(asr_config.backend_name)
        asr_started = time.perf_counter()
        if args.verbose:
            print("stage: asr start")
        if backend_registration.name == "mock":
            asr_backend = MockASRBackend(Path(args.mock_asr_path))
            asr_result = asr_backend.transcribe(str(preprocessing_output.canonical_wav_path), asr_config)
        elif backend_registration.name in {"faster-whisper", "qwen3-asr"}:
            resolution = resolve_device_with_details(asr_config.device)
            device_resolution_reason = resolution.reason
            if backend_registration.name == "faster-whisper":
                asr_backend = FasterWhisperBackend()
            else:
                asr_backend = Qwen3ASRBackend()
            effective_config = ASRConfig(
                backend_name=asr_config.backend_name,
                model_path=resolved_model_path,
                model_id=asr_config.model_id,
                revision=asr_config.revision,
                auto_download=asr_config.auto_download,
                device=asr_config.device,
                compute_type=asr_config.compute_type,
                language=asr_config.language,
                ffmpeg_path=asr_config.ffmpeg_path,
                download_progress=asr_config.download_progress,
                progress_callback=asr_config.progress_callback,
                cancel_check=asr_config.cancel_check,
                log_callback=print if args.verbose else asr_config.log_callback,
            )
            if backend_registration.name == "faster-whisper":
                _print_faster_whisper_cuda_preflight(
                    device=resolution.resolved,
                    compute_type=effective_config.compute_type,
                )
            if backend_registration.name == "faster-whisper" and effective_config.device == "cuda" and os.name == "nt":
                if effective_config.model_path is None:
                    raise CliError("faster-whisper backend requires a resolved model path.")
                asr_result = _run_faster_whisper_subprocess(
                    audio_path=preprocessing_output.canonical_wav_path,
                    resolved_model_path=effective_config.model_path,
                    asr_config=effective_config,
                    verbose=args.verbose,
                )
            else:
                asr_result = asr_backend.transcribe(
                    str(preprocessing_output.canonical_wav_path),
                    effective_config,
                )
        else:
            raise CliError(f"ASR backend '{asr_config.backend_name}' is not implemented yet.")
        timings["asr"] = time.perf_counter() - asr_started
        if args.verbose:
            print("stage: asr end")

        matching_started = time.perf_counter()
        if args.verbose:
            print("stage: matching start")
        matching_output = match_segments_to_script(
            asr_result=asr_result,
            script_table=script_table,
            low_confidence_threshold=args.match_threshold,
            progress_logger=print if args.verbose else None,
            progress_every=50,
        )
        timings["matching"] = time.perf_counter() - matching_started
        if args.verbose:
            print("stage: matching end")

        scene_started = time.perf_counter()
        scene_output = reconstruct_scenes(
            matching_output=matching_output,
            scene_gap_seconds=float(args.scene_gap),
        )
        timings["scene_reconstruction"] = time.perf_counter() - scene_started

        if args.verbose:
            print("stage: exports start")
        write_matches_csv(output_path=out_dir / "matches.csv", matching_output=matching_output)
        write_scenes_json(output_path=out_dir / "scenes.json", scene_output=scene_output)
        write_subs_qa_srt(
            output_path=out_dir / "subs_qa.srt",
            matching_output=matching_output,
            script_table=script_table,
        )
        write_subs_target_srt(
            output_path=out_dir / "subs_target.srt",
            matching_output=matching_output,
            script_table=script_table,
        )
        if args.verbose:
            print("stage: exports end")
        timings["total_runtime"] = time.perf_counter() - runtime_started
    except (CliError, ValueError) as exc:
        message = exc.message if isinstance(exc, CliError) else str(exc)
        print(f"Error: {message}", file=sys.stderr)
        return 1

    if args.verbose:
        print("Verbose: CLI validation, ffmpeg preflight, and script ingestion completed.")
        print(f"Verbose: ffmpeg binary: {ffmpeg_binary}")
        print(f"Verbose: input={input_path} script={Path(args.script_path)} out={out_dir}")
        print(f"Verbose: script rows loaded={len(script_table.rows)} delimiter={repr(script_table.delimiter)}")
        print(f"Verbose: canonical wav={preprocessing_output.canonical_wav_path}")
        print(f"Verbose: chunks generated={len(preprocessing_output.chunk_metadata)} chunk_seconds={args.chunk}")
        low_confidence_count = sum(1 for match in matching_output.matches if match.low_confidence)
        print(
            "Verbose: asr backend="
            f"{asr_result['meta']['backend']} model={asr_result['meta']['model']} "
            f"version={asr_result['meta']['version']} device={asr_result['meta']['device']} "
            f"segments={len(asr_result['segments'])}"
        )
        print(
            "Verbose: asr config="
            f"backend={asr_config.backend_name} requested_device={asr_config.device} compute_type={asr_config.compute_type} "
            f"model_path={resolved_model_path if resolved_model_path is not None else asr_config.model_path} "
            f"model_id={asr_config.model_id} revision={asr_config.revision} "
            f"auto_download={asr_config.auto_download} download_progress={args.progress}"
        )
        for model_resolution_log in model_resolution_logs:
            print(f"Verbose: {model_resolution_log}")
        if device_resolution_reason is not None:
            print(f"Verbose: device resolution reason: {device_resolution_reason}")
        print(
            "Verbose: matches computed="
            f"{len(matching_output.matches)} low_confidence={low_confidence_count} threshold={args.match_threshold}"
        )
        print(f"Verbose: scenes reconstructed={len(scene_output['scenes'])} gap_seconds={float(args.scene_gap)}")
        print(
            "Verbose: timings seconds="
            f"preprocess={timings.get('preprocess', 0.0):.4f} "
            f"asr={timings.get('asr', 0.0):.4f} "
            f"matching={timings.get('matching', 0.0):.4f} "
            f"scene_reconstruction={timings.get('scene_reconstruction', 0.0):.4f} "
            f"total={timings.get('total_runtime', 0.0):.4f}"
        )

    low_confidence_count = sum(1 for match in matching_output.matches if match.low_confidence)

    print("Preflight checks passed. Full pipeline completed and exports written.")
    return 2 if low_confidence_count > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
