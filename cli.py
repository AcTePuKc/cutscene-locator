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
    CapabilityRequirements,
    ASRExecutionContext,
    dispatch_asr_transcription,
    apply_cross_chunk_continuity,
    get_backend,
    list_backend_status,
    parse_asr_result,
    resolve_device_with_details,
    validate_backend_capabilities,
)
from src.asr.device import select_cuda_probe
from src.asr.model_resolution import ModelResolutionError, resolve_model_path
from src.export import (
    write_matches_csv,
    write_scenes_json,
    write_subs_qa_srt,
    write_subs_target_srt,
)
from src.ingest import load_script_table, preprocess_media
from src.match.engine import MatchingConfig, match_segments_to_script
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
    parser.add_argument("--match-quick-threshold", type=float, default=0.25)
    parser.add_argument("--match-length-bucket-size", type=int, default=4)
    parser.add_argument("--match-max-length-bucket-delta", type=int, default=3)
    parser.add_argument("--match-monotonic-window", type=int, default=0)
    parser.add_argument("--match-progress-every", type=int, default=50)
    parser.add_argument("--asr-vad-filter", choices=("on", "off"), default="off")
    parser.add_argument("--asr-merge-short-segments", type=float, default=0.0)
    parser.add_argument("--asr-language")
    parser.add_argument("--asr-beam-size", type=int, default=1)
    parser.add_argument("--asr-temperature", type=float, default=0.0)
    parser.add_argument("--asr-best-of", type=int, default=1)
    parser.add_argument("--asr-no-speech-threshold", type=float, default=None)
    parser.add_argument("--asr-logprob-threshold", type=float, default=None)
    parser.add_argument("--progress", choices=("on", "off"), default=None)
    parser.add_argument("--asr-preflight-only", action="store_true")
    return parser


def _validate_required_args(args: argparse.Namespace) -> None:
    if args.asr_preflight_only:
        return

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
    backend_status_by_name = {status.name: status for status in list_backend_status()}
    status = backend_status_by_name.get(args.asr_backend)
    if status is not None and not status.enabled:
        missing_deps = tuple(status.missing_dependencies)
        message = f"ASR backend '{status.name}' is declared but currently disabled."
        if missing_deps:
            missing = ", ".join(missing_deps)
            message = f"{message} Missing optional dependencies: {missing}."
        else:
            message = f"{message} Reason: {status.reason}."
        if missing_deps and status.install_extra is not None:
            message = (
                f"{message} Install with: `pip install 'cutscene-locator[{status.install_extra}]'`."
            )
        raise CliError(message)

    try:
        registration = get_backend(args.asr_backend)
    except ValueError as exc:
        raise CliError(str(exc)) from exc

    if registration.name == "mock" and not args.mock_asr_path:
        raise CliError("--mock-asr is required when --asr-backend mock is used.")

    requirements = CapabilityRequirements(
        requires_segment_timestamps=True,
        allows_alignment_backends=False,
    )
    try:
        validate_backend_capabilities(
            registration,
            requires_segment_timestamps=requirements.requires_segment_timestamps,
            allows_alignment_backends=requirements.allows_alignment_backends,
        )
    except ValueError as exc:
        raise CliError(str(exc)) from exc




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
    if args.match_progress_every <= 0:
        raise CliError("Invalid --match-progress-every value. Expected an integer greater than 0.")
    if args.match_length_bucket_size <= 0:
        raise CliError("Invalid --match-length-bucket-size value. Expected an integer greater than 0.")
    if args.match_max_length_bucket_delta < 0:
        raise CliError("Invalid --match-max-length-bucket-delta value. Expected an integer greater than or equal to 0.")
    if args.match_monotonic_window < 0:
        raise CliError("Invalid --match-monotonic-window value. Expected an integer greater than or equal to 0.")
    if args.asr_merge_short_segments < 0:
        raise CliError("Invalid --asr-merge-short-segments value. Expected a float greater than or equal to 0.")

    if args.asr_beam_size < 1:
        raise CliError("Invalid --asr-beam-size value. Expected an integer greater than or equal to 1.")
    if args.asr_temperature < 0.0:
        raise CliError("Invalid --asr-temperature value. Expected a float greater than or equal to 0.0.")
    if args.asr_best_of < 1:
        raise CliError("Invalid --asr-best-of value. Expected an integer greater than or equal to 1.")
    if args.asr_temperature == 0.0 and args.asr_best_of > 1:
        raise CliError("Invalid ASR decode options: --asr-best-of must be 1 when --asr-temperature is 0.0.")

    if args.asr_no_speech_threshold is not None and not (0.0 <= args.asr_no_speech_threshold <= 1.0):
        raise CliError("Invalid --asr-no-speech-threshold value. Expected a float in [0.0, 1.0].")
    if args.asr_logprob_threshold is not None and not (0.0 <= args.asr_logprob_threshold <= 1.0):
        raise CliError("Invalid --asr-logprob-threshold value. Expected a float in [0.0, 1.0].")

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
            "-u",
            "-X",
            "faulthandler",
            "-m",
            "src.asr.asr_worker",
            "--asr-backend",
            asr_config.backend_name,
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
            "--asr-beam-size",
            str(asr_config.beam_size),
            "--asr-temperature",
            str(asr_config.temperature),
            "--asr-best-of",
            str(asr_config.best_of),
        ]
        if asr_config.language is not None:
            cmd.extend(["--asr-language", asr_config.language])
        if asr_config.no_speech_threshold is not None:
            cmd.extend(["--asr-no-speech-threshold", str(asr_config.no_speech_threshold)])
        if asr_config.log_prob_threshold is not None:
            cmd.extend(["--asr-logprob-threshold", str(asr_config.log_prob_threshold)])
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


def _build_preflight_output(
    *,
    asr_config: ASRConfig,
    backend_name: str,
    resolved_model_path: Path | None,
    device_resolution_reason: str | None,
    cuda_probe_label: str | None,
) -> dict[str, object]:
    output: dict[str, object] = {
        "mode": "asr_preflight_only",
        "backend": backend_name,
        "model_resolution": {
            "requested": {
                "model_path": str(asr_config.model_path) if asr_config.model_path is not None else None,
                "model_id": asr_config.model_id,
                "revision": asr_config.revision,
                "auto_download": asr_config.auto_download,
            },
            "resolved_model_path": str(resolved_model_path) if resolved_model_path is not None else None,
        },
        "device": {
            "requested": asr_config.device,
            "compute_type": asr_config.compute_type,
            "cuda_probe_label": cuda_probe_label,
            "resolution_reason": device_resolution_reason,
        },
    }
    return output


def _run_asr_preflight_only(
    *,
    args: argparse.Namespace,
    asr_config: ASRConfig,
) -> int:
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
    requirements = CapabilityRequirements(
        requires_segment_timestamps=True,
        allows_alignment_backends=False,
    )
    validate_backend_capabilities(
        backend_registration,
        requires_segment_timestamps=requirements.requires_segment_timestamps,
        allows_alignment_backends=requirements.allows_alignment_backends,
    )

    resolution_reason: str | None = None
    cuda_probe_label: str | None = None
    if backend_registration.name != "mock":
        cuda_checker, cuda_probe_label = select_cuda_probe(backend_registration.name)
        resolution = resolve_device_with_details(
            asr_config.device,
            cuda_available_checker=cuda_checker,
            cuda_probe_reason_label=cuda_probe_label,
        )
        resolution_reason = resolution.reason
    else:
        resolution_reason = "mock backend does not require device probing"

    payload = _build_preflight_output(
        asr_config=asr_config,
        backend_name=backend_registration.name,
        resolved_model_path=resolved_model_path,
        device_resolution_reason=resolution_reason,
        cuda_probe_label=cuda_probe_label,
    )
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0


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
        asr_config = ASRConfig(
            backend_name=args.asr_backend,
            model_path=Path(args.model_path) if args.model_path else None,
            model_id=args.model_id,
            revision=args.revision,
            auto_download=args.auto_download,
            device=args.device,
            compute_type=args.compute_type,
            language=args.asr_language,
            beam_size=args.asr_beam_size,
            temperature=args.asr_temperature,
            best_of=1 if args.asr_temperature == 0.0 else args.asr_best_of,
            no_speech_threshold=args.asr_no_speech_threshold,
            log_prob_threshold=args.asr_logprob_threshold,
            vad_filter=args.asr_vad_filter == "on",
            merge_short_segments_seconds=args.asr_merge_short_segments,
            ffmpeg_path=None,
            download_progress=(args.progress == "on"),
            log_callback=model_resolution_logs.append,
        )
        if args.asr_preflight_only:
            return _run_asr_preflight_only(args=args, asr_config=asr_config)

        ffmpeg_binary = resolve_ffmpeg_binary(args.ffmpeg_path, which=which)
        asr_config = ASRConfig(
            backend_name=asr_config.backend_name,
            model_path=asr_config.model_path,
            model_id=asr_config.model_id,
            revision=asr_config.revision,
            auto_download=asr_config.auto_download,
            device=asr_config.device,
            compute_type=asr_config.compute_type,
            language=asr_config.language,
            beam_size=asr_config.beam_size,
            temperature=asr_config.temperature,
            best_of=asr_config.best_of,
            no_speech_threshold=asr_config.no_speech_threshold,
            log_prob_threshold=asr_config.log_prob_threshold,
            vad_filter=asr_config.vad_filter,
            merge_short_segments_seconds=asr_config.merge_short_segments_seconds,
            ffmpeg_path=ffmpeg_binary,
            download_progress=asr_config.download_progress,
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
        requirements = CapabilityRequirements(
            requires_segment_timestamps=True,
            allows_alignment_backends=False,
        )
        validate_backend_capabilities(
            backend_registration,
            requires_segment_timestamps=requirements.requires_segment_timestamps,
            allows_alignment_backends=requirements.allows_alignment_backends,
        )

        if backend_registration.name != "mock":
            resolution = resolve_device_with_details(asr_config.device)
            device_resolution_reason = resolution.reason

        asr_started = time.perf_counter()
        if args.verbose:
            print("stage: asr start")
        asr_result = dispatch_asr_transcription(
            audio_path=str(preprocessing_output.canonical_wav_path),
            config=asr_config,
            context=ASRExecutionContext(
                resolved_model_path=resolved_model_path,
                verbose=args.verbose,
                mock_asr_path=args.mock_asr_path,
                run_faster_whisper_subprocess=_run_faster_whisper_subprocess,
                faster_whisper_preflight=_print_faster_whisper_cuda_preflight,
            ),
            requirements=requirements,
        )
        asr_result = apply_cross_chunk_continuity(
            asr_result=asr_result,
            chunk_offsets_by_index={
                chunk.chunk_index: chunk.absolute_offset_seconds
                for chunk in preprocessing_output.chunk_metadata
            },
        )
        timings["asr"] = time.perf_counter() - asr_started
        if args.verbose:
            print("stage: asr end")

        matching_started = time.perf_counter()
        if args.verbose:
            print("stage: matching start")
        matching_output = match_segments_to_script(
            asr_result=asr_result,
            script_table=script_table,
            config=MatchingConfig(
                low_confidence_threshold=args.match_threshold,
                quick_filter_threshold=args.match_quick_threshold,
                length_bucket_size=args.match_length_bucket_size,
                max_length_bucket_delta=args.match_max_length_bucket_delta,
                monotonic_window=args.match_monotonic_window,
                progress_every=args.match_progress_every,
            ),
            progress_logger=print if args.verbose else None,
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
            f"auto_download={asr_config.auto_download} download_progress={args.progress} "
            f"language={asr_config.language} beam_size={asr_config.beam_size} temperature={asr_config.temperature} "
            f"best_of={asr_config.best_of} no_speech_threshold={asr_config.no_speech_threshold} "
            f"log_prob_threshold={asr_config.log_prob_threshold} "
            f"vad_filter={asr_config.vad_filter} merge_short_segments={asr_config.merge_short_segments_seconds}"
        )
        print(
            "Verbose: matching config="
            f"quick_threshold={args.match_quick_threshold} length_bucket_size={args.match_length_bucket_size} "
            f"max_length_bucket_delta={args.match_max_length_bucket_delta} "
            f"monotonic_window={args.match_monotonic_window} progress_every={args.match_progress_every}"
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
