import io
import json
import sys
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import cli


class CliPhaseOneTests(unittest.TestCase):
    def test_version_exits_success(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = cli.main(["--version"])

        self.assertEqual(code, 0)
        self.assertIn("cutscene-locator", stdout.getvalue())

    def test_asr_preflight_only_success_outputs_deterministic_json(self) -> None:
        stdout = io.StringIO()
        with patch("cli.resolve_model_path", return_value=Path("models/faster-whisper/tiny")):
            with patch(
                "cli.resolve_device_with_details",
                return_value=SimpleNamespace(
                    requested="auto",
                    resolved="cuda",
                    reason="--device auto selected cuda because ctranslate2 CUDA probe reported available",
                ),
            ):
                with redirect_stdout(stdout):
                    code = cli.main(
                        [
                            "--asr-preflight-only",
                            "--asr-backend",
                            "faster-whisper",
                            "--model-path",
                            "models/faster-whisper/tiny",
                            "--device",
                            "auto",
                        ]
                    )

        self.assertEqual(code, 0)
        stdout_lines = stdout.getvalue().splitlines()
        self.assertEqual(len(stdout_lines), 1)
        payload = json.loads(stdout_lines[0])
        self.assertEqual(payload["mode"], "asr_preflight_only")
        self.assertEqual(payload["backend"], "faster-whisper")
        self.assertEqual(payload["model_resolution"]["resolved_model_path"], "models/faster-whisper/tiny")
        self.assertEqual(payload["device"]["requested"], "auto")
        self.assertEqual(payload["device"]["compute_type"], "auto")
        self.assertEqual(payload["device"]["cuda_probe_label"], "ctranslate2")
        self.assertIn("selected cuda", payload["device"]["resolution_reason"])

    def test_asr_preflight_only_verbose_outputs_single_json_line_without_stage_logs(self) -> None:
        stdout = io.StringIO()
        with patch("cli.resolve_model_path", return_value=Path("models/faster-whisper/tiny")):
            with patch(
                "cli.resolve_device_with_details",
                return_value=SimpleNamespace(
                    requested="auto",
                    resolved="cpu",
                    reason="--device auto selected cpu because ctranslate2 CUDA probe reported unavailable",
                ),
            ):
                with redirect_stdout(stdout):
                    code = cli.main(
                        [
                            "--asr-preflight-only",
                            "--verbose",
                            "--asr-backend",
                            "faster-whisper",
                            "--model-path",
                            "models/faster-whisper/tiny",
                        ]
                    )

        self.assertEqual(code, 0)
        stdout_lines = stdout.getvalue().splitlines()
        self.assertEqual(len(stdout_lines), 1)
        payload = json.loads(stdout_lines[0])
        self.assertEqual(payload["mode"], "asr_preflight_only")
        for forbidden in (
            "stage:",
            "Verbose:",
            "Preflight checks passed.",
        ):
            self.assertNotIn(forbidden, stdout.getvalue())

    def test_asr_preflight_only_includes_backend_probe_label(self) -> None:
        scenarios = (("faster-whisper", "ctranslate2"), ("qwen3-asr", "torch"))
        for backend_name, expected_probe_label in scenarios:
            with self.subTest(backend=backend_name):
                stdout = io.StringIO()
                enabled_backend = SimpleNamespace(
                    name=backend_name,
                    enabled=True,
                    missing_dependencies=(),
                    reason="enabled",
                    install_extra="asr_qwen3",
                )
                registration = SimpleNamespace(
                    name=backend_name,
                    capabilities=SimpleNamespace(supports_segment_timestamps=True, supports_alignment=False),
                )
                with patch("cli.list_backend_status", return_value=[enabled_backend]):
                    with patch("cli.get_backend", return_value=registration):
                        with patch("cli.resolve_model_path", return_value=Path(f"models/{backend_name}")):
                            with patch(
                                "cli.resolve_device_with_details",
                                return_value=SimpleNamespace(
                                    requested="auto",
                                    resolved="cpu",
                                    reason=f"--device auto selected cpu because {expected_probe_label} CUDA probe reported unavailable",
                                ),
                            ) as resolve_patch:
                                with redirect_stdout(stdout):
                                    code = cli.main(
                                        [
                                            "--asr-preflight-only",
                                            "--asr-backend",
                                            backend_name,
                                            "--model-path",
                                            f"models/{backend_name}",
                                            "--device",
                                            "auto",
                                        ]
                                    )

                self.assertEqual(code, 0)
                payload = json.loads(stdout.getvalue().strip().splitlines()[0])
                self.assertEqual(payload["device"]["cuda_probe_label"], expected_probe_label)
                self.assertIn(expected_probe_label, payload["device"]["resolution_reason"])
                self.assertEqual(
                    resolve_patch.call_args.kwargs["cuda_probe_reason_label"],
                    expected_probe_label,
                )

    def test_asr_preflight_only_model_resolution_failure_exits_one(self) -> None:
        stderr = io.StringIO()
        with patch(
            "cli.resolve_model_path",
            side_effect=cli.ModelResolutionError("Model could not be resolved for requested backend."),
        ):
            with redirect_stderr(stderr):
                code = cli.main(
                    [
                        "--asr-preflight-only",
                        "--asr-backend",
                        "faster-whisper",
                    ]
                )

        self.assertEqual(code, 1)
        self.assertIn("Model could not be resolved for requested backend.", stderr.getvalue())

    def test_asr_preflight_only_device_probe_failure_exits_one(self) -> None:
        stderr = io.StringIO()
        enabled_backend = SimpleNamespace(
            name="qwen3-asr",
            enabled=True,
            missing_dependencies=(),
            reason="enabled",
            install_extra="asr_qwen3",
        )
        registration = SimpleNamespace(
            name="qwen3-asr",
            capabilities=SimpleNamespace(supports_segment_timestamps=True, supports_alignment=False),
        )
        with patch("cli.list_backend_status", return_value=[enabled_backend]):
            with patch("cli.get_backend", return_value=registration):
                with patch("cli.resolve_model_path", return_value=Path("models/qwen3-asr")):
                    with patch(
                        "cli.resolve_device_with_details",
                        side_effect=ValueError(
                            "Requested --device cuda, but CUDA is unavailable. Reason: torch CUDA probe reported unavailable."
                        ),
                    ):
                        with redirect_stderr(stderr):
                            code = cli.main(
                                [
                                    "--asr-preflight-only",
                                    "--asr-backend",
                                    "qwen3-asr",
                                    "--model-path",
                                    "models/qwen3-asr",
                                    "--device",
                                    "cuda",
                                ]
                            )

        self.assertEqual(code, 1)
        self.assertIn("Requested --device cuda, but CUDA is unavailable", stderr.getvalue())



    def test_missing_required_args_exits_one(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = cli.main([])

        self.assertEqual(code, 1)
        self.assertIn("Missing required arguments", stderr.getvalue())

    def test_invalid_backend_exits_one(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = cli.main(
                [
                    "--input",
                    "in.wav",
                    "--script",
                    "script.tsv",
                    "--out",
                    "out",
                    "--asr-backend",
                    "bad",
                ],
                which=lambda _: "/usr/bin/ffmpeg",
                runner=lambda *args, **kwargs: subprocess.CompletedProcess(args, 0),
            )

        self.assertEqual(code, 1)
        self.assertIn("Unknown ASR backend", stderr.getvalue())
        self.assertNotIn("declared but currently disabled", stderr.getvalue())

    def test_declared_but_disabled_backend_exits_with_actionable_error(self) -> None:
        stderr = io.StringIO()
        disabled_backend = SimpleNamespace(
            name="qwen3-asr",
            enabled=False,
            missing_dependencies=("torch", "qwen_asr"),
            install_extra="asr_qwen3",
        )
        with patch("cli.list_backend_status", return_value=[disabled_backend]):
            with redirect_stderr(stderr):
                code = cli.main(
                    [
                        "--input",
                        "in.wav",
                        "--script",
                        "script.tsv",
                        "--out",
                        "out",
                        "--asr-backend",
                        "qwen3-asr",
                    ],
                    which=lambda _: "/usr/bin/ffmpeg",
                    runner=lambda *args, **kwargs: subprocess.CompletedProcess(args, 0),
                )

        self.assertEqual(code, 1)
        self.assertIn("declared but currently disabled", stderr.getvalue())
        self.assertIn("Missing optional dependencies: torch, qwen_asr", stderr.getvalue())
        self.assertIn("pip install 'cutscene-locator[asr_qwen3]'", stderr.getvalue())




    def test_declared_but_disabled_vibevoice_backend_exits_with_actionable_error(self) -> None:
        stderr = io.StringIO()
        disabled_backend = SimpleNamespace(
            name="vibevoice",
            enabled=False,
            missing_dependencies=("vibevoice", "torch"),
            install_extra="asr_vibevoice",
        )
        with patch("cli.list_backend_status", return_value=[disabled_backend]):
            with redirect_stderr(stderr):
                code = cli.main(
                    [
                        "--input",
                        "in.wav",
                        "--script",
                        "script.tsv",
                        "--out",
                        "out",
                        "--asr-backend",
                        "vibevoice",
                    ],
                    which=lambda _: "/usr/bin/ffmpeg",
                    runner=lambda *args, **kwargs: subprocess.CompletedProcess(args, 0),
                )

        self.assertEqual(code, 1)
        self.assertIn("declared but currently disabled", stderr.getvalue())
        self.assertIn("Missing optional dependencies: vibevoice, torch", stderr.getvalue())
        self.assertIn("pip install 'cutscene-locator[asr_vibevoice]'", stderr.getvalue())

    def test_enabled_backend_path_unchanged(self) -> None:
        stderr = io.StringIO()
        enabled_backend = SimpleNamespace(
            name="mock",
            enabled=True,
            missing_dependencies=(),
            reason="enabled",
            install_extra=None,
        )
        with patch("cli.list_backend_status", return_value=[enabled_backend]):
            with redirect_stderr(stderr):
                code = cli.main(
                    [
                        "--input",
                        "in.wav",
                        "--script",
                        "script.tsv",
                        "--out",
                        "out",
                        "--asr-backend",
                        "mock",
                    ],
                    which=lambda _: "/usr/bin/ffmpeg",
                    runner=lambda *args, **kwargs: subprocess.CompletedProcess(args, 0),
                )

        self.assertEqual(code, 1)
        self.assertIn("--mock-asr is required", stderr.getvalue())
        self.assertNotIn("declared but currently disabled", stderr.getvalue())

    def test_alignment_backend_rejected_in_asr_mode(self) -> None:
        stderr = io.StringIO()
        alignment_registration = SimpleNamespace(
            name="qwen3-asr",
            capabilities=SimpleNamespace(supports_alignment=True),
        )
        with patch("cli.get_backend", return_value=alignment_registration):
            with patch("cli.list_backend_status", return_value=[]):
                with redirect_stderr(stderr):
                    code = cli.main(
                        [
                            "--input",
                            "in.wav",
                            "--script",
                            "script.tsv",
                            "--out",
                            "out",
                            "--asr-backend",
                            "qwen3-asr",
                        ],
                        which=lambda _: "/usr/bin/ffmpeg",
                        runner=lambda *args, **kwargs: subprocess.CompletedProcess(args, 0),
                    )

        self.assertEqual(code, 1)
        self.assertIn("alignment backend", stderr.getvalue())
        self.assertIn("alignment pipeline path", stderr.getvalue())

    def test_mock_backend_requires_mock_asr(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = cli.main(
                [
                    "--input",
                    "in.wav",
                    "--script",
                    "script.tsv",
                    "--out",
                    "out",
                ],
                which=lambda _: "/usr/bin/ffmpeg",
                runner=lambda *args, **kwargs: subprocess.CompletedProcess(args, 0),
            )

        self.assertEqual(code, 1)
        self.assertIn("--mock-asr is required", stderr.getvalue())


    def test_faster_whisper_missing_dependency_exits_one(self) -> None:
        stderr = io.StringIO()
        with patch("cli.resolve_model_path", return_value=Path("models/faster-whisper")):
            with patch("src.asr.faster_whisper_backend.import_module", side_effect=ModuleNotFoundError()):
                with redirect_stderr(stderr):
                    code = cli.main(
                        [
                            "--input",
                            "in.wav",
                            "--script",
                            "tests/fixtures/script_sample.tsv",
                            "--out",
                            "out",
                            "--asr-backend",
                            "faster-whisper",
                            "--model-path",
                            "models/faster-whisper",
                            "--chunk",
                            "0",
                        ],
                        which=lambda _: "/usr/bin/ffmpeg",
                        runner=lambda *args, **kwargs: subprocess.CompletedProcess(args, 0),
                    )

        self.assertEqual(code, 1)
        self.assertIn("Install it with: pip install 'cutscene-locator[faster-whisper]'", stderr.getvalue())


    def test_resolve_progress_mode_defaults_off_on_windows(self) -> None:
        with patch("cli.os.name", "nt"):
            self.assertEqual(cli._resolve_progress_mode(None), "off")

    def test_resolve_progress_mode_defaults_on_non_windows(self) -> None:
        with patch("cli.os.name", "posix"):
            self.assertEqual(cli._resolve_progress_mode(None), "on")

    def test_resolve_progress_mode_honors_explicit_value(self) -> None:
        with patch("cli.os.name", "nt"):
            self.assertEqual(cli._resolve_progress_mode("on"), "on")


    def test_apply_windows_progress_guard_sets_env_vars_on_windows(self) -> None:
        with patch("cli.os.name", "nt"):
            with patch.dict("cli.os.environ", {}, clear=True):
                cli._apply_windows_progress_guard()
                self.assertEqual(cli.os.environ.get("TQDM_DISABLE"), "1")
                self.assertEqual(cli.os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS"), "1")

    def test_apply_windows_progress_guard_noop_on_non_windows(self) -> None:
        with patch("cli.os.name", "posix"):
            with patch.dict("cli.os.environ", {}, clear=True):
                cli._apply_windows_progress_guard()
                self.assertNotIn("TQDM_DISABLE", cli.os.environ)
                self.assertNotIn("HF_HUB_DISABLE_PROGRESS_BARS", cli.os.environ)



    def test_asr_knob_defaults_parse(self) -> None:
        args = cli.build_parser().parse_args([])
        self.assertIsNone(args.asr_language)
        self.assertEqual(args.asr_beam_size, 1)
        self.assertEqual(args.asr_temperature, 0.0)
        self.assertEqual(args.asr_best_of, 1)
        self.assertIsNone(args.asr_no_speech_threshold)
        self.assertIsNone(args.asr_logprob_threshold)

    def test_invalid_asr_beam_size_exits_one(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = cli.main(
                [
                    "--input",
                    "in.wav",
                    "--script",
                    "script.tsv",
                    "--out",
                    "out",
                    "--mock-asr",
                    "tests/fixtures/mock_asr_valid.json",
                    "--asr-beam-size",
                    "0",
                ],
                which=lambda _: "/usr/bin/ffmpeg",
                runner=lambda *args, **kwargs: subprocess.CompletedProcess(args, 0),
            )

        self.assertEqual(code, 1)
        self.assertIn("Invalid --asr-beam-size value", stderr.getvalue())

    def test_invalid_asr_temperature_exits_one(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = cli.main(
                [
                    "--input",
                    "in.wav",
                    "--script",
                    "script.tsv",
                    "--out",
                    "out",
                    "--mock-asr",
                    "tests/fixtures/mock_asr_valid.json",
                    "--asr-temperature",
                    "-0.1",
                ],
                which=lambda _: "/usr/bin/ffmpeg",
                runner=lambda *args, **kwargs: subprocess.CompletedProcess(args, 0),
            )

        self.assertEqual(code, 1)
        self.assertIn("Invalid --asr-temperature value", stderr.getvalue())

    def test_asr_best_of_rejected_when_temperature_zero(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = cli.main(
                [
                    "--input",
                    "in.wav",
                    "--script",
                    "script.tsv",
                    "--out",
                    "out",
                    "--mock-asr",
                    "tests/fixtures/mock_asr_valid.json",
                    "--asr-temperature",
                    "0.0",
                    "--asr-best-of",
                    "2",
                ],
                which=lambda _: "/usr/bin/ffmpeg",
                runner=lambda *args, **kwargs: subprocess.CompletedProcess(args, 0),
            )

        self.assertEqual(code, 1)
        self.assertIn("--asr-best-of must be 1", stderr.getvalue())

    def test_invalid_asr_thresholds_exits_one(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = cli.main(
                [
                    "--input",
                    "in.wav",
                    "--script",
                    "script.tsv",
                    "--out",
                    "out",
                    "--mock-asr",
                    "tests/fixtures/mock_asr_valid.json",
                    "--asr-no-speech-threshold",
                    "1.2",
                ],
                which=lambda _: "/usr/bin/ffmpeg",
                runner=lambda *args, **kwargs: subprocess.CompletedProcess(args, 0),
            )

        self.assertEqual(code, 1)
        self.assertIn("Invalid --asr-no-speech-threshold value", stderr.getvalue())

    def test_invalid_asr_logprob_threshold_exits_one(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = cli.main(
                [
                    "--input",
                    "in.wav",
                    "--script",
                    "script.tsv",
                    "--out",
                    "out",
                    "--mock-asr",
                    "tests/fixtures/mock_asr_valid.json",
                    "--asr-logprob-threshold",
                    "-0.2",
                ],
                which=lambda _: "/usr/bin/ffmpeg",
                runner=lambda *args, **kwargs: subprocess.CompletedProcess(args, 0),
            )

        self.assertEqual(code, 1)
        self.assertIn("Invalid --asr-logprob-threshold value", stderr.getvalue())

    def test_invalid_match_progress_every_exits_one(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = cli.main(
                [
                    "--input",
                    "in.wav",
                    "--script",
                    "script.tsv",
                    "--out",
                    "out",
                    "--mock-asr",
                    "tests/fixtures/mock_asr_valid.json",
                    "--match-progress-every",
                    "0",
                ],
                which=lambda _: "/usr/bin/ffmpeg",
                runner=lambda *args, **kwargs: subprocess.CompletedProcess(args, 0),
            )

        self.assertEqual(code, 1)
        self.assertIn("Invalid --match-progress-every value", stderr.getvalue())

    def test_invalid_device_value_exits_one(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = cli.main(
                [
                    "--input",
                    "in.wav",
                    "--script",
                    "script.tsv",
                    "--out",
                    "out",
                    "--mock-asr",
                    "tests/fixtures/mock_asr_valid.json",
                    "--device",
                    "tpu",
                ],
                which=lambda _: "/usr/bin/ffmpeg",
                runner=lambda *args, **kwargs: subprocess.CompletedProcess(args, 0),
            )

        self.assertEqual(code, 1)
        self.assertIn("Invalid --device value", stderr.getvalue())


    def test_device_cuda_unavailable_exits_one_with_actionable_error(self) -> None:
        stderr = io.StringIO()
        with patch("src.asr.backends.resolve_device", side_effect=ValueError("Requested --device cuda, but CUDA is unavailable. Install a CUDA-enabled runtime or rerun with --device cpu.")):
            with redirect_stderr(stderr):
                code = cli.main(
                    [
                        "--input",
                        "in.wav",
                        "--script",
                        "tests/fixtures/script_sample.tsv",
                        "--out",
                        "out",
                        "--mock-asr",
                        "tests/fixtures/mock_asr_valid.json",
                        "--chunk",
                        "0",
                        "--device",
                        "cuda",
                    ],
                    which=lambda _: "/usr/bin/ffmpeg",
                    runner=lambda *args, **kwargs: subprocess.CompletedProcess(args, 0),
                )

        self.assertEqual(code, 1)
        self.assertIn("Requested --device cuda, but CUDA is unavailable", stderr.getvalue())

    def test_invalid_auto_download_value_exits_one(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = cli.main(
                [
                    "--input",
                    "in.wav",
                    "--script",
                    "script.tsv",
                    "--out",
                    "out",
                    "--mock-asr",
                    "tests/fixtures/mock_asr_valid.json",
                    "--auto-download",
                    "medium",
                ],
                which=lambda _: "/usr/bin/ffmpeg",
                runner=lambda *args, **kwargs: subprocess.CompletedProcess(args, 0),
            )

        self.assertEqual(code, 1)
        self.assertIn("Invalid --auto-download value", stderr.getvalue())

    def test_auto_download_base_without_url_config_exits_one(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = cli.main(
                [
                    "--input",
                    "in.wav",
                    "--script",
                    "tests/fixtures/script_sample.tsv",
                    "--out",
                    "out",
                    "--mock-asr",
                    "tests/fixtures/mock_asr_valid.json",
                    "--auto-download",
                    "base",
                    "--chunk",
                    "0",
                ],
                which=lambda _: "/usr/bin/ffmpeg",
                runner=lambda *args, **kwargs: subprocess.CompletedProcess(args, 0),
            )

        self.assertEqual(code, 1)
        self.assertIn("Auto-download URL is not configured", stderr.getvalue())

    def test_models_convention_used_when_model_path_not_set(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir)
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = cli.main(
                    [
                        "--input",
                        "in.wav",
                        "--script",
                        "tests/fixtures/script_sample.tsv",
                        "--out",
                        str(out_dir),
                        "--mock-asr",
                        "tests/fixtures/mock_asr_valid.json",
                        "--chunk",
                        "0",
                        "--match-threshold",
                        "0.0",
                        "--verbose",
                    ],
                    which=lambda _: "/usr/bin/ffmpeg",
                    runner=lambda *args, **kwargs: subprocess.CompletedProcess(args, 0),
                )

            self.assertEqual(code, 0)
            self.assertIn("model_path=None", stdout.getvalue())


    def test_explicit_model_path_missing_exits_one(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = cli.main(
                [
                    "--input",
                    "in.wav",
                    "--script",
                    "tests/fixtures/script_sample.tsv",
                    "--out",
                    "out",
                    "--mock-asr",
                    "tests/fixtures/mock_asr_valid.json",
                    "--chunk",
                    "0",
                    "--model-path",
                    "does/not/exist.bin",
                ],
                which=lambda _: "/usr/bin/ffmpeg",
                runner=lambda *args, **kwargs: subprocess.CompletedProcess(args, 0),
            )

        self.assertEqual(code, 1)
        self.assertIn("Model not found at explicit --model-path", stderr.getvalue())


    def test_revision_requires_model_id(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = cli.main(
                [
                    "--input",
                    "in.wav",
                    "--script",
                    "script.tsv",
                    "--out",
                    "out",
                    "--mock-asr",
                    "tests/fixtures/mock_asr_valid.json",
                    "--revision",
                    "main",
                ],
                which=lambda _: "/usr/bin/ffmpeg",
                runner=lambda *args, **kwargs: subprocess.CompletedProcess(args, 0),
            )

        self.assertEqual(code, 1)
        self.assertIn("--revision requires --model-id", stderr.getvalue())

    def test_model_flags_are_mutually_exclusive(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = cli.main(
                [
                    "--input",
                    "in.wav",
                    "--script",
                    "script.tsv",
                    "--out",
                    "out",
                    "--mock-asr",
                    "tests/fixtures/mock_asr_valid.json",
                    "--model-id",
                    "openai/whisper-tiny",
                    "--auto-download",
                    "tiny",
                ],
                which=lambda _: "/usr/bin/ffmpeg",
                runner=lambda *args, **kwargs: subprocess.CompletedProcess(args, 0),
            )

        self.assertEqual(code, 1)
        self.assertIn("Use only one of", stderr.getvalue())

    def test_ffmpeg_not_found_exits_one(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = cli.main(
                [
                    "--input",
                    "in.wav",
                    "--script",
                    "tests/fixtures/script_sample.tsv",
                    "--out",
                    "out",
                    "--mock-asr",
                    "tests/fixtures/mock_asr_valid.json",
                    "--chunk",
                    "0",
                ],
                which=lambda _: None,
            )

        self.assertEqual(code, 1)
        self.assertIn("ffmpeg not found", stderr.getvalue())

    def test_invalid_script_exits_one(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = cli.main(
                [
                    "--input",
                    "in.wav",
                    "--script",
                    "tests/fixtures/script_missing_required.csv",
                    "--out",
                    "out",
                    "--mock-asr",
                    "tests/fixtures/mock_asr_valid.json",
                    "--chunk",
                    "0",
                ],
                which=lambda _: "/usr/bin/ffmpeg",
                runner=lambda *args, **kwargs: subprocess.CompletedProcess(args, 0),
            )

        self.assertEqual(code, 1)
        self.assertIn("Missing required script columns", stderr.getvalue())

    def test_invalid_asr_segment_exits_one(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = cli.main(
                [
                    "--input",
                    "in.wav",
                    "--script",
                    "tests/fixtures/script_sample.tsv",
                    "--out",
                    "out",
                    "--mock-asr",
                    "tests/fixtures/mock_asr_invalid_start_end.json",
                    "--chunk",
                    "0",
                ],
                which=lambda _: "/usr/bin/ffmpeg",
                runner=lambda *args, **kwargs: subprocess.CompletedProcess(args, 0),
            )

        self.assertEqual(code, 1)
        self.assertIn("start must be less than end", stderr.getvalue())

    def test_success_with_low_confidence_exits_two_and_writes_exports(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir)
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = cli.main(
                    [
                        "--input",
                        "in.wav",
                        "--script",
                        "tests/fixtures/script_sample.tsv",
                        "--out",
                        str(out_dir),
                        "--mock-asr",
                        "tests/fixtures/mock_asr_valid.json",
                        "--chunk",
                        "0",
                        "--verbose",
                    ],
                    which=lambda _: "/usr/bin/ffmpeg",
                    runner=lambda *args, **kwargs: subprocess.CompletedProcess(args, 0),
                )

            self.assertEqual(code, 2)
            self.assertTrue((out_dir / "matches.csv").exists())
            self.assertTrue((out_dir / "scenes.json").exists())
            self.assertTrue((out_dir / "subs_qa.srt").exists())
            self.assertTrue((out_dir / "subs_target.srt").exists())
            output = stdout.getvalue()
            self.assertIn("Full pipeline completed and exports written", output)
            self.assertIn("stage: preprocess start", output)
            self.assertIn("stage: preprocess end", output)
            self.assertIn("stage: asr start", output)
            self.assertIn("stage: asr end", output)
            self.assertIn("stage: matching start", output)
            self.assertIn("stage: matching end", output)
            self.assertIn("stage: exports start", output)
            self.assertIn("stage: exports end", output)
            self.assertIn("Verbose: matching progress 2/2 segments", output)
            self.assertIn("Verbose: script rows loaded=2", output)
            self.assertIn("Verbose: timings seconds=", output)

    def test_success_without_low_confidence_exits_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir)
            code = cli.main(
                [
                    "--input",
                    "in.wav",
                    "--script",
                    "tests/fixtures/script_sample.tsv",
                    "--out",
                    str(out_dir),
                    "--mock-asr",
                    "tests/fixtures/mock_asr_valid.json",
                    "--chunk",
                    "0",
                    "--match-threshold",
                    "0.0",
                ],
                which=lambda _: "/usr/bin/ffmpeg",
                runner=lambda *args, **kwargs: subprocess.CompletedProcess(args, 0),
            )

            self.assertEqual(code, 0)


    def test_invalid_compute_type_value_exits_one(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = cli.main(
                [
                    "--input",
                    "in.wav",
                    "--script",
                    "script.tsv",
                    "--out",
                    "out",
                    "--mock-asr",
                    "tests/fixtures/mock_asr_valid.json",
                    "--compute-type",
                    "bf16",
                ],
                which=lambda _: "/usr/bin/ffmpeg",
                runner=lambda *args, **kwargs: subprocess.CompletedProcess(args, 0),
            )

        self.assertEqual(code, 1)
        self.assertIn("Invalid --compute-type value", stderr.getvalue())

    def test_windows_cuda_abort_in_worker_has_actionable_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result_path = Path(temp_dir) / "audio.wav"
            result_path.write_text("fake", encoding="utf-8")
            config = cli.ASRConfig(
                backend_name="faster-whisper",
                model_path=Path("models/faster-whisper"),
                device="cuda",
                compute_type="float16",
            )
            with patch("cli.subprocess.run", return_value=subprocess.CompletedProcess([], 3221226505)):
                with self.assertRaisesRegex(cli.CliError, "compute-type float32") as ctx:
                    cli._run_faster_whisper_subprocess(
                        audio_path=result_path,
                        resolved_model_path=Path("models/faster-whisper"),
                        asr_config=config,
                        verbose=False,
                    )

        self.assertIn("backend=faster-whisper", str(ctx.exception))
        self.assertIn("device=cuda", str(ctx.exception))



    def test_faster_whisper_worker_subprocess_uses_unbuffered_and_faulthandler(self) -> None:
        fake_payload = {
            "segments": [{"segment_id": "seg_0001", "start": 0.0, "end": 1.0, "text": "ok"}],
            "meta": {
                "backend": "faster-whisper",
                "model": "tiny",
                "version": "1.2.1",
                "device": "cuda",
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "audio.wav"
            audio_path.write_text("fake", encoding="utf-8")
            config = cli.ASRConfig(
                backend_name="faster-whisper",
                model_path=Path("models/faster-whisper"),
                device="cuda",
                compute_type="float16",
            )
            captured_cmd: list[str] = []

            def _fake_run(cmd, check, capture_output, text):
                del check, capture_output, text
                captured_cmd[:] = cmd
                result_path = Path(cmd[cmd.index("--result-path") + 1])
                result_path.parent.mkdir(parents=True, exist_ok=True)
                result_path.write_text(json.dumps(fake_payload), encoding="utf-8")
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

            with patch("cli.subprocess.run", side_effect=_fake_run):
                result = cli._run_faster_whisper_subprocess(
                    audio_path=audio_path,
                    resolved_model_path=Path("models/faster-whisper"),
                    asr_config=config,
                    verbose=False,
                )

        self.assertEqual(captured_cmd[0], sys.executable)
        self.assertEqual(captured_cmd[1:5], ["-u", "-X", "faulthandler", "-m"])
        self.assertEqual(captured_cmd[5], "src.asr.asr_worker")
        self.assertEqual(captured_cmd[captured_cmd.index("--asr-backend") + 1], "faster-whisper")
        self.assertEqual(result["segments"][0]["segment_id"], "seg_0001")

    def test_read_ctranslate2_version_not_installed(self) -> None:
        with patch("cli.importlib.import_module", side_effect=ModuleNotFoundError()):
            self.assertEqual(cli._read_ctranslate2_version(), "not-installed")

    def test_print_faster_whisper_cuda_preflight_includes_expected_fields(self) -> None:
        stdout = io.StringIO()
        with patch("cli._read_ctranslate2_version", return_value="4.5.0"):
            with patch("cli._read_ctranslate2_cuda_device_count", return_value=2):
                with redirect_stdout(stdout):
                    cli._print_faster_whisper_cuda_preflight(device="cuda", compute_type="float16")

        output = stdout.getvalue()
        self.assertIn("ctranslate2=4.5.0", output)
        self.assertIn("cuda_device_count=2", output)
        self.assertIn("device=cuda", output)
        self.assertIn("compute_type=float16", output)
        self.assertIn("--compute-type float32", output)


if __name__ == "__main__":
    unittest.main()


class CliAdapterDispatchTests(unittest.TestCase):
    def test_cli_uses_adapter_registry_for_mock_backend_dispatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "out"
            with patch(
                "cli.dispatch_asr_transcription",
                return_value={
                    "segments": [
                        {"segment_id": "seg_0001", "start": 0.0, "end": 1.0, "text": "hello"}
                    ],
                    "meta": {
                        "backend": "mock",
                        "model": "fixture",
                        "version": "1",
                        "device": "cpu",
                    },
                },
            ) as dispatch_call:
                code = cli.main(
                    [
                        "--input",
                        "in.wav",
                        "--script",
                        "tests/fixtures/script_sample.tsv",
                        "--out",
                        str(out_dir),
                        "--asr-backend",
                        "mock",
                        "--mock-asr",
                        "tests/fixtures/mock_asr_valid.json",
                        "--chunk",
                        "0",
                    ],
                    which=lambda _: "/usr/bin/ffmpeg",
                    runner=lambda *args, **kwargs: subprocess.CompletedProcess(args, 0),
                )

        self.assertEqual(code, 0)
        self.assertEqual(dispatch_call.call_count, 1)
        kwargs = dispatch_call.call_args.kwargs
        self.assertTrue(str(kwargs["audio_path"]).endswith("canonical.wav"))
        self.assertEqual(kwargs["config"].backend_name, "mock")



    def test_cli_injects_keyword_compatible_faster_whisper_callbacks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "out"

            def fake_dispatch(*, audio_path: str, config: object, context: object, requirements: object) -> dict[str, object]:
                del audio_path, config, requirements
                context.faster_whisper_preflight(device="cuda", compute_type="float16")
                context.run_faster_whisper_subprocess(
                    audio_path=Path("audio.wav"),
                    resolved_model_path=Path("model"),
                    asr_config=cli.ASRConfig(backend_name="faster-whisper"),
                    verbose=False,
                )
                return {
                    "segments": [
                        {"segment_id": "seg_0001", "start": 0.0, "end": 1.0, "text": "hello"}
                    ],
                    "meta": {
                        "backend": "faster-whisper",
                        "model": "fixture",
                        "version": "1",
                        "device": "cpu",
                    },
                }

            with patch("cli.dispatch_asr_transcription", side_effect=fake_dispatch):
                with patch("cli._run_faster_whisper_subprocess", return_value={
                    "segments": [
                        {"segment_id": "seg_0001", "start": 0.0, "end": 1.0, "text": "hello"}
                    ],
                    "meta": {"backend": "faster-whisper", "model": "fixture", "version": "1", "device": "cpu"},
                }) as subprocess_call:
                    with patch("cli._print_faster_whisper_cuda_preflight") as preflight_call:
                        code = cli.main(
                            [
                                "--input",
                                "in.wav",
                                "--script",
                                "tests/fixtures/script_sample.tsv",
                                "--out",
                                str(out_dir),
                                "--asr-backend",
                                "mock",
                                "--mock-asr",
                                "tests/fixtures/mock_asr_valid.json",
                                "--chunk",
                                "0",
                            ],
                            which=lambda _: "/usr/bin/ffmpeg",
                            runner=lambda *args, **kwargs: subprocess.CompletedProcess(args, 0),
                        )

        self.assertEqual(code, 0)
        preflight_call.assert_called_once_with(device="cuda", compute_type="float16")
        subprocess_call.assert_called_once()
        kwargs = subprocess_call.call_args.kwargs
        self.assertEqual(set(kwargs), {"audio_path", "resolved_model_path", "asr_config", "verbose"})

    def test_cli_surfaces_adapter_registry_error_without_backend_branching(self) -> None:
        stderr = io.StringIO()
        with patch("cli.dispatch_asr_transcription", side_effect=ValueError("No ASR adapter registered for backend 'mock'.")):
            with redirect_stderr(stderr):
                code = cli.main(
                    [
                        "--input",
                        "in.wav",
                        "--script",
                        "tests/fixtures/script_sample.tsv",
                        "--out",
                        "out",
                        "--asr-backend",
                        "mock",
                        "--mock-asr",
                        "tests/fixtures/mock_asr_valid.json",
                        "--chunk",
                        "0",
                    ],
                    which=lambda _: "/usr/bin/ffmpeg",
                    runner=lambda *args, **kwargs: subprocess.CompletedProcess(args, 0),
                )

        self.assertEqual(code, 1)
        self.assertIn("No ASR adapter registered", stderr.getvalue())
