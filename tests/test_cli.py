"""CLI regression tests.

Consolidation policy for debugging clarity:
- Keep backend loader-specific tests separate.
- Keep device/probe selection tests separate.
- Keep subprocess/verbose logging behavior tests separate.
- Consolidate only pure contract assertions and repeated diagnostics formatting checks.

Reviewer checklist note: "consolidate duplicates, keep unique backend behavior isolated."
"""

import io
import json
import os
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
        self.assertEqual(payload["capabilities"]["timestamp_guarantee"], "segment-level")
        self.assertFalse(payload["capabilities"]["supports_alignment"])
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
        scenarios = (("faster-whisper", "ctranslate2"), ("qwen3-asr", "torch"), ("whisperx", "ctranslate2"))
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

    def test_asr_preflight_only_qwen3_outputs_single_json_line_with_expected_payload(self) -> None:
        stdout = io.StringIO()
        enabled_backend = SimpleNamespace(
            name="qwen3-asr",
            enabled=True,
            missing_dependencies=(),
            reason="enabled",
            install_extra="asr_qwen3",
        )
        registration = SimpleNamespace(
            name="qwen3-asr",
            capabilities=SimpleNamespace(
                supports_segment_timestamps=True,
                supports_alignment=False,
                timestamp_guarantee="text-only",
            ),
        )
        with patch("cli.list_backend_status", return_value=[enabled_backend]):
            with patch("cli.get_backend", return_value=registration):
                with patch("cli.resolve_model_path", return_value=Path("models/qwen3-asr")):
                    with patch(
                        "cli.resolve_device_with_details",
                        return_value=SimpleNamespace(
                            requested="auto",
                            resolved="cpu",
                            reason="--device auto selected cpu because torch CUDA probe reported unavailable",
                        ),
                    ):
                        with redirect_stdout(stdout):
                            code = cli.main(
                                [
                                    "--asr-preflight-only",
                                    "--asr-backend",
                                    "qwen3-asr",
                                    "--model-path",
                                    "models/qwen3-asr",
                                    "--device",
                                    "auto",
                                ]
                            )

        self.assertEqual(code, 0)
        stdout_lines = stdout.getvalue().splitlines()
        self.assertEqual(len(stdout_lines), 1)
        payload = json.loads(stdout_lines[0])
        self.assertEqual(payload["mode"], "asr_preflight_only")
        self.assertEqual(payload["backend"], "qwen3-asr")
        self.assertEqual(payload["capabilities"]["timestamp_guarantee"], "text-only")
        self.assertFalse(payload["capabilities"]["supports_alignment"])
        self.assertEqual(payload["model_resolution"]["resolved_model_path"], "models/qwen3-asr")
        self.assertEqual(payload["device"]["requested"], "auto")
        self.assertEqual(payload["device"]["compute_type"], "auto")
        self.assertEqual(payload["device"]["cuda_probe_label"], "torch")
        self.assertEqual(
            payload["device"]["resolution_reason"],
            "--device auto selected cpu because torch CUDA probe reported unavailable",
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


    def test_alignment_preflight_only_success_outputs_deterministic_json(self) -> None:
        stdout = io.StringIO()
        alignment_backend = SimpleNamespace(
            name="qwen3-forced-aligner",
            enabled=True,
            missing_dependencies=(),
            reason="enabled",
            install_extra="asr_qwen3",
            supports_alignment=True,
        )
        registration = SimpleNamespace(
            name="qwen3-forced-aligner",
            capabilities=SimpleNamespace(
                supports_segment_timestamps=False,
                supports_alignment=True,
                timestamp_guarantee="alignment-required",
            ),
        )
        with patch("cli.list_backend_status", return_value=[alignment_backend]):
            with patch("cli.get_backend", return_value=registration):
                with patch("cli.resolve_model_path", return_value=Path("models/qwen3-forced-aligner")):
                    with patch(
                        "cli.resolve_device_with_details",
                        return_value=SimpleNamespace(
                            requested="auto",
                            resolved="cpu",
                            reason="--device auto selected cpu because torch CUDA probe reported unavailable",
                        ),
                    ):
                        with redirect_stdout(stdout):
                            code = cli.main(
                                [
                                    "--alignment-preflight-only",
                                    "--asr-backend",
                                    "qwen3-forced-aligner",
                                    "--model-path",
                                    "models/qwen3-forced-aligner",
                                ]
                            )

        self.assertEqual(code, 0)
        stdout_lines = stdout.getvalue().splitlines()
        self.assertEqual(len(stdout_lines), 1)
        payload = json.loads(stdout_lines[0])
        self.assertEqual(payload["mode"], "alignment_preflight_only")
        self.assertEqual(payload["backend"], "qwen3-forced-aligner")
        self.assertTrue(payload["capabilities"]["supports_alignment"])
        self.assertEqual(payload["capabilities"]["timestamp_guarantee"], "alignment-required")
        self.assertEqual(payload["model_resolution"]["resolved_model_path"], "models/qwen3-forced-aligner")
        self.assertEqual(payload["device"]["cuda_probe_label"], "ctranslate2")

    def test_alignment_preflight_only_rejects_non_alignment_backends(self) -> None:
        stderr = io.StringIO()
        statuses = [
            SimpleNamespace(name="qwen3-asr", enabled=True, missing_dependencies=(), reason="enabled", install_extra="asr_qwen3", supports_alignment=False),
            SimpleNamespace(name="qwen3-forced-aligner", enabled=True, missing_dependencies=(), reason="enabled", install_extra="asr_qwen3", supports_alignment=True),
        ]
        fake_registration = SimpleNamespace(
            name="qwen3-asr",
            backend_class=object,
            capabilities=SimpleNamespace(supports_segment_timestamps=True, supports_alignment=False, timestamp_guarantee="text-only"),
        )
        with patch("cli.list_backend_status", return_value=statuses):
            with patch("cli.get_backend", return_value=fake_registration):
                with redirect_stderr(stderr):
                    code = cli.main(
                [
                    "--alignment-preflight-only",
                    "--asr-backend",
                    "faster-whisper",
                    "--model-path",
                    "models/faster-whisper/tiny",
                ]
            )

        self.assertEqual(code, 1)
        self.assertIn("is not an alignment backend", stderr.getvalue())

    def test_alignment_preflight_only_model_resolution_failure_exits_one(self) -> None:
        stderr = io.StringIO()
        alignment_backend = SimpleNamespace(
            name="qwen3-forced-aligner",
            enabled=True,
            missing_dependencies=(),
            reason="enabled",
            install_extra="asr_qwen3",
            supports_alignment=True,
        )
        registration = SimpleNamespace(
            name="qwen3-forced-aligner",
            capabilities=SimpleNamespace(
                supports_segment_timestamps=False,
                supports_alignment=True,
                timestamp_guarantee="alignment-required",
            ),
        )
        with patch("cli.list_backend_status", return_value=[alignment_backend]):
            with patch("cli.get_backend", return_value=registration):
                with patch(
                    "cli.resolve_model_path",
                    side_effect=cli.ModelResolutionError("Resolved qwen3-forced-aligner model is missing required artifacts."),
                ):
                    with redirect_stderr(stderr):
                        code = cli.main(["--alignment-preflight-only", "--asr-backend", "qwen3-forced-aligner"])

        self.assertEqual(code, 1)
        self.assertIn("missing required artifacts", stderr.getvalue())

    def test_preflight_modes_are_mutually_exclusive(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = cli.main(["--asr-preflight-only", "--alignment-preflight-only"])

        self.assertEqual(code, 1)
        self.assertIn("Use only one preflight mode", stderr.getvalue())



    def test_missing_required_args_exits_one(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = cli.main([])

        self.assertEqual(code, 1)
        self.assertIn("Missing required arguments", stderr.getvalue())

    def test_text_only_timestamp_backend_rejected_when_deterministic_timestamps_required(self) -> None:
        stderr = io.StringIO()
        registration = SimpleNamespace(
            name="qwen3-asr",
            capabilities=SimpleNamespace(
                supports_segment_timestamps=True,
                supports_alignment=False,
                timestamp_guarantee="text-only",
            ),
        )
        enabled_status = SimpleNamespace(
            name="qwen3-asr",
            enabled=True,
            missing_dependencies=(),
            reason="enabled",
            install_extra="asr_qwen3",
            supports_alignment=False,
        )
        with patch("cli.list_backend_status", return_value=[enabled_status]):
            with patch("cli.get_backend", return_value=registration):
                with patch("cli.validate_backend_capabilities") as validate_patch:
                    validate_patch.side_effect = ValueError("ASR backend 'qwen3-asr' is text-first and does not guarantee deterministic timestamps.")
                    with redirect_stderr(stderr):
                        code = cli.main([
                            "--input", "in.wav", "--script", "script.tsv", "--out", "out", "--asr-backend", "qwen3-asr",
                        ], which=lambda _: "/usr/bin/ffmpeg", runner=lambda *args, **kwargs: subprocess.CompletedProcess(args, 0))

        self.assertEqual(code, 1)
        self.assertIn("does not guarantee deterministic timestamps", stderr.getvalue())

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

    def test_declared_but_disabled_backends_exits_with_actionable_error(self) -> None:
        scenarios = (
            {
                "backend_mode_label": "qwen3-asr/asr-mode",
                "backend_name": "qwen3-asr",
                "missing_dependencies": ("torch", "qwen_asr"),
                "install_extra": "asr_qwen3",
            },
            {
                "backend_mode_label": "vibevoice/asr-mode",
                "backend_name": "vibevoice",
                "missing_dependencies": ("vibevoice", "torch"),
                "install_extra": "asr_vibevoice",
            },
        )

        for scenario in scenarios:
            with self.subTest(backend_mode_label=scenario["backend_mode_label"]):
                stderr = io.StringIO()
                disabled_backend = SimpleNamespace(
                    name=scenario["backend_name"],
                    enabled=False,
                    missing_dependencies=scenario["missing_dependencies"],
                    install_extra=scenario["install_extra"],
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
                                scenario["backend_name"],
                            ],
                            which=lambda _: "/usr/bin/ffmpeg",
                            runner=lambda *args, **kwargs: subprocess.CompletedProcess(args, 0),
                        )

                self.assertEqual(code, 1)
                self.assertIn("declared but currently disabled", stderr.getvalue())
                expected_missing = ", ".join(scenario["missing_dependencies"])
                self.assertIn(f"Missing optional dependencies: {expected_missing}", stderr.getvalue())
                self.assertIn(
                    f"pip install 'cutscene-locator[{scenario['install_extra']}]'",
                    stderr.getvalue(),
                )

    def test_declared_but_disabled_backend_without_missing_deps_uses_reason_without_install_hint(self) -> None:
        stderr = io.StringIO()
        disabled_backend = SimpleNamespace(
            name="experimental-asr",
            enabled=False,
            missing_dependencies=(),
            reason="feature flag disabled",
            install_extra="asr_experimental",
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
                        "experimental-asr",
                    ],
                    which=lambda _: "/usr/bin/ffmpeg",
                    runner=lambda *args, **kwargs: subprocess.CompletedProcess(args, 0),
                )

        self.assertEqual(code, 1)
        self.assertIn("declared but currently disabled", stderr.getvalue())
        self.assertIn("Reason: feature flag disabled", stderr.getvalue())
        self.assertNotIn("Install with:", stderr.getvalue())

    def test_enabled_backend_path_unchanged(self) -> None:
        stderr = io.StringIO()
        enabled_backend = SimpleNamespace(
            name="mock",
            enabled=True,
            missing_dependencies=(),
            reason="enabled",
            install_extra=None,
            supports_alignment=False,
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


    def test_unknown_backend_returns_registry_error_when_status_is_missing(self) -> None:
        stderr = io.StringIO()
        with patch("cli.list_backend_status", return_value=[]):
            with patch("cli.get_backend", side_effect=ValueError("Unknown ASR backend 'not-real'.")):
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
                            "not-real",
                        ],
                        which=lambda _: "/usr/bin/ffmpeg",
                        runner=lambda *args, **kwargs: subprocess.CompletedProcess(args, 0),
                    )

        self.assertEqual(code, 1)
        self.assertIn("Unknown ASR backend 'not-real'.", stderr.getvalue())

    def test_alignment_backend_rejected_in_asr_mode_with_deterministic_error(self) -> None:
        stderr = io.StringIO()
        alignment_backend = SimpleNamespace(
            name="qwen3-forced-aligner",
            enabled=True,
            missing_dependencies=(),
            reason="enabled",
            install_extra="asr_qwen3",
            supports_alignment=True,
        )

        with patch("cli.list_backend_status", return_value=[alignment_backend]):
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
                        "qwen3-forced-aligner",
                    ],
                    which=lambda _: "/usr/bin/ffmpeg",
                    runner=lambda *args, **kwargs: subprocess.CompletedProcess(args, 0),
                )

        self.assertEqual(code, 1)
        self.assertEqual(
            stderr.getvalue().strip(),
            "Error: 'qwen3-forced-aligner' is an alignment backend and cannot be used with --asr-backend. "
            "Use the explicit alignment pipeline path and alignment input contract "
            "(`reference_spans[]`) instead of ASR-only transcription mode.",
        )

    def test_alignment_backends_are_rejected_in_asr_mode_before_disabled_dependency_diagnostics(self) -> None:
        scenarios = (
            {
                "backend_mode_label": "qwen3-asr/asr-mode",
                "backend_name": "qwen3-asr",
                "backend_registration": SimpleNamespace(
                    name="qwen3-asr",
                    capabilities=SimpleNamespace(supports_alignment=True),
                ),
                "status_rows": [],
                "expected_fragments": ("alignment backend", "alignment pipeline path"),
            },
            {
                "backend_mode_label": "qwen-forced-aligner/asr-mode",
                "backend_name": "qwen3-forced-aligner",
                "backend_registration": None,
                "status_rows": [
                    SimpleNamespace(
                        name="qwen3-forced-aligner",
                        enabled=False,
                        missing_dependencies=("qwen_asr",),
                        reason="missing optional dependencies: qwen_asr",
                        install_extra="asr_qwen3",
                        supports_alignment=True,
                    )
                ],
                "expected_fragments": ("alignment backend", "reference_spans[]"),
            },
        )

        for scenario in scenarios:
            with self.subTest(backend_mode_label=scenario["backend_mode_label"]):
                stderr = io.StringIO()
                with patch("cli.list_backend_status", return_value=scenario["status_rows"]):
                    with redirect_stderr(stderr):
                        if scenario["backend_registration"] is None:
                            code = cli.main(
                                [
                                    "--input",
                                    "in.wav",
                                    "--script",
                                    "script.tsv",
                                    "--out",
                                    "out",
                                    "--asr-backend",
                                    scenario["backend_name"],
                                ],
                                which=lambda _: "/usr/bin/ffmpeg",
                                runner=lambda *args, **kwargs: subprocess.CompletedProcess(args, 0),
                            )
                        else:
                            with patch("cli.get_backend", return_value=scenario["backend_registration"]):
                                code = cli.main(
                                    [
                                        "--input",
                                        "in.wav",
                                        "--script",
                                        "script.tsv",
                                        "--out",
                                        "out",
                                        "--asr-backend",
                                        scenario["backend_name"],
                                    ],
                                    which=lambda _: "/usr/bin/ffmpeg",
                                    runner=lambda *args, **kwargs: subprocess.CompletedProcess(args, 0),
                                )

                self.assertEqual(code, 1)
                for fragment in scenario["expected_fragments"]:
                    self.assertIn(fragment, stderr.getvalue())
                self.assertNotIn("declared but currently disabled", stderr.getvalue())

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



    def test_matching_knob_defaults_parse(self) -> None:
        args = cli.build_parser().parse_args([])
        self.assertEqual(args.chunk, 300)
        self.assertEqual(args.match_quick_threshold, 0.25)
        self.assertEqual(args.match_monotonic_window, 0)
        self.assertEqual(args.match_threshold, 0.85)

    def test_asr_knob_defaults_parse(self) -> None:
        args = cli.build_parser().parse_args([])
        self.assertIsNone(args.asr_language)
        self.assertEqual(args.asr_beam_size, 1)
        self.assertEqual(args.asr_temperature, 0.0)
        self.assertEqual(args.asr_best_of, 1)
        self.assertIsNone(args.asr_no_speech_threshold)
        self.assertIsNone(args.asr_logprob_threshold)
        self.assertIsNone(args.qwen3_batch_size)
        self.assertIsNone(args.qwen3_chunk_length_s)

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

    def test_invalid_qwen3_batch_size_exits_one(self) -> None:
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
                    "--qwen3-batch-size",
                    "0",
                ],
                which=lambda _: "/usr/bin/ffmpeg",
                runner=lambda *args, **kwargs: subprocess.CompletedProcess(args, 0),
            )

        self.assertEqual(code, 1)
        self.assertIn("Invalid --qwen3-batch-size value", stderr.getvalue())

    def test_invalid_qwen3_chunk_length_exits_one(self) -> None:
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
                    "--qwen3-chunk-length-s",
                    "0",
                ],
                which=lambda _: "/usr/bin/ffmpeg",
                runner=lambda *args, **kwargs: subprocess.CompletedProcess(args, 0),
            )

        self.assertEqual(code, 1)
        self.assertIn("Invalid --qwen3-chunk-length-s value", stderr.getvalue())

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

    def test_worker_failure_non_verbose_keeps_output_clean(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "audio.wav"
            audio_path.write_text("fake", encoding="utf-8")
            config = cli.ASRConfig(
                backend_name="faster-whisper",
                model_path=Path("models/faster-whisper"),
                device="cpu",
                compute_type="float32",
            )
            stdout = io.StringIO()
            stderr = io.StringIO()

            completed = subprocess.CompletedProcess(
                ["worker"],
                17,
                stdout="worker stdout trace\n",
                stderr="worker stderr trace\n",
            )
            with patch("cli.subprocess.run", return_value=completed):
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    with self.assertRaisesRegex(cli.CliError, "ASR worker failed with exit code 17") as ctx:
                        cli._run_faster_whisper_subprocess(
                            audio_path=audio_path,
                            resolved_model_path=Path("models/faster-whisper"),
                            asr_config=config,
                            verbose=False,
                        )

        self.assertNotIn("worker stdout trace", str(ctx.exception))
        self.assertNotIn("worker stderr trace", str(ctx.exception))
        self.assertEqual(stdout.getvalue().replace("\r\n", "\n"), "")
        self.assertEqual(stderr.getvalue().replace("\r\n", "\n"), "")

    def test_worker_failure_verbose_prints_stdout_then_stderr(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "audio.wav"
            audio_path.write_text("fake", encoding="utf-8")
            config = cli.ASRConfig(
                backend_name="faster-whisper",
                model_path=Path("models/faster-whisper"),
                device="cpu",
                compute_type="float32",
            )
            stdout = io.StringIO()
            stderr = io.StringIO()

            completed = subprocess.CompletedProcess(
                ["worker"],
                19,
                stdout="worker stdout trace\n",
                stderr="worker stderr trace\n",
            )
            with patch("cli.subprocess.run", return_value=completed):
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    with self.assertRaisesRegex(cli.CliError, "ASR worker failed with exit code 19") as ctx:
                        cli._run_faster_whisper_subprocess(
                            audio_path=audio_path,
                            resolved_model_path=Path("models/faster-whisper"),
                            asr_config=config,
                            verbose=True,
                        )

        self.assertNotIn("worker stdout trace", str(ctx.exception))
        self.assertNotIn("worker stderr trace", str(ctx.exception))

        self.assertEqual(
            stdout.getvalue().replace("\r\n", "\n"),
            "----- ASR worker stdout -----\nworker stdout trace\n",
        )
        self.assertEqual(
            stderr.getvalue().replace("\r\n", "\n"),
            "----- ASR worker stderr -----\nworker stderr trace\n",
        )

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


    def test_worker_subprocess_forwards_selected_asr_backend(self) -> None:
        fake_payload = {
            "segments": [{"segment_id": "seg_0001", "start": 0.0, "end": 1.0, "text": "ok"}],
            "meta": {
                "backend": "whisperx",
                "model": "tiny",
                "version": "1.0.0",
                "device": "cpu",
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "audio.wav"
            audio_path.write_text("fake", encoding="utf-8")
            config = cli.ASRConfig(
                backend_name="whisperx",
                model_path=Path("models/whisperx"),
                device="cpu",
                compute_type="float32",
                qwen3_batch_size=3,
                qwen3_chunk_length_s=12.5,
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
                cli._run_faster_whisper_subprocess(
                    audio_path=audio_path,
                    resolved_model_path=Path("models/whisperx"),
                    asr_config=config,
                    verbose=False,
                )

        self.assertIn("--asr-backend", captured_cmd)
        self.assertEqual(captured_cmd[captured_cmd.index("--asr-backend") + 1], "whisperx")
        self.assertEqual(captured_cmd[captured_cmd.index("--qwen3-batch-size") + 1], "3")
        self.assertEqual(captured_cmd[captured_cmd.index("--qwen3-chunk-length-s") + 1], "12.5")
        self.assertEqual(Path(captured_cmd[captured_cmd.index("--audio-path") + 1]), audio_path)
        self.assertEqual(Path(captured_cmd[captured_cmd.index("--model-path") + 1]), Path("models/whisperx"))


    def test_worker_subprocess_result_is_reparsed_with_contract_validator(self) -> None:
        fake_payload = {
            "segments": [{"segment_id": "seg_0001", "start": 0.0, "end": 1.0, "text": "ok"}],
            "meta": {
                "backend": "whisperx",
                "model": "tiny",
                "version": "1.0.0",
                "device": "cpu",
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "audio.wav"
            audio_path.write_text("fake", encoding="utf-8")
            config = cli.ASRConfig(
                backend_name="whisperx",
                model_path=Path("models/whisperx"),
                device="cpu",
                compute_type="float32",
            )

            def _fake_run(cmd, check, capture_output, text):
                del check, capture_output, text
                result_path = Path(cmd[cmd.index("--result-path") + 1])
                result_path.parent.mkdir(parents=True, exist_ok=True)
                result_path.write_text(json.dumps(fake_payload), encoding="utf-8")
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

            with patch("cli.subprocess.run", side_effect=_fake_run):
                with patch("cli.parse_asr_result", wraps=cli.parse_asr_result) as parse_mock:
                    result = cli._run_faster_whisper_subprocess(
                        audio_path=audio_path,
                        resolved_model_path=Path("models/whisperx"),
                        asr_config=config,
                        verbose=False,
                    )

        parse_mock.assert_called_once()
        self.assertEqual(result["meta"]["backend"], "whisperx")

    def test_faster_whisper_worker_subprocess_rejects_untyped_segment_payload(self) -> None:
        fake_payload = {
            "segments": [{"start": 0.0, "end": 1.0, "text": "ok"}],
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

            def _fake_run(cmd, check, capture_output, text):
                del check, capture_output, text
                result_path = Path(cmd[cmd.index("--result-path") + 1])
                result_path.parent.mkdir(parents=True, exist_ok=True)
                result_path.write_text(json.dumps(fake_payload), encoding="utf-8")
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

            with patch("cli.subprocess.run", side_effect=_fake_run):
                with self.assertRaisesRegex(ValueError, r"segments\[0\]\.segment_id must be a non-empty string"):
                    cli._run_faster_whisper_subprocess(
                        audio_path=audio_path,
                        resolved_model_path=Path("models/faster-whisper"),
                        asr_config=config,
                        verbose=False,
                    )

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


@unittest.skipUnless(
    os.environ.get("CUTSCENE_QWEN3_INIT_SMOKE") == "1",
    "Set CUTSCENE_QWEN3_INIT_SMOKE=1 and CUTSCENE_QWEN3_MODEL_PATH=<local_model_dir> to run qwen3 loader init smoke tests.",
)
class Qwen3ReadinessSmokeTests(unittest.TestCase):
    def test_qwen3_from_pretrained_init_only_smoke(self) -> None:
        """Optional init-only smoke test.

        Usage:
        - CUTSCENE_QWEN3_INIT_SMOKE=1
        - CUTSCENE_QWEN3_MODEL_PATH=/absolute/or/relative/path/to/local/qwen3-asr/snapshot

        This smoke test verifies that qwen_asr can initialize `Qwen3ASRModel` from a local
        snapshot path without running full transcription/inference, so default CI stays
        deterministic and offline.
        """

        local_model_path = os.environ.get("CUTSCENE_QWEN3_MODEL_PATH")
        if not local_model_path:
            self.skipTest("CUTSCENE_QWEN3_MODEL_PATH is required when CUTSCENE_QWEN3_INIT_SMOKE=1")

        model_path = Path(local_model_path)
        if not model_path.exists():
            self.skipTest(f"CUTSCENE_QWEN3_MODEL_PATH does not exist: {model_path}")

        try:
            qwen_asr_module = __import__("qwen_asr")
        except ModuleNotFoundError as exc:
            self.skipTest(f"Optional qwen_asr dependency is not installed: {exc}")

        qwen_model_class = getattr(qwen_asr_module, "Qwen3ASRModel", None)
        if qwen_model_class is None:
            self.skipTest("Installed qwen_asr package does not expose Qwen3ASRModel")

        call_kwargs: dict[str, object] = {"dtype": "auto"}
        model = qwen_model_class.from_pretrained(str(model_path), **call_kwargs)

        self.assertNotIn(
            "device",
            call_kwargs,
            "qwen3 loader init smoke must not pass `device` into from_pretrained(); "
            "device transfer happens post-load via .to(device).",
        )

        torch_module = getattr(model, "model", None)
        if torch_module is not None and callable(getattr(torch_module, "to", None)):
            torch_module.to("cpu")
        elif callable(getattr(model, "to", None)):
            model.to("cpu")
        else:
            self.fail(
                "Installed qwen_asr runtime does not expose a supported post-load device transfer API "
                "(expected model.model.to('cpu') or model.to('cpu'))."
            )

        self.assertIsNotNone(model)


@unittest.skipUnless(
    os.environ.get("CUTSCENE_QWEN3_RUNTIME_SMOKE") == "1",
    "Set CUTSCENE_QWEN3_RUNTIME_SMOKE=1, CUTSCENE_QWEN3_MODEL_PATH=<local_model_dir>, and "
    "CUTSCENE_QWEN3_RUNTIME_AUDIO=<local_audio_file> to run qwen3 runtime smoke.",
)
class Qwen3RuntimeSmokeTests(unittest.TestCase):
    def test_qwen3_runtime_smoke_is_explicitly_env_gated(self) -> None:
        """Optional runtime smoke test (full inference).

        Usage:
        - CUTSCENE_QWEN3_RUNTIME_SMOKE=1
        - CUTSCENE_QWEN3_MODEL_PATH=/absolute/or/relative/path/to/local/qwen3-asr/snapshot
        - CUTSCENE_QWEN3_RUNTIME_AUDIO=/absolute/or/relative/path/to/local/audio.wav

        This remains disabled by default to keep CI deterministic and offline.
        """

        local_model_path = os.environ.get("CUTSCENE_QWEN3_MODEL_PATH")
        runtime_audio_path = os.environ.get("CUTSCENE_QWEN3_RUNTIME_AUDIO")

        if not local_model_path:
            self.skipTest("CUTSCENE_QWEN3_MODEL_PATH is required when CUTSCENE_QWEN3_RUNTIME_SMOKE=1")
        if not runtime_audio_path:
            self.skipTest("CUTSCENE_QWEN3_RUNTIME_AUDIO is required when CUTSCENE_QWEN3_RUNTIME_SMOKE=1")

        model_path = Path(local_model_path)
        audio_path = Path(runtime_audio_path)
        if not model_path.exists():
            self.skipTest(f"CUTSCENE_QWEN3_MODEL_PATH does not exist: {model_path}")
        if not audio_path.exists():
            self.skipTest(f"CUTSCENE_QWEN3_RUNTIME_AUDIO does not exist: {audio_path}")

        try:
            from src.asr.qwen3_asr_backend import Qwen3ASRBackend
            from src.asr.config import ASRConfig
        except ModuleNotFoundError as exc:
            self.skipTest(f"Optional qwen3 runtime dependencies are not installed: {exc}")

        backend = Qwen3ASRBackend()
        result = backend.transcribe(
            str(audio_path),
            ASRConfig(
                backend_name="qwen3-asr",
                model_path=model_path,
                device="cpu",
                compute_type="auto",
            ),
        )

        self.assertIsInstance(result.get("segments"), list)
        self.assertGreater(len(result["segments"]), 0)
        self.assertEqual(result["meta"]["backend"], "qwen3-asr")


class CliAdapterDispatchTests(unittest.TestCase):
    def test_alignment_mode_runs_forced_aligner_and_produces_timestamped_outputs(self) -> None:
        class _MockAlignmentBackend:
            def __init__(self, config: object):
                self.config = config

            def align(self, audio_path: str, reference_spans: list[dict[str, str]]) -> dict[str, object]:
                self_audio_path = audio_path
                del self_audio_path
                spans: list[dict[str, object]] = []
                for index, ref in enumerate(reference_spans):
                    start = float(index)
                    spans.append(
                        {
                            "span_id": ref["ref_id"],
                            "start": start,
                            "end": start + 0.75,
                            "text": ref["text"],
                            "confidence": 1.0,
                        }
                    )
                return {
                    "transcript_text": " ".join(ref["text"] for ref in reference_spans),
                    "spans": spans,
                    "meta": {
                        "backend": "qwen3-forced-aligner",
                        "version": "test",
                        "device": "cpu",
                    },
                }

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "out"
            registration = SimpleNamespace(
                name="qwen3-forced-aligner",
                backend_class=_MockAlignmentBackend,
                capabilities=SimpleNamespace(
                    supports_segment_timestamps=True,
                    supports_alignment=True,
                    timestamp_guarantee="alignment-required",
                ),
            )
            alignment_backend_status = SimpleNamespace(
                name="qwen3-forced-aligner",
                enabled=True,
                missing_dependencies=(),
                reason="enabled",
                install_extra="asr_qwen3",
                supports_alignment=True,
            )
            fake_preprocess = SimpleNamespace(canonical_wav_path=Path("out/_tmp/canonical.wav"), chunk_metadata=[])
            with patch("cli.list_backend_status", return_value=[alignment_backend_status]):
                with patch("cli.get_backend", return_value=registration):
                    with patch("cli.resolve_model_path", return_value=Path("models/qwen3-forced-aligner")):
                        with patch("cli.preprocess_media", return_value=fake_preprocess):
                            with patch("cli.load_script_table", wraps=cli.load_script_table) as load_script_table_call:
                                code = cli.main(
                                    [
                                        "--input",
                                        "in.wav",
                                        "--script",
                                        "tests/fixtures/script_sample.tsv",
                                        "--out",
                                        str(out_dir),
                                        "--alignment-backend",
                                        "qwen3-forced-aligner",
                                        "--alignment-model-path",
                                        "models/qwen3-forced-aligner",
                                        "--chunk",
                                        "0",
                                    ],
                                    which=lambda _: "/usr/bin/ffmpeg",
                                    runner=lambda *args, **kwargs: subprocess.CompletedProcess(args, 0),
                                )

                                self.assertEqual(code, 0)
                                self.assertEqual(load_script_table_call.call_count, 1)
                                matches_path = out_dir / "matches.csv"
                                self.assertTrue(matches_path.exists())
                                lines = matches_path.read_text(encoding="utf-8").splitlines()
                                self.assertGreater(len(lines), 1)
                                self.assertIn(",0,0.75,", lines[1])


    def test_two_stage_qwen3_run_uses_stage1_text_then_stage2_alignment_timestamps(self) -> None:
        class _MockQwenAsrBackend:
            def transcribe(self, audio_path: str, config: object) -> dict[str, object]:
                del audio_path, config
                return {
                    "segments": [
                        {"segment_id": "seg_0001", "start": 0.0, "end": 0.5, "text": "You don't get it."},
                        {"segment_id": "seg_0002", "start": 0.5, "end": 1.0, "text": "Yeah."},
                    ],
                    "meta": {"backend": "qwen3-asr", "model": "fixture", "version": "1", "device": "cpu"},
                }

        class _MockAlignerBackend:
            def __init__(self, config: object):
                self.config = config

            def align(self, audio_path: str, reference_spans: list[dict[str, str]]) -> dict[str, object]:
                del audio_path
                self.last_reference_spans = reference_spans
                return {
                    "transcript_text": " ".join(item["text"] for item in reference_spans),
                    "spans": [
                        {
                            "span_id": reference_spans[0]["ref_id"],
                            "start": 10.0,
                            "end": 10.8,
                            "text": reference_spans[0]["text"],
                            "confidence": 1.0,
                        },
                        {
                            "span_id": reference_spans[1]["ref_id"],
                            "start": 11.0,
                            "end": 11.6,
                            "text": reference_spans[1]["text"],
                            "confidence": 1.0,
                        },
                    ],
                    "meta": {"backend": "qwen3-forced-aligner", "version": "test", "device": "cpu"},
                }

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "out"
            fake_preprocess = SimpleNamespace(canonical_wav_path=Path("out/_tmp/canonical.wav"), chunk_metadata=[])
            statuses = [
                SimpleNamespace(name="qwen3-asr", enabled=True, missing_dependencies=(), reason="enabled", install_extra="asr_qwen3", supports_alignment=False),
                SimpleNamespace(name="qwen3-forced-aligner", enabled=True, missing_dependencies=(), reason="enabled", install_extra="asr_qwen3", supports_alignment=True),
            ]

            def fake_get_backend(name: str) -> SimpleNamespace:
                if name == "qwen3-asr":
                    return SimpleNamespace(
                        name="qwen3-asr",
                        backend_class=_MockQwenAsrBackend,
                        capabilities=SimpleNamespace(supports_segment_timestamps=True, supports_alignment=False, timestamp_guarantee="text-only"),
                    )
                if name == "qwen3-forced-aligner":
                    return SimpleNamespace(
                        name="qwen3-forced-aligner",
                        backend_class=_MockAlignerBackend,
                        capabilities=SimpleNamespace(supports_segment_timestamps=True, supports_alignment=True, timestamp_guarantee="alignment-required"),
                    )
                raise AssertionError(name)

            stage1_result = {
                "segments": [
                    {"segment_id": "seg_0001", "start": 0.0, "end": 0.5, "text": "You don't get it."},
                    {"segment_id": "seg_0002", "start": 0.5, "end": 1.0, "text": "Yeah."},
                ],
                "meta": {"backend": "qwen3-asr", "model": "fixture", "version": "1", "device": "cpu"},
            }
            with patch("cli.list_backend_status", return_value=statuses):
                with patch("cli.get_backend", side_effect=fake_get_backend):
                    with patch("cli.dispatch_asr_transcription", return_value=stage1_result):
                        with patch("cli.preprocess_media", return_value=fake_preprocess):
                            with patch("cli.resolve_model_path", side_effect=[Path("models/qwen3-asr"), Path("models/qwen3-aligner")]):
                                code = cli.main(
                                [
                                    "--input", "in.wav",
                                    "--script", "tests/fixtures/script_sample.tsv",
                                    "--out", str(out_dir),
                                    "--asr-backend", "qwen3-asr",
                                    "--alignment-model-path", "models/qwen3-aligner",
                                    "--two-stage-qwen3",
                                    "--chunk", "0",
                                    "--match-threshold", "0.0",
                                ],
                                which=lambda _: "/usr/bin/ffmpeg",
                                runner=lambda *args, **kwargs: subprocess.CompletedProcess(args, 0),
                            )

            self.assertEqual(code, 0)
            lines = (out_dir / "matches.csv").read_text(encoding="utf-8").splitlines()
            self.assertIn(",10,10.8,", lines[1])
            self.assertIn(",11,11.6,", lines[2])

    def test_two_stage_qwen3_missing_alignment_model_fails_deterministically(self) -> None:
        stderr = io.StringIO()
        statuses = [
            SimpleNamespace(name="qwen3-asr", enabled=True, missing_dependencies=(), reason="enabled", install_extra="asr_qwen3", supports_alignment=False),
            SimpleNamespace(name="qwen3-forced-aligner", enabled=True, missing_dependencies=(), reason="enabled", install_extra="asr_qwen3", supports_alignment=True),
        ]
        fake_registration = SimpleNamespace(
            name="qwen3-asr",
            backend_class=object,
            capabilities=SimpleNamespace(supports_segment_timestamps=True, supports_alignment=False, timestamp_guarantee="text-only"),
        )
        with patch("cli.list_backend_status", return_value=statuses):
            with patch("cli.get_backend", return_value=fake_registration):
                with redirect_stderr(stderr):
                    code = cli.main(
                    [
                        "--input", "in.wav",
                        "--script", "tests/fixtures/script_sample.tsv",
                        "--out", "out",
                        "--asr-backend", "qwen3-asr",
                        "--two-stage-qwen3",
                        "--chunk", "0",
                    ],
                    which=lambda _: "/usr/bin/ffmpeg",
                    runner=lambda *args, **kwargs: subprocess.CompletedProcess(args, 0),
                )

        self.assertEqual(code, 1)
        self.assertIn("--two-stage-qwen3 requires --alignment-model-path or --alignment-model-id", stderr.getvalue())

    def test_two_stage_qwen3_fails_when_aligner_dependencies_missing(self) -> None:
        stderr = io.StringIO()
        statuses = [
            SimpleNamespace(name="qwen3-asr", enabled=True, missing_dependencies=(), reason="enabled", install_extra="asr_qwen3", supports_alignment=False),
            SimpleNamespace(name="qwen3-forced-aligner", enabled=False, missing_dependencies=("torch", "transformers"), reason="missing optional deps", install_extra="asr_qwen3", supports_alignment=True),
        ]
        with patch("cli.list_backend_status", return_value=statuses):
            with redirect_stderr(stderr):
                code = cli.main(
                    [
                        "--input", "in.wav",
                        "--script", "tests/fixtures/script_sample.tsv",
                        "--out", "out",
                        "--asr-backend", "qwen3-asr",
                        "--alignment-model-path", "models/qwen3-aligner",
                        "--two-stage-qwen3",
                    ],
                    which=lambda _: "/usr/bin/ffmpeg",
                    runner=lambda *args, **kwargs: subprocess.CompletedProcess(args, 0),
                )

        self.assertEqual(code, 1)
        self.assertIn("Two-stage qwen3 flow requires enabled alignment backend 'qwen3-forced-aligner'", stderr.getvalue())
        self.assertIn("Missing optional dependencies: torch, transformers", stderr.getvalue())

    def test_two_stage_qwen3_verbose_logs_stage_boundaries(self) -> None:
        class _MockQwenAsrBackend:
            def transcribe(self, audio_path: str, config: object) -> dict[str, object]:
                del audio_path, config
                return {
                    "segments": [{"segment_id": "seg_0001", "start": 0.0, "end": 0.5, "text": "You don't get it."}],
                    "meta": {"backend": "qwen3-asr", "model": "fixture", "version": "1", "device": "cpu"},
                }

        class _MockAlignerBackend:
            def __init__(self, config: object):
                self.config = config

            def align(self, audio_path: str, reference_spans: list[dict[str, str]]) -> dict[str, object]:
                del audio_path
                return {
                    "transcript_text": reference_spans[0]["text"],
                    "spans": [{"span_id": reference_spans[0]["ref_id"], "start": 3.0, "end": 3.5, "text": reference_spans[0]["text"], "confidence": 1.0}],
                    "meta": {"backend": "qwen3-forced-aligner", "version": "test", "device": "cpu"},
                }

        stdout = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "out"
            fake_preprocess = SimpleNamespace(canonical_wav_path=Path("out/_tmp/canonical.wav"), chunk_metadata=[])

            def fake_get_backend(name: str) -> SimpleNamespace:
                if name == "qwen3-asr":
                    return SimpleNamespace(name="qwen3-asr", backend_class=_MockQwenAsrBackend, capabilities=SimpleNamespace(supports_segment_timestamps=True, supports_alignment=False, timestamp_guarantee="text-only"))
                return SimpleNamespace(name="qwen3-forced-aligner", backend_class=_MockAlignerBackend, capabilities=SimpleNamespace(supports_segment_timestamps=True, supports_alignment=True, timestamp_guarantee="alignment-required"))

            statuses = [
                SimpleNamespace(name="qwen3-asr", enabled=True, missing_dependencies=(), reason="enabled", install_extra="asr_qwen3", supports_alignment=False),
                SimpleNamespace(name="qwen3-forced-aligner", enabled=True, missing_dependencies=(), reason="enabled", install_extra="asr_qwen3", supports_alignment=True),
            ]
            stage1_result = {
                "segments": [{"segment_id": "seg_0001", "start": 0.0, "end": 0.5, "text": "You don't get it."}],
                "meta": {"backend": "qwen3-asr", "model": "fixture", "version": "1", "device": "cpu"},
            }
            with patch("cli.list_backend_status", return_value=statuses):
                with patch("cli.get_backend", side_effect=fake_get_backend):
                    with patch("cli.dispatch_asr_transcription", return_value=stage1_result):
                        with patch("cli.preprocess_media", return_value=fake_preprocess):
                            with patch("cli.resolve_model_path", side_effect=[Path("models/qwen3-asr"), Path("models/qwen3-aligner")]):
                                with redirect_stdout(stdout):
                                    code = cli.main(
                                    [
                                        "--input", "in.wav",
                                        "--script", "tests/fixtures/script_sample.tsv",
                                        "--out", str(out_dir),
                                        "--asr-backend", "qwen3-asr",
                                        "--alignment-model-path", "models/qwen3-aligner",
                                        "--two-stage-qwen3",
                                        "--chunk", "0",
                                        "--match-threshold", "0.0",
                                        "--verbose",
                                    ],
                                    which=lambda _: "/usr/bin/ffmpeg",
                                    runner=lambda *args, **kwargs: subprocess.CompletedProcess(args, 0),
                                )

        self.assertEqual(code, 0)
        logs = stdout.getvalue()
        self.assertIn("stage: two-stage stage-1 qwen3-asr start", logs)
        self.assertIn("stage: two-stage stage-1 qwen3-asr end", logs)
        self.assertIn("stage: two-stage stage-2 qwen3-forced-aligner start", logs)
        self.assertIn("stage: two-stage stage-2 qwen3-forced-aligner end", logs)
        self.assertLess(logs.index("stage: two-stage stage-1 qwen3-asr start"), logs.index("stage: two-stage stage-1 qwen3-asr end"))
        self.assertLess(logs.index("stage: two-stage stage-1 qwen3-asr end"), logs.index("stage: two-stage stage-2 qwen3-forced-aligner start"))
        self.assertLess(logs.index("stage: two-stage stage-2 qwen3-forced-aligner start"), logs.index("stage: two-stage stage-2 qwen3-forced-aligner end"))
        self.assertIn("asr_stage1=", logs)
        self.assertIn("asr_stage2=", logs)

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


    def test_qwen_text_only_transcript_only_output_fails_with_forced_alignment_guidance(self) -> None:
        stderr = io.StringIO()
        backend_registration = SimpleNamespace(
            name="qwen3-asr",
            capabilities=SimpleNamespace(
                supports_segment_timestamps=True,
                supports_alignment=False,
                timestamp_guarantee="text-only",
            ),
        )
        enabled_status = SimpleNamespace(
            name="qwen3-asr",
            enabled=True,
            missing_dependencies=(),
            reason="enabled",
            install_extra="asr_qwen3",
            supports_alignment=False,
        )
        fake_preprocess = SimpleNamespace(canonical_wav_path=Path("out/_tmp/canonical.wav"), chunk_metadata=[])

        with patch("cli.list_backend_status", return_value=[enabled_status]):
            with patch("cli.get_backend", return_value=backend_registration):
                with patch("cli.preprocess_media", return_value=fake_preprocess):
                    with patch("cli.load_script_table", return_value=SimpleNamespace(rows=[], delimiter="\t")):
                        with patch("cli.resolve_model_path", return_value=Path("models/qwen3-asr")):
                            with patch(
                                "cli.dispatch_asr_transcription",
                                side_effect=ValueError(
                                    "qwen3-asr backend did not return timestamped chunks. "
                                    "Expected a non-empty 'chunks' list with per-segment timestamps; "
                                    "qwen3-asr may emit text-only output depending on runtime/model behavior."
                                ),
                            ):
                                with patch("cli.match_segments_to_script") as matching_call:
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
        self.assertIn("timestamp_guarantee='text-only'", stderr.getvalue())
        self.assertIn("Run explicit forced alignment first", stderr.getvalue())
        self.assertIn("qwen3-forced-aligner", stderr.getvalue())
        matching_call.assert_not_called()

    def test_non_text_only_timestamp_errors_are_not_rewritten(self) -> None:
        stderr = io.StringIO()
        backend_registration = SimpleNamespace(
            name="faster-whisper",
            capabilities=SimpleNamespace(
                supports_segment_timestamps=True,
                supports_alignment=False,
                timestamp_guarantee="segment-level",
            ),
        )
        enabled_status = SimpleNamespace(
            name="faster-whisper",
            enabled=True,
            missing_dependencies=(),
            reason="enabled",
            install_extra="asr_faster_whisper",
            supports_alignment=False,
        )
        fake_preprocess = SimpleNamespace(canonical_wav_path=Path("out/_tmp/canonical.wav"), chunk_metadata=[])

        with patch("cli.list_backend_status", return_value=[enabled_status]):
            with patch("cli.get_backend", return_value=backend_registration):
                with patch("cli.preprocess_media", return_value=fake_preprocess):
                    with patch("cli.load_script_table", return_value=SimpleNamespace(rows=[], delimiter="\t")):
                        with patch("cli.resolve_model_path", return_value=Path("models/faster-whisper/tiny")):
                            with patch("cli.dispatch_asr_transcription", side_effect=ValueError("backend timestamp parse failure")):
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
                                            "faster-whisper",
                                        ],
                                        which=lambda _: "/usr/bin/ffmpeg",
                                        runner=lambda *args, **kwargs: subprocess.CompletedProcess(args, 0),
                                    )

        self.assertEqual(code, 1)
        self.assertIn("backend timestamp parse failure", stderr.getvalue())
        self.assertNotIn("Run explicit forced alignment first", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
