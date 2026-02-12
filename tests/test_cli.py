import io
import json
import sys
import subprocess
import tempfile
import unittest
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
