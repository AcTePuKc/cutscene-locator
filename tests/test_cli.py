import io
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
        self.assertIn("To allow download, pass --auto-download tiny", stderr.getvalue())

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
            self.assertIn("Verbose: script rows loaded=2", output)

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


if __name__ == "__main__":
    unittest.main()
