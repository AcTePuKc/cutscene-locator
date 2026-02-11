import io
import subprocess
import unittest
from contextlib import redirect_stderr, redirect_stdout

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
        self.assertIn("Invalid ASR backend", stderr.getvalue())

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
                ],
                which=lambda _: "/usr/bin/ffmpeg",
                runner=lambda *args, **kwargs: subprocess.CompletedProcess(args, 0),
            )

        self.assertEqual(code, 1)
        self.assertIn("start must be less than end", stderr.getvalue())

    def test_success_exits_zero(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
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
                    "--verbose",
                ],
                which=lambda _: "/usr/bin/ffmpeg",
                runner=lambda *args, **kwargs: subprocess.CompletedProcess(args, 0),
            )

        self.assertEqual(code, 0)
        output = stdout.getvalue()
        self.assertIn("ASR validation completed", output)
        self.assertIn("Verbose: script rows loaded=2", output)
        self.assertIn("Verbose: asr backend=mock segments=2", output)


if __name__ == "__main__":
    unittest.main()
