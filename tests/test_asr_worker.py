import json
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from src.asr import asr_worker


class ASRWorkerTests(unittest.TestCase):
    def test_configure_runtime_environment_sets_progress_guards(self) -> None:
        with patch.dict("src.asr.asr_worker.os.environ", {}, clear=True):
            asr_worker._configure_runtime_environment(device="cuda")
            self.assertEqual(asr_worker.os.environ.get("TQDM_DISABLE"), "1")
            self.assertEqual(asr_worker.os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS"), "1")

    def test_main_uses_direct_module_imports_and_env_is_set(self) -> None:
        captured: dict[str, object] = {}

        class _FakeBackend:
            def transcribe(self, audio_path: str, config: object):
                captured["audio_path"] = audio_path
                captured["config"] = config
                captured["env"] = {
                    "TQDM_DISABLE": asr_worker.os.environ.get("TQDM_DISABLE"),
                    "HF_HUB_DISABLE_PROGRESS_BARS": asr_worker.os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS"),
                }
                return {
                    "segments": [{"segment_id": "seg_0001", "start": 0.0, "end": 1.0, "text": "ok"}],
                    "meta": {"backend": "faster-whisper", "model": "tiny", "version": "1.2.1", "device": "cuda"},
                }

        with tempfile.TemporaryDirectory() as temp_dir:
            out_path = Path(temp_dir) / "result.json"
            with patch.dict("src.asr.asr_worker.os.environ", {}, clear=True):
                with patch("src.asr.asr_worker.FasterWhisperBackend", return_value=_FakeBackend()):
                    with patch("src.asr.asr_worker.parse_asr_result", side_effect=lambda payload, source: payload):
                        code = asr_worker.main(
                            [
                                "--asr-backend",
                                "faster-whisper",
                                "--audio-path",
                                "in.wav",
                                "--model-path",
                                "models/faster-whisper",
                                "--device",
                                "cuda",
                                "--compute-type",
                                "float16",
                                "--result-path",
                                str(out_path),
                                "--asr-beam-size",
                                "1",
                                "--asr-temperature",
                                "0.0",
                                "--asr-best-of",
                                "1",
                            ]
                        )

        self.assertEqual(code, 0)
        self.assertEqual(captured["audio_path"], "in.wav")
        self.assertEqual(captured["env"], {"TQDM_DISABLE": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1"})
        config = captured["config"]
        self.assertEqual(config.backend_name, "faster-whisper")
        self.assertEqual(config.compute_type, "float16")
        self.assertEqual(config.device, "cuda")

    def test_main_serializes_validated_asr_result_contract(self) -> None:
        class _FakeBackend:
            def transcribe(self, audio_path: str, config: object):
                del audio_path, config
                return {
                    "segments": [{"segment_id": "seg_0001", "start": 0.0, "end": 1.0, "text": "hello"}],
                    "meta": {"backend": "faster-whisper", "model": "tiny", "version": "1.2.1", "device": "cuda"},
                }

        with tempfile.TemporaryDirectory() as temp_dir:
            out_path = Path(temp_dir) / "result.json"
            with patch("src.asr.asr_worker.FasterWhisperBackend", return_value=_FakeBackend()):
                with patch("src.asr.asr_worker.parse_asr_result", side_effect=lambda payload, source: payload):
                    code = asr_worker.main(
                        [
                            "--asr-backend",
                            "faster-whisper",
                            "--audio-path",
                            "in.wav",
                            "--model-path",
                            "models/faster-whisper",
                            "--device",
                            "cuda",
                            "--compute-type",
                            "float16",
                            "--result-path",
                            str(out_path),
                            "--asr-beam-size",
                            "1",
                            "--asr-temperature",
                            "0.0",
                            "--asr-best-of",
                            "1",
                        ]
                    )
            payload = json.loads(out_path.read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(payload["segments"][0]["segment_id"], "seg_0001")

    def test_main_verbose_cpu_runs_preflight_with_vad_disabled(self) -> None:
        captured: dict[str, object] = {}

        class _FakeBackend:
            def transcribe(self, audio_path: str, config: object):
                captured["backend_audio_path"] = audio_path
                return {
                    "segments": [{"segment_id": "seg_0001", "start": 0.0, "end": 1.0, "text": "ok"}],
                    "meta": {"backend": "faster-whisper", "model": "tiny", "version": "1.2.1", "device": "cpu"},
                }

        with tempfile.TemporaryDirectory() as temp_dir:
            out_path = Path(temp_dir) / "result.json"
            with patch.dict("src.asr.asr_worker.os.environ", {"PATH": "/bin"}, clear=True):
                with patch("src.asr.asr_worker._print_verbose_environment_dump"):
                    with patch("src.asr.asr_worker._run_minimal_whisper_preflight") as preflight:
                        with patch("src.asr.asr_worker.FasterWhisperBackend", return_value=_FakeBackend()):
                            with patch("src.asr.asr_worker.parse_asr_result", side_effect=lambda payload, source: payload):
                                with patch("sys.stdout", new_callable=StringIO) as stdout:
                                    code = asr_worker.main(
                                        [
                                            "--asr-backend",
                                            "faster-whisper",
                                            "--audio-path",
                                            "in.wav",
                                            "--model-path",
                                            "models/faster-whisper",
                                            "--device",
                                            "cpu",
                                            "--compute-type",
                                            "float32",
                                            "--result-path",
                                            str(out_path),
                                            "--asr-beam-size",
                                            "1",
                                            "--asr-temperature",
                                            "0.0",
                                            "--asr-best-of",
                                            "1",
                                            "--verbose",
                                        ]
                                    )

        self.assertEqual(code, 0)
        preflight.assert_called_once_with(
            audio_path="in.wav",
            model_path=Path("models/faster-whisper"),
            device="cpu",
            compute_type="float32",
        )
        self.assertEqual(captured["backend_audio_path"], "in.wav")
        output = stdout.getvalue()
        self.assertIn("asr-worker: backend.transcribe begin", output)

    def test_main_verbose_non_faster_whisper_skips_whisper_preflight(self) -> None:
        captured: dict[str, object] = {}

        class _FakeBackend:
            def transcribe(self, audio_path: str, config: object):
                captured["audio_path"] = audio_path
                captured["config"] = config
                return {
                    "segments": [{"segment_id": "seg_0001", "start": 0.0, "end": 1.0, "text": "ok"}],
                    "meta": {"backend": "qwen3-asr", "model": "tiny", "version": "1.0", "device": "cpu"},
                }

        with tempfile.TemporaryDirectory() as temp_dir:
            out_path = Path(temp_dir) / "result.json"
            with patch("src.asr.asr_worker._build_runtime_backend", return_value=_FakeBackend()):
                with patch("src.asr.asr_worker.parse_asr_result", side_effect=lambda payload, source: payload):
                    with patch("src.asr.asr_worker._run_minimal_whisper_preflight") as preflight:
                        code = asr_worker.main(
                            [
                                "--asr-backend",
                                "qwen3-asr",
                                "--audio-path",
                                "in.wav",
                                "--model-path",
                                "models/qwen3-asr",
                                "--device",
                                "cpu",
                                "--compute-type",
                                "float32",
                                "--result-path",
                                str(out_path),
                                "--verbose",
                            ]
                        )

        self.assertEqual(code, 0)
        preflight.assert_not_called()
        self.assertEqual(captured["audio_path"], "in.wav")
        self.assertEqual(captured["config"].backend_name, "qwen3-asr")

    def test_main_unsupported_backend_has_deterministic_error_message(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Unsupported --asr-backend 'invalid-backend' for ASR worker. "
            "Expected one of: faster-whisper, qwen3-asr, whisperx, vibevoice.",
        ):
            asr_worker.main(
                [
                    "--asr-backend",
                    "invalid-backend",
                    "--audio-path",
                    "in.wav",
                    "--model-path",
                    "models/invalid-backend",
                    "--device",
                    "cpu",
                    "--compute-type",
                    "float32",
                    "--result-path",
                    "out.json",
                ]
            )


if __name__ == "__main__":
    unittest.main()
