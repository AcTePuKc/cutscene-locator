import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from src.asr import ASRResult
from src.asr import asr_worker


class ASRWorkerTests(unittest.TestCase):

    def test_main_sets_cuda_env_before_asr_import(self) -> None:
        fake_result: ASRResult = {
            "segments": [{"segment_id": "seg_0001", "start": 0.0, "end": 1.0, "text": "ok"}],
            "meta": {"backend": "faster-whisper", "model": "tiny", "version": "1.2.1", "device": "cuda"},
        }

        class _FakeBackend:
            def transcribe(self, audio_path: str, config: object) -> ASRResult:
                del audio_path, config
                return fake_result

        captured_env: dict[str, str | None] = {}

        def _import_module(name: str):
            self.assertEqual(name, "src.asr")
            captured_env["TQDM_DISABLE"] = asr_worker.os.environ.get("TQDM_DISABLE")
            captured_env["HF_HUB_DISABLE_PROGRESS_BARS"] = asr_worker.os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
            return types.SimpleNamespace(
                ASRConfig=lambda **kwargs: types.SimpleNamespace(**kwargs),
                FasterWhisperBackend=lambda: _FakeBackend(),
                parse_asr_result=lambda payload, source: payload,
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            out_path = Path(temp_dir) / "result.json"
            with patch.dict("src.asr.asr_worker.os.environ", {}, clear=True):
                with patch("src.asr.asr_worker.import_module", side_effect=_import_module):
                    code = asr_worker.main(
                        [
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
                        ]
                    )

        self.assertEqual(code, 0)
        self.assertEqual(captured_env["TQDM_DISABLE"], "1")
        self.assertEqual(captured_env["HF_HUB_DISABLE_PROGRESS_BARS"], "1")
    def test_configure_runtime_environment_sets_cuda_progress_guards(self) -> None:
        with patch.dict("src.asr.asr_worker.os.environ", {}, clear=True):
            asr_worker._configure_runtime_environment(device="cuda")
            self.assertEqual(asr_worker.os.environ.get("TQDM_DISABLE"), "1")
            self.assertEqual(asr_worker.os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS"), "1")

    def test_configure_runtime_environment_is_noop_for_cpu(self) -> None:
        with patch.dict("src.asr.asr_worker.os.environ", {}, clear=True):
            asr_worker._configure_runtime_environment(device="cpu")
            self.assertNotIn("TQDM_DISABLE", asr_worker.os.environ)
            self.assertNotIn("HF_HUB_DISABLE_PROGRESS_BARS", asr_worker.os.environ)

    def test_main_serializes_validated_asr_result_contract(self) -> None:
        fake_result: ASRResult = {
            "segments": [
                {
                    "segment_id": "seg_0001",
                    "start": 0.0,
                    "end": 1.0,
                    "text": "hello",
                }
            ],
            "meta": {
                "backend": "faster-whisper",
                "model": "tiny",
                "version": "1.2.1",
                "device": "cuda",
            },
        }

        class _FakeBackend:
            def transcribe(self, audio_path: str, config: object) -> ASRResult:
                del audio_path, config
                return fake_result

        fake_module = types.SimpleNamespace(
            ASRConfig=lambda **kwargs: types.SimpleNamespace(**kwargs),
            FasterWhisperBackend=lambda: _FakeBackend(),
            parse_asr_result=lambda payload, source: payload,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            out_path = Path(temp_dir) / "result.json"
            with patch("src.asr.asr_worker.import_module", return_value=fake_module):
                code = asr_worker.main(
                    [
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
                    ]
                )
            payload = json.loads(out_path.read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertIsInstance(payload, dict)
        self.assertEqual(payload["segments"][0]["segment_id"], "seg_0001")


if __name__ == "__main__":
    unittest.main()
