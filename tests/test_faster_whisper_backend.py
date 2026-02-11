import types
import unittest
from pathlib import Path
from unittest.mock import patch

from src.asr import ASRConfig
from src.asr.faster_whisper_backend import FasterWhisperBackend


class _FakeSegment:
    def __init__(self, start: float, end: float, text: str) -> None:
        self.start = start
        self.end = end
        self.text = text


class _FakeWhisperModel:
    def __init__(self, model_path: str, device: str) -> None:
        self.model_path = model_path
        self.device = device

    def transcribe(self, audio_path: str):
        del audio_path
        return [
            _FakeSegment(0.0, 1.2, " Hello there "),
            _FakeSegment(1.3, 2.7, "General Kenobi"),
        ], object()


class FasterWhisperBackendTests(unittest.TestCase):
    def test_missing_dependency_has_clear_install_hint(self) -> None:
        backend = FasterWhisperBackend()

        with patch("src.asr.faster_whisper_backend.import_module", side_effect=ModuleNotFoundError()):
            with self.assertRaisesRegex(ValueError, "pip install 'cutscene-locator\\[faster-whisper\\]'"):
                backend.transcribe(
                    "in.wav",
                    ASRConfig(backend_name="faster-whisper", model_path=Path("models/faster-whisper")),
                )

    def test_backend_emits_contract_without_speaker(self) -> None:
        backend = FasterWhisperBackend()
        fake_module = types.SimpleNamespace(WhisperModel=_FakeWhisperModel)

        with patch("src.asr.faster_whisper_backend.import_module", return_value=fake_module):
            with patch("src.asr.faster_whisper_backend.version", return_value="1.1.0"):
                result = backend.transcribe(
                    "in.wav",
                    ASRConfig(backend_name="faster-whisper", model_path=Path("models/faster-whisper"), device="cpu"),
                )

        self.assertEqual(result["meta"]["backend"], "faster-whisper")
        self.assertEqual(result["meta"]["model"], "faster-whisper")
        self.assertEqual(result["meta"]["version"], "1.1.0")
        self.assertEqual(result["meta"]["device"], "cpu")
        self.assertEqual(result["segments"][0]["segment_id"], "seg_0001")
        self.assertEqual(result["segments"][0]["start"], 0.0)
        self.assertEqual(result["segments"][0]["end"], 1.2)
        self.assertEqual(result["segments"][0]["text"], "Hello there")
        self.assertNotIn("speaker", result["segments"][0])


if __name__ == "__main__":
    unittest.main()
