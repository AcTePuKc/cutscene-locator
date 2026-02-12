import types
import unittest
from pathlib import Path
from unittest.mock import patch

from src.asr import ASRConfig
from src.asr.vibevoice_backend import VibeVoiceBackend


class VibeVoiceBackendTests(unittest.TestCase):
    def test_missing_dependency_has_clear_install_hint(self) -> None:
        backend = VibeVoiceBackend()

        with patch("src.asr.vibevoice_backend.import_module", side_effect=ModuleNotFoundError()):
            with self.assertRaisesRegex(ValueError, "cutscene-locator\\[asr_vibevoice\\]"):
                backend.transcribe(
                    "in.wav",
                    ASRConfig(backend_name="vibevoice", model_path=Path("models/vibevoice"), device="cpu"),
                )

    def test_backend_emits_standard_contract_with_normalized_segments(self) -> None:
        backend = VibeVoiceBackend()

        fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))

        def _fake_transcribe_file(**kwargs):
            return {
                "segments": [
                    {"start": 1.2000000004, "end": 2.2000000005, "text": "hello"},
                    {"start": 0.0, "end": 1.0, "text": "world"},
                ]
            }

        fake_vibevoice = types.SimpleNamespace(transcribe_file=_fake_transcribe_file)

        def _fake_import(name: str):
            if name == "torch":
                return fake_torch
            if name == "vibevoice":
                return fake_vibevoice
            raise ModuleNotFoundError(name)

        with patch("src.asr.vibevoice_backend.import_module", side_effect=_fake_import):
            with patch("src.asr.vibevoice_backend.version", return_value="0.5.0"):
                result = backend.transcribe(
                    "in.wav",
                    ASRConfig(backend_name="vibevoice", model_path=Path("models/vibevoice"), device="cpu"),
                )

        self.assertEqual(result["meta"]["backend"], "vibevoice")
        self.assertEqual(result["meta"]["model"], "vibevoice")
        self.assertEqual(result["meta"]["version"], "0.5.0")
        self.assertEqual(result["meta"]["device"], "cpu")
        self.assertEqual(result["segments"][0]["start"], 0.0)
        self.assertEqual(result["segments"][0]["end"], 1.0)
        self.assertEqual(result["segments"][1]["start"], 1.2)
        self.assertEqual(result["segments"][1]["end"], 2.2)


if __name__ == "__main__":
    unittest.main()
