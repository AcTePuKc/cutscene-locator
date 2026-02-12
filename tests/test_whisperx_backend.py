import json
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from src.asr import ASRConfig
from src.asr.whisperx_backend import WhisperXBackend


class _FakeWhisperXModel:
    def transcribe(self, audio: object, batch_size: int) -> dict[str, object]:
        del audio, batch_size
        return {
            "segments": [
                {"start": 0.0, "end": 1.0, "text": " hello ", "speaker": "A"},
                {"start": 1.1, "end": 2.0, "text": "world"},
            ]
        }


class WhisperXBackendTests(unittest.TestCase):
    def test_missing_dependency_has_clear_install_hint(self) -> None:
        backend = WhisperXBackend()

        with patch("src.asr.whisperx_backend.import_module", side_effect=ModuleNotFoundError()):
            with self.assertRaisesRegex(ValueError, "cutscene-locator\\[asr_whisperx\\]"):
                backend.transcribe(
                    "in.wav",
                    ASRConfig(backend_name="whisperx", model_path=Path("models/whisperx"), device="cpu"),
                )

    def test_backend_emits_standard_contract(self) -> None:
        backend = WhisperXBackend()

        fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
        fake_whisperx = types.SimpleNamespace(
            load_model=lambda *args, **kwargs: _FakeWhisperXModel(),
            load_audio=lambda audio_path: object(),
        )

        def _fake_import(name: str):
            if name == "torch":
                return fake_torch
            if name == "whisperx":
                return fake_whisperx
            raise ModuleNotFoundError(name)

        with patch("src.asr.whisperx_backend.import_module", side_effect=_fake_import):
            with patch("src.asr.whisperx_backend.version", return_value="3.1.1"):
                result = backend.transcribe(
                    "in.wav",
                    ASRConfig(backend_name="whisperx", model_path=Path("models/whisperx"), device="cpu"),
                )

        self.assertEqual(result["meta"]["backend"], "whisperx")
        self.assertEqual(result["meta"]["model"], "whisperx")
        self.assertEqual(result["meta"]["version"], "3.1.1")
        self.assertEqual(result["meta"]["device"], "cpu")
        self.assertEqual(result["segments"][0]["segment_id"], "seg_0001")
        self.assertEqual(result["segments"][0]["start"], 0.0)
        self.assertEqual(result["segments"][0]["end"], 1.0)
        self.assertEqual(result["segments"][0]["text"], " hello ")
        self.assertEqual(result["segments"][0]["speaker"], "A")

    def test_timestamp_normalization_is_stable_for_backend_edge_fixture(self) -> None:
        backend = WhisperXBackend()
        fixture = json.loads(Path("tests/fixtures/asr_timestamp_edges_faster_whisper.json").read_text(encoding="utf-8"))
        raw_segments = []
        for segment in fixture["raw_segments"]:
            raw_segments.append({"start": segment["start"], "end": segment["end"], "text": segment["text"]})

        fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
        fake_whisperx = types.SimpleNamespace(
            load_model=lambda *args, **kwargs: types.SimpleNamespace(
                transcribe=lambda audio, batch_size: {"segments": raw_segments}
            ),
            load_audio=lambda audio_path: object(),
        )

        def _fake_import(name: str):
            if name == "torch":
                return fake_torch
            if name == "whisperx":
                return fake_whisperx
            raise ModuleNotFoundError(name)

        with patch("src.asr.whisperx_backend.import_module", side_effect=_fake_import):
            result = backend.transcribe(
                "in.wav",
                ASRConfig(backend_name="whisperx", model_path=Path("models/whisperx"), device="cpu"),
            )

        self.assertEqual(result["segments"], fixture["expected"])


if __name__ == "__main__":
    unittest.main()
