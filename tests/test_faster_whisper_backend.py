import sys
import tempfile
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

class _FakeWhisperModelWithInternalProgress:
    def __init__(self, model_path: str, device: str) -> None:
        self.model_path = model_path
        self.device = device

    def transcribe(self, audio_path: str):
        del audio_path
        transcribe_module = sys.modules["faster_whisper.transcribe"]
        pbar = transcribe_module.tqdm(total=1)
        pbar.update(1)
        pbar.close()
        return [_FakeSegment(0.0, 0.5, "line")], object()

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

def _create_fake_model_dir(base_dir: Path) -> Path:
    model_dir = base_dir / "faster-whisper"
    model_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("config.json", "tokenizer.json", "vocabulary.json", "model.bin"):
        (model_dir / filename).write_text("{}", encoding="utf-8")
    return model_dir

class FasterWhisperBackendTests(unittest.TestCase):
    def test_missing_dependency_has_clear_install_hint(self) -> None:
        backend = FasterWhisperBackend()

        with patch("src.asr.faster_whisper_backend.import_module", side_effect=ModuleNotFoundError()):
            with self.assertRaisesRegex(ValueError, "pip install 'cutscene-locator\\[faster-whisper\\]'"):
                backend.transcribe(
                    "in.wav",
                    ASRConfig(backend_name="faster-whisper", model_path=Path("models/faster-whisper")),
                )

    def test_cuda_request_without_cuda_is_actionable(self) -> None:
        backend = FasterWhisperBackend()

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = _create_fake_model_dir(Path(temp_dir))
            with self.assertRaisesRegex(ValueError, "docs/CUDA.md"):
                backend.transcribe(
                    "in.wav",
                    ASRConfig(backend_name="faster-whisper", model_path=model_path, device="cuda"),
                )

    def test_backend_emits_contract_without_speaker(self) -> None:
        backend = FasterWhisperBackend()
        fake_module = types.SimpleNamespace(WhisperModel=_FakeWhisperModel)

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = _create_fake_model_dir(Path(temp_dir))
            with patch("src.asr.faster_whisper_backend.import_module", return_value=fake_module):
                with patch("src.asr.faster_whisper_backend.version", return_value="1.1.0"):
                    result = backend.transcribe(
                        "in.wav",
                        ASRConfig(backend_name="faster-whisper", model_path=model_path, device="cpu"),
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

    def test_progress_off_uses_null_progress_bar_and_restores_tqdm(self) -> None:
        backend = FasterWhisperBackend()

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = _create_fake_model_dir(Path(temp_dir))

            def _none_tqdm(*_args: object, **_kwargs: object):
                return None

            fake_transcribe_module = types.SimpleNamespace(tqdm=_none_tqdm)
            fake_root_module = types.SimpleNamespace(WhisperModel=_FakeWhisperModelWithInternalProgress)

            def _import_module(name: str):
                if name == "faster_whisper":
                    return fake_root_module
                if name == "faster_whisper.transcribe":
                    return fake_transcribe_module
                raise ModuleNotFoundError(name)

            with patch("src.asr.faster_whisper_backend.import_module", side_effect=_import_module):
                with patch("src.asr.faster_whisper_backend.version", return_value="1.1.0"):
                    with patch.dict(sys.modules, {"faster_whisper.transcribe": fake_transcribe_module}, clear=False):
                        result = backend.transcribe(
                            "in.wav",
                            ASRConfig(
                                backend_name="faster-whisper",
                                model_path=model_path,
                                device="cpu",
                                download_progress=False,
                            ),
                        )

        self.assertEqual(result["segments"][0]["text"], "line")
        self.assertIs(fake_transcribe_module.tqdm, _none_tqdm)

if __name__ == "__main__":
    unittest.main()
