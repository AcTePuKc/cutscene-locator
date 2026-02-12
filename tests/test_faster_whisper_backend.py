import json
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


class _FakeWhisperModel:
    def __init__(self, model_path: str, device: str, compute_type: str) -> None:
        self.model_path = model_path
        self.device = device
        self.compute_type = compute_type
        self.calls: list[dict[str, object]] = []

    def transcribe(self, audio_path: str, **kwargs: object):
        self.calls.append({"audio_path": audio_path, "kwargs": kwargs})
        return [
            _FakeSegment(0.0, 1.2, " Hello there "),
            _FakeSegment(1.3, 2.7, "General Kenobi"),
        ], object()


class _FakeWhisperModelFactory:
    def __init__(self) -> None:
        self.instance: _FakeWhisperModel | None = None

    def __call__(self, model_path: str, device: str, compute_type: str) -> _FakeWhisperModel:
        self.instance = _FakeWhisperModel(model_path, device, compute_type)
        return self.instance


class _FakeWhisperModelNoKwargs:
    def __init__(self, model_path: str, device: str, compute_type: str) -> None:
        self.model_path = model_path
        self.device = device
        self.compute_type = compute_type
        self.calls: list[dict[str, object]] = []

    def transcribe(self, audio_path: str):
        self.calls.append({"audio_path": audio_path})
        return [_FakeSegment(0.0, 1.0, "No kwargs")], object()


class _FakeWhisperModelNoKwargsFactory:
    def __init__(self) -> None:
        self.instance: _FakeWhisperModelNoKwargs | None = None

    def __call__(self, model_path: str, device: str, compute_type: str) -> _FakeWhisperModelNoKwargs:
        self.instance = _FakeWhisperModelNoKwargs(model_path, device, compute_type)
        return self.instance




class _FakeWhisperModelFromFixture:
    def __init__(self, model_path: str, device: str, compute_type: str, fixture_segments: list[dict[str, object]]) -> None:
        del model_path
        del device
        del compute_type
        self.fixture_segments = fixture_segments

    def transcribe(self, audio_path: str, **kwargs: object):
        del audio_path
        del kwargs
        segments = [_FakeSegment(float(item["start"]), float(item["end"]), str(item["text"])) for item in self.fixture_segments]
        return segments, object()


class _FakeWhisperModelFromFixtureFactory:
    def __init__(self, fixture_segments: list[dict[str, object]]) -> None:
        self.fixture_segments = fixture_segments

    def __call__(self, model_path: str, device: str, compute_type: str) -> _FakeWhisperModelFromFixture:
        return _FakeWhisperModelFromFixture(model_path, device, compute_type, self.fixture_segments)

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
        fake_factory = _FakeWhisperModelFactory()
        fake_module = types.SimpleNamespace(WhisperModel=fake_factory)

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = _create_fake_model_dir(Path(temp_dir))
            with patch("src.asr.faster_whisper_backend.import_module", return_value=fake_module):
                with patch("src.asr.faster_whisper_backend.version", return_value="1.1.0"):
                    result = backend.transcribe(
                        "in.wav",
                        ASRConfig(
                            backend_name="faster-whisper",
                            model_path=model_path,
                            device="cpu",
                            compute_type="float16",
                        ),
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
        self.assertIsNotNone(fake_factory.instance)
        assert fake_factory.instance is not None
        self.assertEqual(fake_factory.instance.compute_type, "float16")
        self.assertEqual(
            fake_factory.instance.calls[0]["kwargs"],
            {"vad_filter": False, "temperature": 0.0},
        )

    def test_cuda_transcribe_uses_no_progress_kwargs(self) -> None:
        backend = FasterWhisperBackend()
        fake_factory = _FakeWhisperModelFactory()
        fake_module = types.SimpleNamespace(WhisperModel=fake_factory)
        imported_modules: list[str] = []

        def _import_module(name: str):
            imported_modules.append(name)
            if name == "faster_whisper":
                return fake_module
            raise ModuleNotFoundError(name)

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = _create_fake_model_dir(Path(temp_dir))
            with patch("src.asr.faster_whisper_backend.resolve_device_with_details", return_value=types.SimpleNamespace(resolved="cuda")):
                with patch("src.asr.faster_whisper_backend.import_module", side_effect=_import_module):
                    with patch("src.asr.faster_whisper_backend.version", return_value="1.1.0"):
                        backend.transcribe(
                            "in.wav",
                            ASRConfig(
                                backend_name="faster-whisper",
                                model_path=model_path,
                                device="cuda",
                                download_progress=False,
                            ),
                        )

        self.assertNotIn("faster_whisper.transcribe", imported_modules)
        self.assertIsNotNone(fake_factory.instance)
        assert fake_factory.instance is not None
        self.assertEqual(
            fake_factory.instance.calls[0]["kwargs"],
            {"vad_filter": False, "temperature": 0.0},
        )


    def test_transcribe_kwargs_never_include_progress(self) -> None:
        backend = FasterWhisperBackend()
        fake_factory = _FakeWhisperModelFactory()
        fake_module = types.SimpleNamespace(WhisperModel=fake_factory)

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = _create_fake_model_dir(Path(temp_dir))
            with patch("src.asr.faster_whisper_backend.import_module", return_value=fake_module):
                with patch("src.asr.faster_whisper_backend.version", return_value="1.2.1"):
                    backend.transcribe(
                        "in.wav",
                        ASRConfig(
                            backend_name="faster-whisper",
                            model_path=model_path,
                            language="en",
                        ),
                    )

        self.assertIsNotNone(fake_factory.instance)
        assert fake_factory.instance is not None
        passed_kwargs = fake_factory.instance.calls[0]["kwargs"]
        self.assertNotIn("progress", passed_kwargs)
        self.assertEqual(passed_kwargs.get("language"), "en")
        self.assertEqual(passed_kwargs.get("vad_filter"), False)

    def test_transcribe_kwargs_are_filtered_by_signature(self) -> None:
        backend = FasterWhisperBackend()
        fake_factory = _FakeWhisperModelNoKwargsFactory()
        fake_module = types.SimpleNamespace(WhisperModel=fake_factory)
        logs: list[str] = []

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = _create_fake_model_dir(Path(temp_dir))
            with patch("src.asr.faster_whisper_backend.import_module", return_value=fake_module):
                with patch("src.asr.faster_whisper_backend.version", return_value="1.2.1"):
                    result = backend.transcribe(
                        "in.wav",
                        ASRConfig(
                            backend_name="faster-whisper",
                            model_path=model_path,
                            language="en",
                            log_callback=logs.append,
                        ),
                    )

        self.assertEqual(result["segments"][0]["text"], "No kwargs")
        self.assertTrue(
            any(
                "filtered unsupported transcribe kwargs: language, temperature, vad_filter" in log
                for log in logs
            )
        )

    def test_transcribe_logs_exact_kwargs_when_log_callback_configured(self) -> None:
        backend = FasterWhisperBackend()
        fake_factory = _FakeWhisperModelFactory()
        fake_module = types.SimpleNamespace(WhisperModel=fake_factory)
        logs: list[str] = []

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = _create_fake_model_dir(Path(temp_dir))
            with patch("src.asr.faster_whisper_backend.import_module", return_value=fake_module):
                with patch("src.asr.faster_whisper_backend.version", return_value="1.2.1"):
                    backend.transcribe(
                        "in.wav",
                        ASRConfig(
                            backend_name="faster-whisper",
                            model_path=model_path,
                            language="en",
                            log_callback=logs.append,
                        ),
                    )

        self.assertIn("asr: transcribe kwargs={'vad_filter': False, 'temperature': 0.0, 'language': 'en'}", logs)



    def test_temperature_zero_does_not_pass_best_of_even_if_configured(self) -> None:
        backend = FasterWhisperBackend()
        fake_factory = _FakeWhisperModelFactory()
        fake_module = types.SimpleNamespace(WhisperModel=fake_factory)

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = _create_fake_model_dir(Path(temp_dir))
            with patch("src.asr.faster_whisper_backend.import_module", return_value=fake_module):
                with patch("src.asr.faster_whisper_backend.version", return_value="1.2.1"):
                    backend.transcribe(
                        "in.wav",
                        ASRConfig(
                            backend_name="faster-whisper",
                            model_path=model_path,
                            best_of=5,
                            temperature=0.0,
                        ),
                    )

        self.assertIsNotNone(fake_factory.instance)
        assert fake_factory.instance is not None
        passed_kwargs = fake_factory.instance.calls[0]["kwargs"]
        self.assertNotIn("progress", passed_kwargs)
        self.assertNotIn("best_of", passed_kwargs)


    def test_merge_short_segments_merges_adjacent_short_segments(self) -> None:
        backend = FasterWhisperBackend()
        fake_factory = _FakeWhisperModelFactory()
        fake_module = types.SimpleNamespace(WhisperModel=fake_factory)

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = _create_fake_model_dir(Path(temp_dir))
            with patch("src.asr.faster_whisper_backend.import_module", return_value=fake_module):
                with patch("src.asr.faster_whisper_backend.version", return_value="1.2.1"):
                    result = backend.transcribe(
                        "in.wav",
                        ASRConfig(
                            backend_name="faster-whisper",
                            model_path=model_path,
                            merge_short_segments_seconds=2.0,
                        ),
                    )

        self.assertEqual(len(result["segments"]), 1)
        self.assertEqual(result["segments"][0]["segment_id"], "seg_0001")
        self.assertIn("Hello there", result["segments"][0]["text"])
        self.assertIn("General Kenobi", result["segments"][0]["text"])


    def test_timestamp_normalization_is_stable_for_backend_edge_fixture(self) -> None:
        backend = FasterWhisperBackend()
        fixture = json.loads(Path("tests/fixtures/asr_timestamp_edges_faster_whisper.json").read_text(encoding="utf-8"))
        fake_factory = _FakeWhisperModelFromFixtureFactory(fixture["raw_segments"])
        fake_module = types.SimpleNamespace(WhisperModel=fake_factory)

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = _create_fake_model_dir(Path(temp_dir))
            with patch("src.asr.faster_whisper_backend.import_module", return_value=fake_module):
                with patch("src.asr.faster_whisper_backend.version", return_value="1.2.1"):
                    result = backend.transcribe(
                        "in.wav",
                        ASRConfig(
                            backend_name="faster-whisper",
                            model_path=model_path,
                        ),
                    )

        self.assertEqual(result["segments"], fixture["expected"])


if __name__ == "__main__":
    unittest.main()
