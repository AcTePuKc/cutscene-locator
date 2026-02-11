import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.asr import ASRConfig
from src.asr.model_resolution import ModelResolutionError, resolve_model_cache_dir, resolve_model_path


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self._position = 0
        self.headers = {"Content-Length": str(len(payload))}

    def read(self, amount: int) -> bytes:
        if self._position >= len(self._payload):
            return b""
        chunk = self._payload[self._position : self._position + amount]
        self._position += len(chunk)
        return chunk

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class ModelResolutionTests(unittest.TestCase):
    def test_resolve_model_cache_dir_uses_home(self) -> None:
        with tempfile.TemporaryDirectory() as temp_home:
            with patch("pathlib.Path.home", return_value=Path(temp_home)):
                cache_dir = resolve_model_cache_dir()
                self.assertTrue(cache_dir.exists())
                self.assertEqual(cache_dir, Path(temp_home) / ".cutscene-locator" / "models")

    def test_resolve_model_path_uses_explicit_path_first(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_file = Path(temp_dir) / "my-model.bin"
            model_file.write_bytes(b"ok")
            config = ASRConfig(backend_name="mock", model_path=model_file)

            resolved = resolve_model_path(config)

        self.assertEqual(resolved, model_file)

    def test_resolve_model_path_errors_without_model_or_autodownload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_home:
            with patch("pathlib.Path.home", return_value=Path(temp_home)):
                config = ASRConfig(backend_name="mock")
                with self.assertRaisesRegex(ModelResolutionError, "Model could not be resolved"):
                    resolve_model_path(config)

    def test_auto_download_tiny_writes_metadata_and_reports_progress(self) -> None:
        with tempfile.TemporaryDirectory() as temp_home:
            progress_events: list[float] = []
            with patch("pathlib.Path.home", return_value=Path(temp_home)):
                with patch.dict(
                    "os.environ",
                    {"CUTSCENE_LOCATOR_MODEL_DOWNLOAD_URL": "https://example.invalid/model.bin"},
                    clear=False,
                ):
                    with patch("src.asr.model_resolution.urlopen", return_value=_FakeResponse(b"abc123")):
                        config = ASRConfig(
                            backend_name="mock",
                            auto_download="tiny",
                            progress_callback=progress_events.append,
                        )
                        resolved = resolve_model_path(config)

                        model_file = resolved / "model.bin"
                        metadata_file = resolved / "model_metadata.json"
                        self.assertTrue(model_file.exists())
                        self.assertEqual(model_file.read_bytes(), b"abc123")
                        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
                        self.assertEqual(metadata["backend"], "mock")
                        self.assertEqual(metadata["model_size"], "tiny")
                        self.assertIsNone(metadata["version"])
                        self.assertGreaterEqual(len(progress_events), 1)
                        self.assertEqual(progress_events[-1], 100.0)

    def test_auto_download_can_cancel_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_home:
            with patch("pathlib.Path.home", return_value=Path(temp_home)):
                with patch.dict(
                    "os.environ",
                    {"CUTSCENE_LOCATOR_MODEL_DOWNLOAD_URL": "https://example.invalid/model.bin"},
                    clear=False,
                ):
                    config = ASRConfig(
                        backend_name="mock",
                        auto_download="tiny",
                        cancel_check=lambda: True,
                    )
                    with self.assertRaisesRegex(ModelResolutionError, "cancelled"):
                        resolve_model_path(config)


if __name__ == "__main__":
    unittest.main()
