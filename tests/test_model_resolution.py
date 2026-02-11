import json
import tempfile
import types
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

    def test_faster_whisper_auto_download_size_maps_to_hf_repo_and_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp_home:
            snapshot_calls: list[tuple[str, str, str | None]] = []

            def _snapshot_download(*, repo_id: str, local_dir: str, revision: str | None) -> None:
                snapshot_calls.append((repo_id, local_dir, revision))
                model_dir = Path(local_dir)
                model_dir.mkdir(parents=True, exist_ok=True)
                for filename in ("config.json", "tokenizer.json", "vocabulary.json", "model.bin"):
                    (model_dir / filename).write_text("{}", encoding="utf-8")

            fake_hf_module = types.SimpleNamespace(snapshot_download=_snapshot_download)

            with patch("pathlib.Path.home", return_value=Path(temp_home)):
                with patch("src.asr.model_resolution.import_module", return_value=fake_hf_module):
                    resolved = resolve_model_path(
                        ASRConfig(backend_name="faster-whisper", auto_download="small")
                    )

            expected_dir = Path(temp_home) / ".cutscene-locator" / "models" / "faster-whisper" / "small"
            self.assertEqual(resolved, expected_dir)
            self.assertEqual(
                snapshot_calls,
                [("Systran/faster-whisper-small", str(expected_dir), None)],
            )

    def test_faster_whisper_auto_download_reports_failure_with_repo_hint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_home:
            fake_hf_module = types.SimpleNamespace(
                snapshot_download=lambda *, repo_id, local_dir, revision: (_ for _ in ()).throw(RuntimeError("boom"))
            )
            with patch("pathlib.Path.home", return_value=Path(temp_home)):
                with patch("src.asr.model_resolution.import_module", return_value=fake_hf_module):
                    with self.assertRaisesRegex(ModelResolutionError, "Systran/faster-whisper-base"):
                        resolve_model_path(ASRConfig(backend_name="faster-whisper", auto_download="base"))

    def test_model_id_download_uses_deterministic_cache_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_home:
            snapshot_calls: list[tuple[str, str | None, str]] = []

            def _snapshot_download(*, repo_id: str, revision: str | None, local_dir: str) -> None:
                snapshot_calls.append((repo_id, revision, local_dir))
                model_dir = Path(local_dir)
                model_dir.mkdir(parents=True, exist_ok=True)
                for filename in ("config.json", "tokenizer.json", "vocabulary.json", "model.bin"):
                    (model_dir / filename).write_text("{}", encoding="utf-8")

            fake_hf_module = types.SimpleNamespace(snapshot_download=_snapshot_download)
            with patch("pathlib.Path.home", return_value=Path(temp_home)):
                with patch("src.asr.model_resolution.import_module", return_value=fake_hf_module):
                    resolved = resolve_model_path(
                        ASRConfig(
                            backend_name="faster-whisper",
                            model_id="openai/whisper-tiny",
                            revision="main",
                        )
                    )

            expected_dir = (
                Path(temp_home)
                / ".cutscene-locator"
                / "models"
                / "faster-whisper"
                / "openai--whisper-tiny"
                / "main"
            )
            self.assertEqual(resolved, expected_dir)
            self.assertEqual(
                snapshot_calls,
                [("openai/whisper-tiny", "main", str(expected_dir))],
            )

    def test_model_id_download_default_revision_folder(self) -> None:
        with tempfile.TemporaryDirectory() as temp_home:
            fake_hf_module = types.SimpleNamespace(
                snapshot_download=lambda *, repo_id, revision, local_dir: [
                    Path(local_dir).mkdir(parents=True, exist_ok=True),
                    (Path(local_dir) / "config.json").write_text("{}", encoding="utf-8"),
                    (Path(local_dir) / "tokenizer.json").write_text("{}", encoding="utf-8"),
                    (Path(local_dir) / "vocabulary.json").write_text("{}", encoding="utf-8"),
                    (Path(local_dir) / "model.bin").write_text("{}", encoding="utf-8"),
                ]
            )
            with patch("pathlib.Path.home", return_value=Path(temp_home)):
                with patch("src.asr.model_resolution.import_module", return_value=fake_hf_module):
                    resolved = resolve_model_path(
                        ASRConfig(backend_name="faster-whisper", model_id="openai/whisper-tiny")
                    )
            self.assertTrue(str(resolved).endswith("/openai--whisper-tiny/default"))


if __name__ == "__main__":
    unittest.main()
