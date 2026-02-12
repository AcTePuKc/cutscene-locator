import json
import tempfile
import types
import unittest
from io import StringIO
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
            if name == "tqdm":
                return types.SimpleNamespace(
                    tqdm=types.SimpleNamespace(monitor_interval=10),
                    std=types.SimpleNamespace(tqdm=types.SimpleNamespace(monitor_interval=10)),
                )
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
    def test_configure_runtime_environment_sets_progress_guards_for_cuda(self) -> None:
        with patch.dict("src.asr.asr_worker.os.environ", {}, clear=True):
            asr_worker._configure_runtime_environment(device="cuda")
            self.assertEqual(asr_worker.os.environ.get("TQDM_DISABLE"), "1")
            self.assertEqual(asr_worker.os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS"), "1")

    def test_configure_runtime_environment_sets_progress_guards_for_cpu(self) -> None:
        with patch.dict("src.asr.asr_worker.os.environ", {}, clear=True):
            asr_worker._configure_runtime_environment(device="cpu")
            self.assertEqual(asr_worker.os.environ.get("TQDM_DISABLE"), "1")
            self.assertEqual(asr_worker.os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS"), "1")

    def test_configure_runtime_environment_disables_tqdm_monitor_for_cuda(self) -> None:
        fake_tqdm = types.SimpleNamespace(monitor_interval=10)
        fake_tqdm_std = types.SimpleNamespace(tqdm=types.SimpleNamespace(monitor_interval=10))
        fake_module = types.SimpleNamespace(tqdm=fake_tqdm, std=fake_tqdm_std)

        with patch("src.asr.asr_worker.import_module", return_value=fake_module):
            asr_worker._configure_runtime_environment(device="cuda")

        self.assertEqual(fake_tqdm.monitor_interval, 0)
        self.assertEqual(fake_tqdm_std.tqdm.monitor_interval, 0)

    def test_configure_runtime_environment_tqdm_disable_failure_logs_in_verbose_mode(self) -> None:
        with patch("src.asr.asr_worker.import_module", side_effect=RuntimeError("boom")):
            with patch("sys.stdout", new_callable=StringIO) as stdout:
                asr_worker._configure_runtime_environment(device="cuda", verbose=True)

        self.assertIn("failed to disable tqdm monitor thread for cuda: boom", stdout.getvalue())

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

    def test_main_verbose_runs_environment_dump_and_minimal_preflight(self) -> None:
        fake_result: ASRResult = {
            "segments": [{"segment_id": "seg_0001", "start": 0.0, "end": 1.0, "text": "ok"}],
            "meta": {"backend": "faster-whisper", "model": "tiny", "version": "1.2.1", "device": "cuda"},
        }
        captured: dict[str, object] = {}

        class _FakeRawSegment:
            pass

        class _FakeWhisperModel:
            def __init__(self, model_path: str, device: str, compute_type: str) -> None:
                captured["model_init"] = (model_path, device, compute_type)

            def transcribe(self, audio_path: str, vad_filter: bool):
                captured["preflight_audio_path"] = audio_path
                captured["preflight_vad_filter"] = vad_filter
                return iter([_FakeRawSegment(), _FakeRawSegment()]), object()

        class _FakeBackend:
            def transcribe(self, audio_path: str, config: object) -> ASRResult:
                captured["backend_audio_path"] = audio_path
                captured["backend_config"] = config
                return fake_result

        def _import_module(name: str):
            if name == "tqdm":
                return types.SimpleNamespace(
                    tqdm=types.SimpleNamespace(monitor_interval=10),
                    std=types.SimpleNamespace(tqdm=types.SimpleNamespace(monitor_interval=10)),
                )
            if name == "src.asr":
                return types.SimpleNamespace(
                    ASRConfig=lambda **kwargs: types.SimpleNamespace(**kwargs),
                    FasterWhisperBackend=lambda: _FakeBackend(),
                    parse_asr_result=lambda payload, source: payload,
                )
            if name == "ctranslate2":
                return types.SimpleNamespace(__file__="/tmp/ctranslate2/__init__.py")
            if name == "faster_whisper":
                return types.SimpleNamespace(WhisperModel=_FakeWhisperModel)
            raise AssertionError(f"unexpected import: {name}")

        with tempfile.TemporaryDirectory() as temp_dir:
            out_path = Path(temp_dir) / "result.json"
            with patch.dict("src.asr.asr_worker.os.environ", {"PATH": "/bin"}, clear=True):
                with patch("src.asr.asr_worker.import_module", side_effect=_import_module):
                    with patch("sys.stdout", new_callable=StringIO) as stdout:
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
                                "--verbose",
                            ]
                        )

        self.assertEqual(code, 0)
        self.assertNotIn("model_init", captured)
        self.assertNotIn("preflight_audio_path", captured)
        self.assertEqual(captured["backend_audio_path"], "in.wav")
        output = stdout.getvalue()
        self.assertIn("asr-worker: sys.executable=", output)
        self.assertIn("asr-worker: sys.path[0:3]=", output)
        self.assertIn("asr-worker: ctranslate2.__file__=/tmp/ctranslate2/__init__.py", output)
        self.assertIn("asr-worker: env={'PATH': '/bin', 'CUDA_PATH': None, 'CUDNN_PATH': None}", output)
        self.assertIn("asr-worker: minimal preflight skipped on cuda", output)
        self.assertIn("asr-worker: backend.transcribe begin", output)
        self.assertIn("asr-worker: backend.transcribe end", output)
        self.assertIn("asr-worker: parse_asr_result end", output)
        self.assertIn("asr-worker: write_result_json end", output)

    def test_main_verbose_cpu_runs_preflight_with_vad_disabled(self) -> None:
        fake_result: ASRResult = {
            "segments": [{"segment_id": "seg_0001", "start": 0.0, "end": 1.0, "text": "ok"}],
            "meta": {"backend": "faster-whisper", "model": "tiny", "version": "1.2.1", "device": "cpu"},
        }
        captured: dict[str, object] = {}

        class _FakeRawSegment:
            pass

        class _FakeWhisperModel:
            def __init__(self, model_path: str, device: str, compute_type: str) -> None:
                captured["model_init"] = (model_path, device, compute_type)

            def transcribe(self, audio_path: str, vad_filter: bool):
                captured["preflight_audio_path"] = audio_path
                captured["preflight_vad_filter"] = vad_filter
                return iter([_FakeRawSegment()]), object()

        class _FakeBackend:
            def transcribe(self, audio_path: str, config: object) -> ASRResult:
                captured["backend_audio_path"] = audio_path
                return fake_result

        def _import_module(name: str):
            if name == "tqdm":
                return types.SimpleNamespace(
                    tqdm=types.SimpleNamespace(monitor_interval=10),
                    std=types.SimpleNamespace(tqdm=types.SimpleNamespace(monitor_interval=10)),
                )
            if name == "src.asr":
                return types.SimpleNamespace(
                    ASRConfig=lambda **kwargs: types.SimpleNamespace(**kwargs),
                    FasterWhisperBackend=lambda: _FakeBackend(),
                    parse_asr_result=lambda payload, source: payload,
                )
            if name == "ctranslate2":
                return types.SimpleNamespace(__file__="/tmp/ctranslate2/__init__.py")
            if name == "faster_whisper":
                return types.SimpleNamespace(WhisperModel=_FakeWhisperModel)
            raise AssertionError(f"unexpected import: {name}")

        with tempfile.TemporaryDirectory() as temp_dir:
            out_path = Path(temp_dir) / "result.json"
            with patch.dict("src.asr.asr_worker.os.environ", {"PATH": "/bin"}, clear=True):
                with patch("src.asr.asr_worker.import_module", side_effect=_import_module):
                    with patch("sys.stdout", new_callable=StringIO) as stdout:
                        code = asr_worker.main(
                            [
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
                                "--verbose",
                            ]
                        )

        self.assertEqual(code, 0)
        self.assertEqual(captured["model_init"], ("models/faster-whisper", "cpu", "float32"))
        self.assertEqual(captured["preflight_audio_path"], "in.wav")
        self.assertEqual(captured["preflight_vad_filter"], False)
        self.assertIn("asr-worker: minimal preflight transcribe start", stdout.getvalue())
        self.assertIn("asr-worker: minimal preflight first segment observed", stdout.getvalue())
        output = stdout.getvalue()
        self.assertIn("asr-worker: minimal preflight transcribe end", output)
        self.assertIn("asr-worker: backend.transcribe begin", output)
        self.assertIn("asr-worker: backend.transcribe end", output)
        self.assertIn("asr-worker: parse_asr_result end", output)
        self.assertIn("asr-worker: write_result_json end", output)

    def test_main_verbose_preflight_failure_is_non_fatal(self) -> None:
        fake_result: ASRResult = {
            "segments": [{"segment_id": "seg_0001", "start": 0.0, "end": 1.0, "text": "ok"}],
            "meta": {"backend": "faster-whisper", "model": "tiny", "version": "1.2.1", "device": "cpu"},
        }

        class _FakeWhisperModel:
            def __init__(self, model_path: str, device: str, compute_type: str) -> None:
                del model_path, device, compute_type

            def transcribe(self, audio_path: str, vad_filter: bool):
                del audio_path, vad_filter
                raise RuntimeError("preflight boom")

        class _FakeBackend:
            def transcribe(self, audio_path: str, config: object) -> ASRResult:
                del audio_path, config
                return fake_result

        def _import_module(name: str):
            if name == "tqdm":
                return types.SimpleNamespace(
                    tqdm=types.SimpleNamespace(monitor_interval=10),
                    std=types.SimpleNamespace(tqdm=types.SimpleNamespace(monitor_interval=10)),
                )
            if name == "src.asr":
                return types.SimpleNamespace(
                    ASRConfig=lambda **kwargs: types.SimpleNamespace(**kwargs),
                    FasterWhisperBackend=lambda: _FakeBackend(),
                    parse_asr_result=lambda payload, source: payload,
                )
            if name == "ctranslate2":
                return types.SimpleNamespace(__file__="/tmp/ctranslate2/__init__.py")
            if name == "faster_whisper":
                return types.SimpleNamespace(WhisperModel=_FakeWhisperModel)
            raise AssertionError(f"unexpected import: {name}")

        with tempfile.TemporaryDirectory() as temp_dir:
            out_path = Path(temp_dir) / "result.json"
            with patch("src.asr.asr_worker.import_module", side_effect=_import_module):
                with patch("sys.stdout", new_callable=StringIO) as stdout:
                    code = asr_worker.main(
                        [
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
                            "--verbose",
                        ]
                    )
            payload = json.loads(out_path.read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(payload["segments"][0]["segment_id"], "seg_0001")
        output = stdout.getvalue()
        self.assertIn("asr-worker: warning: minimal preflight failed; continuing: preflight boom", output)
        self.assertIn("asr-worker: backend.transcribe begin", output)
        self.assertIn("asr-worker: backend.transcribe end", output)
        self.assertIn("asr-worker: parse_asr_result end", output)
        self.assertIn("asr-worker: write_result_json end", output)


if __name__ == "__main__":
    unittest.main()
