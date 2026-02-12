import json
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from src.asr import ASRConfig
from src.asr.qwen3_asr_backend import Qwen3ASRBackend


class _FakePipeline:
    def __call__(self, audio_path: str, return_timestamps: bool):
        del audio_path
        del return_timestamps
        return {
            "chunks": [
                {"timestamp": (0.0, 1.0), "text": " hello "},
                {"timestamp": (1.1, 2.0), "text": "world"},
            ]
        }


class Qwen3ASRBackendTests(unittest.TestCase):
    def test_pipeline_smoke_contract_for_qwen3_variants(self) -> None:
        backend = Qwen3ASRBackend()
        fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
        pipeline_calls: list[dict[str, object]] = []
        transcribe_calls: list[dict[str, object]] = []

        def _pipeline(**kwargs: object):
            pipeline_calls.append(kwargs)

            def _transcribe(audio_path: str, return_timestamps: bool):
                transcribe_calls.append(
                    {
                        "audio_path": audio_path,
                        "return_timestamps": return_timestamps,
                    }
                )
                return {
                    "chunks": [
                        {"timestamp": (0.0, 0.9), "text": "variant line one"},
                        {"timestamp": (1.0, 2.5), "text": "variant line two"},
                    ]
                }

            return _transcribe

        fake_transformers = types.SimpleNamespace(pipeline=_pipeline)

        def _fake_import(name: str):
            if name == "torch":
                return fake_torch
            if name == "transformers":
                return fake_transformers
            raise ModuleNotFoundError(name)

        with patch("src.asr.qwen3_asr_backend.import_module", side_effect=_fake_import):
            backend.transcribe(
                "in.wav",
                ASRConfig(
                    backend_name="qwen3-asr",
                    model_path=Path("models/Qwen3-ASR-1.7B"),
                    device="cpu",
                ),
            )

        self.assertEqual(len(pipeline_calls), 1)
        self.assertEqual(
            pipeline_calls[0],
            {
                "task": "automatic-speech-recognition",
                "model": "models/Qwen3-ASR-1.7B",
                "device": -1,
                "trust_remote_code": True,
            },
        )
        self.assertEqual(
            transcribe_calls,
            [{"audio_path": "in.wav", "return_timestamps": True}],
        )

    def test_pipeline_smoke_rejects_non_tuple_timestamps(self) -> None:
        backend = Qwen3ASRBackend()
        fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))

        def _pipeline(**kwargs: object):
            del kwargs

            def _transcribe(audio_path: str, return_timestamps: bool):
                del audio_path
                del return_timestamps
                return {
                    "chunks": [
                        {
                            "timestamp": [0.0, 1.0],
                            "text": "invalid list timestamp should fail",
                        }
                    ]
                }

            return _transcribe

        fake_transformers = types.SimpleNamespace(pipeline=_pipeline)

        def _fake_import(name: str):
            if name == "torch":
                return fake_torch
            if name == "transformers":
                return fake_transformers
            raise ModuleNotFoundError(name)

        with patch("src.asr.qwen3_asr_backend.import_module", side_effect=_fake_import):
            with self.assertRaisesRegex(ValueError, "invalid timestamp format"):
                backend.transcribe(
                    "in.wav",
                    ASRConfig(
                        backend_name="qwen3-asr",
                        model_path=Path("models/Qwen3-ASR-0.6B"),
                        device="cpu",
                    ),
                )

    def test_missing_dependency_has_clear_install_hint(self) -> None:
        backend = Qwen3ASRBackend()

        with patch("src.asr.qwen3_asr_backend.import_module", side_effect=ModuleNotFoundError()):
            with self.assertRaisesRegex(ValueError, "cutscene-locator\\[asr_qwen3\\]"):
                backend.transcribe(
                    "in.wav",
                    ASRConfig(backend_name="qwen3-asr", model_path=Path("models/qwen3"), device="cpu"),
                )

    def test_backend_emits_standard_contract(self) -> None:
        backend = Qwen3ASRBackend()
        fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
        fake_transformers = types.SimpleNamespace(
            pipeline=lambda **kwargs: _FakePipeline(),
        )

        def _fake_import(name: str):
            if name == "torch":
                return fake_torch
            if name == "transformers":
                return fake_transformers
            raise ModuleNotFoundError(name)

        with patch("src.asr.qwen3_asr_backend.import_module", side_effect=_fake_import):
            with patch("src.asr.qwen3_asr_backend.version", return_value="9.9.9"):
                result = backend.transcribe(
                    "in.wav",
                    ASRConfig(backend_name="qwen3-asr", model_path=Path("models/qwen3"), device="cpu"),
                )

        self.assertEqual(result["meta"]["backend"], "qwen3-asr")
        self.assertEqual(result["meta"]["model"], "qwen3")
        self.assertEqual(result["meta"]["version"], "9.9.9")
        self.assertEqual(result["meta"]["device"], "cpu")
        self.assertEqual(result["segments"][0]["segment_id"], "seg_0001")
        self.assertEqual(result["segments"][0]["start"], 0.0)
        self.assertEqual(result["segments"][0]["end"], 1.0)
        self.assertEqual(result["segments"][0]["text"], " hello ")

    def test_model_init_error_mentions_core_contract_and_runtime_hints(self) -> None:
        backend = Qwen3ASRBackend()
        fake_transformers = types.SimpleNamespace(
            pipeline=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("init failed")),
        )

        with patch("src.asr.qwen3_asr_backend.import_module", return_value=fake_transformers):
            with self.assertRaisesRegex(ValueError, r"config\.json") as ctx:
                backend.transcribe(
                    "in.wav",
                    ASRConfig(
                        backend_name="qwen3-asr",
                        model_path=Path("models/qwen3"),
                        device="cpu",
                    ),
                )

        message = str(ctx.exception)
        self.assertIn("tokenizer_config.json", message)
        self.assertIn("processor_config.json / preprocessor_config.json are optional", message)
        self.assertNotIn("processor/preprocessor config", message)
        self.assertIn("transformers/torch version compatibility", message)
        self.assertIn("model repo supports Transformers pipeline", message)
        self.assertIn("optional runtime dependencies", message)


    def test_timestamp_normalization_is_stable_for_backend_edge_fixture(self) -> None:
        backend = Qwen3ASRBackend()
        fixture = json.loads(Path("tests/fixtures/asr_timestamp_edges_qwen3.json").read_text(encoding="utf-8"))
        raw_chunks = []
        for chunk in fixture["raw_chunks"]:
            raw_chunks.append({"timestamp": tuple(chunk["timestamp"]), "text": chunk["text"]})

        fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
        fake_transformers = types.SimpleNamespace(
            pipeline=lambda **kwargs: (lambda audio_path, return_timestamps: {"chunks": raw_chunks}),
        )

        def _fake_import(name: str):
            if name == "torch":
                return fake_torch
            if name == "transformers":
                return fake_transformers
            raise ModuleNotFoundError(name)

        with patch("src.asr.qwen3_asr_backend.import_module", side_effect=_fake_import):
            result = backend.transcribe(
                "in.wav",
                ASRConfig(backend_name="qwen3-asr", model_path=Path("models/qwen3"), device="cpu"),
            )

        self.assertEqual(result["segments"], fixture["expected"])

    def test_pathological_timestamps_are_rejected_without_fabrication(self) -> None:
        backend = Qwen3ASRBackend()
        fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
        fake_transformers = types.SimpleNamespace(
            pipeline=lambda **kwargs: (
                lambda audio_path, return_timestamps: {
                    "chunks": [{"timestamp": (-0.1, 1.0), "text": "bad"}]
                }
            ),
        )

        def _fake_import(name: str):
            if name == "torch":
                return fake_torch
            if name == "transformers":
                return fake_transformers
            raise ModuleNotFoundError(name)

        with patch("src.asr.qwen3_asr_backend.import_module", side_effect=_fake_import):
            with self.assertRaisesRegex(ValueError, "must be non-negative"):
                backend.transcribe(
                    "in.wav",
                    ASRConfig(backend_name="qwen3-asr", model_path=Path("models/qwen3"), device="cpu"),
                )


if __name__ == "__main__":
    unittest.main()
