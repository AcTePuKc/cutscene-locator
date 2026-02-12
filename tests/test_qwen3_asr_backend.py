import json
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from src.asr import ASRConfig
from src.asr.qwen3_asr_backend import Qwen3ASRBackend


class _FakeQwen3Model:
    def transcribe(self, audio_path: str, **kwargs: object):
        del audio_path
        del kwargs
        return {
            "chunks": [
                {"timestamp": (0.0, 1.0), "text": " hello "},
                {"timestamp": (1.1, 2.0), "text": "world"},
            ]
        }


class Qwen3ASRBackendTests(unittest.TestCase):
    def test_model_init_and_transcribe_call_shape(self) -> None:
        backend = Qwen3ASRBackend()
        from_pretrained_calls: list[dict[str, object]] = []
        transcribe_calls: list[dict[str, object]] = []

        class _FakeModel:
            def transcribe(self, audio_path: str, **kwargs: object):
                transcribe_calls.append({"audio_path": audio_path, **kwargs})
                return {
                    "chunks": [
                        {"timestamp": (0.0, 0.9), "text": "variant line one"},
                        {"timestamp": (1.0, 2.5), "text": "variant line two"},
                    ]
                }

        class _FakeQwen3ASRModel:
            @classmethod
            def from_pretrained(cls, model_path: str, **kwargs: object):
                from_pretrained_calls.append({"model_path": model_path, **kwargs})
                return _FakeModel()

        fake_qwen_asr = types.SimpleNamespace(Qwen3ASRModel=_FakeQwen3ASRModel)

        with patch("src.asr.qwen3_asr_backend.import_module", return_value=fake_qwen_asr):
            backend.transcribe(
                "in.wav",
                ASRConfig(
                    backend_name="qwen3-asr",
                    model_path=Path("models/Qwen3-ASR-1.7B"),
                    device="cpu",
                    language="en",
                    compute_type="float32",
                ),
            )

        self.assertEqual(
            from_pretrained_calls,
            [
                {
                    "model_path": "models/Qwen3-ASR-1.7B",
                    "device": "cpu",
                    "torch_dtype": "float32",
                }
            ],
        )
        self.assertEqual(
            transcribe_calls,
            [{"audio_path": "in.wav", "language": "en", "return_timestamps": True}],
        )

    def test_pipeline_smoke_rejects_non_tuple_timestamps(self) -> None:
        backend = Qwen3ASRBackend()

        class _FakeModel:
            def transcribe(self, audio_path: str, **kwargs: object):
                del audio_path
                del kwargs
                return {
                    "chunks": [
                        {
                            "timestamp": [0.0, 1.0],
                            "text": "invalid list timestamp should fail",
                        }
                    ]
                }

        class _FakeQwen3ASRModel:
            @classmethod
            def from_pretrained(cls, model_path: str, **kwargs: object):
                del model_path
                del kwargs
                return _FakeModel()

        fake_qwen_asr = types.SimpleNamespace(Qwen3ASRModel=_FakeQwen3ASRModel)

        with patch("src.asr.qwen3_asr_backend.import_module", return_value=fake_qwen_asr):
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

        class _FakeQwen3ASRModel:
            @classmethod
            def from_pretrained(cls, model_path: str, **kwargs: object):
                del model_path
                del kwargs
                return _FakeQwen3Model()

        fake_qwen_asr = types.SimpleNamespace(Qwen3ASRModel=_FakeQwen3ASRModel)

        with patch("src.asr.qwen3_asr_backend.import_module", return_value=fake_qwen_asr):
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

        class _FakeQwen3ASRModel:
            @classmethod
            def from_pretrained(cls, model_path: str, **kwargs: object):
                del model_path
                del kwargs
                raise RuntimeError("init failed")

        fake_qwen_asr = types.SimpleNamespace(Qwen3ASRModel=_FakeQwen3ASRModel)

        with patch("src.asr.qwen3_asr_backend.import_module", return_value=fake_qwen_asr):
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
        self.assertIn("qwen_asr/transformers/torch version compatibility", message)
        self.assertIn("Qwen3ASRModel loading", message)
        self.assertIn("optional runtime dependencies", message)

    def test_unsupported_options_raise_deterministic_error(self) -> None:
        backend = Qwen3ASRBackend()

        class _FakeQwen3ASRModel:
            @classmethod
            def from_pretrained(cls, model_path: str, **kwargs: object):
                del model_path
                del kwargs
                return _FakeQwen3Model()

        fake_qwen_asr = types.SimpleNamespace(Qwen3ASRModel=_FakeQwen3ASRModel)

        with patch("src.asr.qwen3_asr_backend.import_module", return_value=fake_qwen_asr):
            with self.assertRaisesRegex(ValueError, "does not support options: beam_size, vad_filter") as ctx:
                backend.transcribe(
                    "in.wav",
                    ASRConfig(
                        backend_name="qwen3-asr",
                        model_path=Path("models/qwen3"),
                        device="cpu",
                        beam_size=4,
                        vad_filter=True,
                    ),
                )

        self.assertIn("Supported options are: language, return_timestamps, device, torch_dtype", str(ctx.exception))

    def test_timestamp_normalization_is_stable_for_backend_edge_fixture(self) -> None:
        backend = Qwen3ASRBackend()
        fixture = json.loads(Path("tests/fixtures/asr_timestamp_edges_qwen3.json").read_text(encoding="utf-8"))
        raw_chunks = []
        for chunk in fixture["raw_chunks"]:
            raw_chunks.append({"timestamp": tuple(chunk["timestamp"]), "text": chunk["text"]})

        class _FakeModel:
            def transcribe(self, audio_path: str, **kwargs: object):
                del audio_path
                del kwargs
                return {"chunks": raw_chunks}

        class _FakeQwen3ASRModel:
            @classmethod
            def from_pretrained(cls, model_path: str, **kwargs: object):
                del model_path
                del kwargs
                return _FakeModel()

        fake_qwen_asr = types.SimpleNamespace(Qwen3ASRModel=_FakeQwen3ASRModel)

        with patch("src.asr.qwen3_asr_backend.import_module", return_value=fake_qwen_asr):
            result = backend.transcribe(
                "in.wav",
                ASRConfig(backend_name="qwen3-asr", model_path=Path("models/qwen3"), device="cpu"),
            )

        self.assertEqual(result["segments"], fixture["expected"])

    def test_pathological_timestamps_are_rejected_without_fabrication(self) -> None:
        backend = Qwen3ASRBackend()

        class _FakeModel:
            def transcribe(self, audio_path: str, **kwargs: object):
                del audio_path
                del kwargs
                return {"chunks": [{"timestamp": (-0.1, 1.0), "text": "bad"}]}

        class _FakeQwen3ASRModel:
            @classmethod
            def from_pretrained(cls, model_path: str, **kwargs: object):
                del model_path
                del kwargs
                return _FakeModel()

        fake_qwen_asr = types.SimpleNamespace(Qwen3ASRModel=_FakeQwen3ASRModel)

        with patch("src.asr.qwen3_asr_backend.import_module", return_value=fake_qwen_asr):
            with self.assertRaisesRegex(ValueError, "must be non-negative"):
                backend.transcribe(
                    "in.wav",
                    ASRConfig(backend_name="qwen3-asr", model_path=Path("models/qwen3"), device="cpu"),
                )


if __name__ == "__main__":
    unittest.main()
