import json
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from src.asr import ASRConfig
from src.asr.qwen3_asr_backend import Qwen3ASRBackend


def _load_fixture(name: str) -> dict[str, object]:
    return json.loads(Path("tests/fixtures", name).read_text(encoding="utf-8"))


class _FakeQwen3Model:
    def transcribe(self, audio_path: str, **kwargs: object):
        del audio_path
        del kwargs
        fixture = _load_fixture("qwen3_asr_minimal_segments.json")
        return {
            "chunks": [
                {
                    "timestamp": tuple(chunk["timestamp"]),
                    "text": chunk["text"],
                }
                for chunk in fixture["raw_chunks"]
            ]
        }


class _TorchModuleWithTo:
    def __init__(self) -> None:
        self.to_calls: list[str] = []

    def to(self, device: str) -> None:
        self.to_calls.append(device)


class Qwen3ASRBackendTests(unittest.TestCase):
    def test_model_init_and_transcribe_call_shape(self) -> None:
        backend = Qwen3ASRBackend()
        from_pretrained_calls: list[dict[str, object]] = []
        transcribe_calls: list[dict[str, object]] = []

        class _FakeModel:
            def transcribe(self, audio_path: str, language: str | None = None):
                transcribe_calls.append({"audio_path": audio_path, "language": language})
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
                model = _FakeModel()
                model.model = _TorchModuleWithTo()
                return model

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

        self.assertEqual(len(from_pretrained_calls), 1)
        call = from_pretrained_calls[0]
        self.assertEqual(call["dtype"], "float32")
        self.assertEqual(
            Path(str(call["model_path"])).resolve(),
            Path("models/Qwen3-ASR-1.7B").resolve(),
        )
        self.assertEqual(
            transcribe_calls,
            [{"audio_path": "in.wav", "language": "en"}],
        )

    def test_transcribe_does_not_forward_temperature_kwarg(self) -> None:
        backend = Qwen3ASRBackend()
        transcribe_kwargs: list[dict[str, object]] = []

        class _FakeModel:
            def transcribe(self, audio_path: str, language: str | None = None):
                del audio_path
                transcribe_kwargs.append({"language": language})
                return {"chunks": [{"timestamp": (0.0, 1.0), "text": "hello"}]}

        class _FakeQwen3ASRModel:
            @classmethod
            def from_pretrained(cls, model_path: str, **kwargs: object):
                del model_path
                del kwargs
                model = _FakeModel()
                model.model = _TorchModuleWithTo()
                return model

        fake_qwen_asr = types.SimpleNamespace(Qwen3ASRModel=_FakeQwen3ASRModel)

        with patch("src.asr.qwen3_asr_backend.import_module", return_value=fake_qwen_asr):
            backend.transcribe(
                "in.wav",
                ASRConfig(
                    backend_name="qwen3-asr",
                    model_path=Path("models/Qwen3-ASR-0.6B"),
                    device="cpu",
                    language="en",
                    temperature=0.7,
                ),
            )

        self.assertEqual(len(transcribe_kwargs), 1)
        self.assertNotIn("temperature", transcribe_kwargs[0])
        self.assertEqual(transcribe_kwargs[0], {"language": "en"})


    def test_transcribe_filters_kwargs_by_runtime_signature(self) -> None:
        backend = Qwen3ASRBackend()
        transcribe_kwargs: list[dict[str, object]] = []

        class _FakeModel:
            def transcribe(self, audio_path: str, language: str | None = None):
                del audio_path
                transcribe_kwargs.append({"language": language})
                return {"chunks": [{"timestamp": (0.0, 1.0), "text": "hello"}]}

        class _FakeQwen3ASRModel:
            @classmethod
            def from_pretrained(cls, model_path: str, **kwargs: object):
                del model_path
                del kwargs
                model = _FakeModel()
                model.model = _TorchModuleWithTo()
                return model

        fake_qwen_asr = types.SimpleNamespace(Qwen3ASRModel=_FakeQwen3ASRModel)

        with patch("src.asr.qwen3_asr_backend.import_module", return_value=fake_qwen_asr):
            backend.transcribe(
                "in.wav",
                ASRConfig(
                    backend_name="qwen3-asr",
                    model_path=Path("models/Qwen3-ASR-0.6B"),
                    device="cpu",
                    language="en",
                    temperature=0.7,
                ),
            )

        self.assertEqual(transcribe_kwargs, [{"language": "en"}])

    def test_transcribe_passes_temperature_only_when_signature_supports_it(self) -> None:
        backend = Qwen3ASRBackend()
        transcribe_calls: list[dict[str, object]] = []

        class _FakeModel:
            def transcribe(
                self,
                audio_path: str,
                *,
                language: str | None = None,
                temperature: float | None = None,
            ):
                del audio_path
                transcribe_calls.append({"language": language, "temperature": temperature})
                return {"chunks": [{"timestamp": (0.0, 1.0), "text": "hello"}]}

        class _FakeQwen3ASRModel:
            @classmethod
            def from_pretrained(cls, model_path: str, **kwargs: object):
                del model_path
                del kwargs
                model = _FakeModel()
                model.model = _TorchModuleWithTo()
                return model

        fake_qwen_asr = types.SimpleNamespace(Qwen3ASRModel=_FakeQwen3ASRModel)

        with patch("src.asr.qwen3_asr_backend.import_module", return_value=fake_qwen_asr):
            backend.transcribe(
                "in.wav",
                ASRConfig(
                    backend_name="qwen3-asr",
                    model_path=Path("models/Qwen3-ASR-0.6B"),
                    device="cpu",
                    language="en",
                    temperature=0.7,
                ),
            )

        self.assertEqual(transcribe_calls, [{"language": "en", "temperature": 0.7}])
    def test_temperature_filtering_avoids_unsupported_generation_warning_path(self) -> None:
        backend = Qwen3ASRBackend()
        fixture = _load_fixture("qwen3_asr_minimal_segments.json")

        class _FakeModel:
            def transcribe(self, audio_path: str, **kwargs: object):
                del audio_path
                if "temperature" in kwargs:
                    raise RuntimeError("unsupported generation flag: temperature")
                return {
                    "chunks": [
                        {
                            "timestamp": tuple(fixture["raw_chunks"][0]["timestamp"]),
                            "text": fixture["raw_chunks"][0]["text"],
                        }
                    ]
                }

        class _FakeQwen3ASRModel:
            @classmethod
            def from_pretrained(cls, model_path: str, **kwargs: object):
                del model_path
                del kwargs
                model = _FakeModel()
                model.model = _TorchModuleWithTo()
                return model

        fake_qwen_asr = types.SimpleNamespace(Qwen3ASRModel=_FakeQwen3ASRModel)

        with patch("src.asr.qwen3_asr_backend.import_module", return_value=fake_qwen_asr):
            result = backend.transcribe(
                "in.wav",
                ASRConfig(
                    backend_name="qwen3-asr",
                    model_path=Path("models/Qwen3-ASR-0.6B"),
                    device="cpu",
                    language="en",
                    temperature=0.7,
                ),
            )

        self.assertEqual(result["segments"][0]["text"], fixture["raw_chunks"][0]["text"])

    def test_from_pretrained_does_not_receive_device_kwarg(self) -> None:
        backend = Qwen3ASRBackend()
        from_pretrained_kwargs: list[dict[str, object]] = []

        class _FakeModel:
            def transcribe(self, audio_path: str, **kwargs: object):
                del audio_path
                del kwargs
                return {"chunks": [{"timestamp": (0.0, 1.0), "text": "hello"}]}

        class _FakeQwen3ASRModel:
            @classmethod
            def from_pretrained(cls, model_path: str, **kwargs: object):
                del model_path
                from_pretrained_kwargs.append(dict(kwargs))
                model = _FakeModel()
                model.model = _TorchModuleWithTo()
                return model

        fake_qwen_asr = types.SimpleNamespace(Qwen3ASRModel=_FakeQwen3ASRModel)

        with patch("src.asr.qwen3_asr_backend.import_module", return_value=fake_qwen_asr):
            backend.transcribe(
                "in.wav",
                ASRConfig(
                    backend_name="qwen3-asr",
                    model_path=Path("models/Qwen3-ASR-0.6B"),
                    device="cuda",
                    compute_type="float16",
                ),
            )

        self.assertEqual(len(from_pretrained_kwargs), 1)
        self.assertNotIn("device", from_pretrained_kwargs[0])
        self.assertEqual(from_pretrained_kwargs[0].get("dtype"), "float16")

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
                model = _FakeModel()
                model.model = _TorchModuleWithTo()
                return model

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
            with self.assertRaises(ValueError) as ctx:
                backend.transcribe(
                    "in.wav",
                    ASRConfig(backend_name="qwen3-asr", model_path=Path("models/qwen3"), device="cpu"),
                )

        message = str(ctx.exception)
        self.assertIn("cutscene-locator[asr_qwen3]", message)
        self.assertIn("qwen_asr", message)

    def test_qwen3_error_paths_are_deterministic_and_preserve_exception_chaining(self) -> None:
        class _FakeModelWithoutTo:
            def transcribe(self, audio_path: str, **kwargs: object):
                del audio_path
                del kwargs
                return {"chunks": [{"timestamp": (0.0, 1.0), "text": "hello"}]}

        class _FakeQwen3ASRModelLoaderMismatch:
            @classmethod
            def from_pretrained(cls, model_path: str, **kwargs: object):
                del model_path
                del kwargs
                return _FakeModelWithoutTo()

        class _FakeQwen3ASRModelInitFail:
            @classmethod
            def from_pretrained(cls, model_path: str, **kwargs: object):
                del model_path
                del kwargs
                raise RuntimeError("init failed")

        scenarios = (
            {
                "name": "missing_optional_dependency",
                "import_side_effect": ModuleNotFoundError("No module named 'qwen_asr'"),
                "expected_message_fragments": ("cutscene-locator[asr_qwen3]", "qwen_asr"),
                "expected_cause_type": ModuleNotFoundError,
                "expect_verbose_traceback": False,
            },
            {
                "name": "loader_api_mismatch",
                "import_return": types.SimpleNamespace(Qwen3ASRModel=_FakeQwen3ASRModelLoaderMismatch),
                "expected_message_fragments": (
                    "device transfer is unsupported",
                    "loader/API mismatch",
                    "Qwen3ASRModel.model.to(device)",
                ),
                "expected_cause_type": None,
                "expect_verbose_traceback": False,
            },
            {
                "name": "runtime_init_failure_with_verbose_traceback",
                "import_return": types.SimpleNamespace(Qwen3ASRModel=_FakeQwen3ASRModelInitFail),
                "expected_message_fragments": (
                    "config.json",
                    "tokenizer_config.json",
                    "processor_config.json / preprocessor_config.json are optional",
                    "qwen_asr/transformers/torch version compatibility",
                    "Qwen3ASRModel loading",
                    "optional runtime dependencies",
                ),
                "expected_cause_type": RuntimeError,
                "expect_verbose_traceback": True,
                "expected_traceback_fragment": "RuntimeError: init failed",
            },
        )

        for scenario in scenarios:
            with self.subTest(name=scenario["name"]):
                backend = Qwen3ASRBackend()
                verbose_logs: list[str] = []
                config = ASRConfig(
                    backend_name="qwen3-asr",
                    model_path=Path("models/qwen3"),
                    device="cpu",
                    log_callback=verbose_logs.append,
                )

                if "import_side_effect" in scenario:
                    import_patch = patch(
                        "src.asr.qwen3_asr_backend.import_module",
                        side_effect=scenario["import_side_effect"],
                    )
                else:
                    import_patch = patch(
                        "src.asr.qwen3_asr_backend.import_module",
                        return_value=scenario["import_return"],
                    )

                with import_patch:
                    with self.assertRaises(ValueError) as ctx:
                        backend.transcribe("in.wav", config)

                message = str(ctx.exception)
                for fragment in scenario["expected_message_fragments"]:
                    self.assertIn(fragment, message)

                expected_cause_type = scenario["expected_cause_type"]
                if expected_cause_type is None:
                    self.assertIsNone(ctx.exception.__cause__)
                else:
                    self.assertIsInstance(ctx.exception.__cause__, expected_cause_type)

                if scenario["expect_verbose_traceback"]:
                    self.assertTrue(any("Traceback (most recent call last):" in entry for entry in verbose_logs))
                    self.assertTrue(
                        any(scenario["expected_traceback_fragment"] in entry for entry in verbose_logs)
                    )
                else:
                    self.assertEqual(verbose_logs, [])


    def test_transcribe_failure_logs_traceback_only_when_verbose_and_preserves_cause(self) -> None:
        backend = Qwen3ASRBackend()

        class _FakeModel:
            def __init__(self) -> None:
                self.model = _TorchModuleWithTo()

            def transcribe(self, audio_path: str, **kwargs: object):
                del audio_path
                del kwargs
                raise RuntimeError("decode boom")

        class _FakeQwen3ASRModel:
            @classmethod
            def from_pretrained(cls, model_path: str, **kwargs: object):
                del model_path
                del kwargs
                return _FakeModel()

        fake_qwen_asr = types.SimpleNamespace(Qwen3ASRModel=_FakeQwen3ASRModel)

        with patch("src.asr.qwen3_asr_backend.import_module", return_value=fake_qwen_asr):
            with self.assertRaises(ValueError) as ctx_non_verbose:
                backend.transcribe(
                    "in.wav",
                    ASRConfig(backend_name="qwen3-asr", model_path=Path("models/qwen3"), device="cpu"),
                )

        self.assertIsInstance(ctx_non_verbose.exception.__cause__, RuntimeError)

        verbose_logs: list[str] = []
        with patch("src.asr.qwen3_asr_backend.import_module", return_value=fake_qwen_asr):
            with self.assertRaises(ValueError) as ctx_verbose:
                backend.transcribe(
                    "in.wav",
                    ASRConfig(
                        backend_name="qwen3-asr",
                        model_path=Path("models/qwen3"),
                        device="cpu",
                        log_callback=verbose_logs.append,
                    ),
                )

        self.assertIsInstance(ctx_verbose.exception.__cause__, RuntimeError)
        self.assertTrue(any("Traceback (most recent call last):" in entry for entry in verbose_logs))
        self.assertTrue(any("RuntimeError: decode boom" in entry for entry in verbose_logs))
    def test_unsupported_options_raise_deterministic_error(self) -> None:
        backend = Qwen3ASRBackend()

        class _FakeQwen3ASRModel:
            @classmethod
            def from_pretrained(cls, model_path: str, **kwargs: object):
                del model_path
                del kwargs
                model = _FakeQwen3Model()
                model.model = _TorchModuleWithTo()
                return model

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

        self.assertIn("Supported backend controls are: language, device, dtype", str(ctx.exception))

    def test_device_move_uses_top_level_to_when_model_attribute_missing(self) -> None:
        backend = Qwen3ASRBackend()
        to_calls: list[str] = []

        class _FakeModelWithTo:
            def to(self, device: str) -> None:
                to_calls.append(device)

            def transcribe(self, audio_path: str, **kwargs: object):
                del audio_path
                del kwargs
                return {"chunks": [{"timestamp": (0.0, 1.0), "text": "hello"}]}

        class _FakeQwen3ASRModel:
            @classmethod
            def from_pretrained(cls, model_path: str, **kwargs: object):
                del model_path
                del kwargs
                return _FakeModelWithTo()

        fake_qwen_asr = types.SimpleNamespace(Qwen3ASRModel=_FakeQwen3ASRModel)

        with patch("src.asr.qwen3_asr_backend.import_module", return_value=fake_qwen_asr):
            backend.transcribe(
                "in.wav",
                ASRConfig(backend_name="qwen3-asr", model_path=Path("models/qwen3"), device="cuda"),
            )

        self.assertEqual(to_calls, ["cuda"])

    def test_timestamp_normalization_is_stable_for_backend_edge_fixture(self) -> None:
        backend = Qwen3ASRBackend()
        fixture = _load_fixture("asr_timestamp_edges_qwen3.json")
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
                model = _FakeModel()
                model.model = _TorchModuleWithTo()
                return model

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
                model = _FakeModel()
                model.model = _TorchModuleWithTo()
                return model

        fake_qwen_asr = types.SimpleNamespace(Qwen3ASRModel=_FakeQwen3ASRModel)

        with patch("src.asr.qwen3_asr_backend.import_module", return_value=fake_qwen_asr):
            with self.assertRaisesRegex(ValueError, "must be non-negative"):
                backend.transcribe(
                    "in.wav",
                    ASRConfig(backend_name="qwen3-asr", model_path=Path("models/qwen3"), device="cpu"),
                )


if __name__ == "__main__":
    unittest.main()
