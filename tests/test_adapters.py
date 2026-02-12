from __future__ import annotations

import inspect
import unittest

from src.asr import ASRConfig
from src.asr.adapters import (
    ASRExecutionContext,
    _ADAPTER_REGISTRY,
    _BaseASRAdapter,
    get_asr_adapter,
    list_asr_adapters,
)


class ASRAdapterTypingContractTests(unittest.TestCase):
    def test_registry_entries_are_concrete_base_adapter_subclasses(self) -> None:
        for name, adapter_class in _ADAPTER_REGISTRY.items():
            self.assertTrue(issubclass(adapter_class, _BaseASRAdapter), msg=name)
            self.assertFalse(inspect.isabstract(adapter_class), msg=name)

    def test_base_adapter_uses_explicit_abstract_transcribe_contract(self) -> None:
        self.assertTrue(inspect.isabstract(_BaseASRAdapter))
        self.assertIn("transcribe", getattr(_BaseASRAdapter, "__abstractmethods__", set()))

    def test_get_asr_adapter_returns_concrete_adapter_instance(self) -> None:
        adapter = get_asr_adapter("mock")
        self.assertIsInstance(adapter, _BaseASRAdapter)
        self.assertEqual(adapter.backend_name, "mock")

    def test_base_helpers_are_total_on_control_paths(self) -> None:
        adapter = get_asr_adapter("mock")
        config = ASRConfig(backend_name="mock")
        backend_kwargs = adapter.build_backend_kwargs(config)
        self.assertIsInstance(backend_kwargs, dict)
        self.assertIn("language", backend_kwargs)

        filtered = adapter.filter_backend_kwargs(
            {"beam_size": 1, "language": None, "vad_filter": True},
            allowed_keys={"beam_size", "language", "vad_filter"},
        )
        self.assertEqual(filtered, {"beam_size": 1, "vad_filter": True})

    def test_all_registered_adapters_support_transcribe_call_shape(self) -> None:
        for name in list_asr_adapters():
            adapter = get_asr_adapter(name)
            signature = inspect.signature(adapter.transcribe)
            self.assertEqual(list(signature.parameters), ["audio_path", "config", "context"])

            context = ASRExecutionContext(resolved_model_path=None, verbose=False)
            if name == "mock":
                with self.assertRaisesRegex(ValueError, "--mock-asr is required"):
                    adapter.transcribe("audio.wav", ASRConfig(backend_name=name), context)


if __name__ == "__main__":
    unittest.main()
