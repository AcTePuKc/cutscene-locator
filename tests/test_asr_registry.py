import importlib
import unittest
from unittest.mock import patch

from src.asr import (
    ASRConfig,
    get_backend,
    list_backend_status,
    list_backends,
    list_declared_backends,
)


class ASRRegistryTests(unittest.TestCase):
    def test_registry_lists_required_backends(self) -> None:
        backends = list_backends()
        self.assertIn("faster-whisper", backends)
        self.assertIn("mock", backends)

    def test_registry_includes_qwen3_when_dependencies_installed(self) -> None:
        with patch("importlib.util.find_spec", side_effect=lambda name: object()):
            registry_module = importlib.import_module("src.asr.registry")
            registry_module = importlib.reload(registry_module)
            self.assertIn("qwen3-asr", registry_module.list_backends())


    def test_declared_backends_lists_all_names(self) -> None:
        self.assertEqual(list_declared_backends(), ["faster-whisper", "mock", "qwen3-asr"])

    def test_registry_reports_disabled_backend_dependencies(self) -> None:
        def fake_find_spec(name: str) -> object | None:
            if name in {"torch", "transformers"}:
                return None
            return object()

        with patch("src.asr.registry.find_spec", side_effect=fake_find_spec):
            statuses = {status.name: status for status in list_backend_status()}

        self.assertIn("qwen3-asr", statuses)
        qwen_status = statuses["qwen3-asr"]
        self.assertFalse(qwen_status.enabled)
        self.assertEqual(qwen_status.missing_dependencies, ("torch", "transformers"))
        self.assertEqual(qwen_status.install_extra, "asr_qwen3")
        self.assertEqual(
            qwen_status.reason,
            "missing optional dependencies: torch, transformers",
        )

    def test_get_backend_returns_faster_whisper_capabilities(self) -> None:
        backend = get_backend("faster-whisper")

        self.assertEqual(backend.name, "faster-whisper")
        self.assertFalse(backend.capabilities.supports_alignment)
        self.assertFalse(backend.capabilities.supports_word_timestamps)

    def test_get_backend_returns_capabilities(self) -> None:
        backend = get_backend("mock")

        self.assertEqual(backend.name, "mock")
        self.assertFalse(backend.capabilities.supports_alignment)
        self.assertFalse(backend.capabilities.supports_word_timestamps)


    def test_qwen3_backend_marks_alignment_capability(self) -> None:
        with patch("importlib.util.find_spec", side_effect=lambda name: object()):
            registry_module = importlib.import_module("src.asr.registry")
            registry_module = importlib.reload(registry_module)
            backend = registry_module.get_backend("qwen3-asr")

        self.assertTrue(backend.capabilities.supports_alignment)

    def test_get_backend_unknown_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown ASR backend"):
            get_backend("missing")

    def test_asr_config_defaults(self) -> None:
        config = ASRConfig(backend_name="mock")

        self.assertEqual(config.backend_name, "mock")
        self.assertEqual(config.device, "auto")
        self.assertIsNone(config.model_path)
        self.assertIsNone(config.progress_callback)
        self.assertIsNone(config.cancel_check)


if __name__ == "__main__":
    unittest.main()
