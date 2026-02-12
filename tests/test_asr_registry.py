import importlib
import unittest
from unittest.mock import patch

from src.asr import (
    ASRConfig,
    dispatch_asr_transcription,
    get_asr_adapter,
    get_backend,
    list_asr_adapters,
    list_backend_status,
    list_backends,
    list_declared_backends,
    validate_backend_capabilities,
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
        self.assertEqual(list_declared_backends(), ["faster-whisper", "mock", "qwen3-asr", "qwen3-forced-aligner", "vibevoice", "whisperx"])

    def test_registry_reports_disabled_backend_dependencies(self) -> None:
        def fake_find_spec(name: str) -> object | None:
            if name in {"torch", "transformers"}:
                return None
            return object()

        with patch("src.asr.registry.find_spec", side_effect=fake_find_spec):
            statuses = {status.name: status for status in list_backend_status()}

        self.assertIn("qwen3-asr", statuses)
        self.assertIn("qwen3-forced-aligner", statuses)
        qwen_status = statuses["qwen3-asr"]
        self.assertFalse(qwen_status.enabled)
        self.assertEqual(qwen_status.missing_dependencies, ("torch", "transformers"))
        self.assertEqual(qwen_status.install_extra, "asr_qwen3")
        self.assertEqual(
            qwen_status.reason,
            "missing optional dependencies: torch, transformers",
        )


    def test_registry_reports_whisperx_disabled_dependencies(self) -> None:
        def fake_find_spec(name: str) -> object | None:
            if name in {"whisperx", "torch"}:
                return None
            return object()

        with patch("src.asr.registry.find_spec", side_effect=fake_find_spec):
            statuses = {status.name: status for status in list_backend_status()}

        self.assertIn("whisperx", statuses)
        whisperx_status = statuses["whisperx"]
        self.assertFalse(whisperx_status.enabled)
        self.assertEqual(whisperx_status.missing_dependencies, ("whisperx", "torch"))
        self.assertEqual(whisperx_status.install_extra, "asr_whisperx")
        self.assertEqual(
            whisperx_status.reason,
            "missing optional dependencies: whisperx, torch",
        )


    def test_registry_reports_vibevoice_disabled_dependencies(self) -> None:
        def fake_find_spec(name: str) -> object | None:
            if name in {"vibevoice", "torch"}:
                return None
            return object()

        with patch("src.asr.registry.find_spec", side_effect=fake_find_spec):
            statuses = {status.name: status for status in list_backend_status()}

        self.assertIn("vibevoice", statuses)
        vibevoice_status = statuses["vibevoice"]
        self.assertFalse(vibevoice_status.enabled)
        self.assertEqual(vibevoice_status.missing_dependencies, ("vibevoice", "torch"))
        self.assertEqual(vibevoice_status.install_extra, "asr_vibevoice")
        self.assertEqual(
            vibevoice_status.reason,
            "missing optional dependencies: vibevoice, torch",
        )

    def test_get_backend_returns_faster_whisper_capabilities(self) -> None:
        backend = get_backend("faster-whisper")

        self.assertEqual(backend.name, "faster-whisper")
        self.assertTrue(backend.capabilities.supports_segment_timestamps)
        self.assertFalse(backend.capabilities.supports_alignment)
        self.assertFalse(backend.capabilities.supports_word_timestamps)

    def test_get_backend_returns_capabilities(self) -> None:
        backend = get_backend("mock")

        self.assertEqual(backend.name, "mock")
        self.assertTrue(backend.capabilities.supports_segment_timestamps)
        self.assertFalse(backend.capabilities.supports_alignment)
        self.assertFalse(backend.capabilities.supports_word_timestamps)

    def test_qwen3_asr_backend_does_not_mark_alignment_capability(self) -> None:
        with patch("importlib.util.find_spec", side_effect=lambda name: object()):
            registry_module = importlib.import_module("src.asr.registry")
            registry_module = importlib.reload(registry_module)
            backend = registry_module.get_backend("qwen3-asr")

        self.assertFalse(backend.capabilities.supports_alignment)

    def test_qwen3_forced_aligner_marks_alignment_capability(self) -> None:
        with patch("importlib.util.find_spec", side_effect=lambda name: object()):
            registry_module = importlib.import_module("src.asr.registry")
            registry_module = importlib.reload(registry_module)
            backend = registry_module.get_backend("qwen3-forced-aligner")

        self.assertTrue(backend.capabilities.supports_alignment)


    def test_whisperx_backend_capabilities(self) -> None:
        with patch("importlib.util.find_spec", side_effect=lambda name: object()):
            registry_module = importlib.import_module("src.asr.registry")
            registry_module = importlib.reload(registry_module)
            backend = registry_module.get_backend("whisperx")

        self.assertTrue(backend.capabilities.supports_segment_timestamps)
        self.assertFalse(backend.capabilities.supports_alignment)
        self.assertFalse(backend.capabilities.supports_word_timestamps)
        self.assertTrue(backend.capabilities.supports_diarization)


    def test_vibevoice_backend_capabilities(self) -> None:
        with patch("importlib.util.find_spec", side_effect=lambda name: object()):
            registry_module = importlib.import_module("src.asr.registry")
            registry_module = importlib.reload(registry_module)
            backend = registry_module.get_backend("vibevoice")

        self.assertTrue(backend.capabilities.supports_segment_timestamps)
        self.assertFalse(backend.capabilities.supports_alignment)
        self.assertFalse(backend.capabilities.supports_word_timestamps)

    def test_get_backend_unknown_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown ASR backend"):
            get_backend("missing")

    def test_validate_backend_capabilities_rejects_alignment_when_disallowed(self) -> None:
        backend = get_backend("mock")
        validate_backend_capabilities(
            backend,
            requires_segment_timestamps=True,
            allows_alignment_backends=False,
        )

    def test_adapter_registry_returns_backend_adapter(self) -> None:
        adapter = get_asr_adapter("mock")
        self.assertEqual(adapter.backend_name, "mock")
        self.assertIn("mock", list_asr_adapters())
        self.assertIn("vibevoice", list_asr_adapters())

    def test_adapter_registry_unknown_backend_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "No ASR adapter registered"):
            get_asr_adapter("qwen3-forced-aligner")


    def test_dispatch_asr_transcription_uses_registered_adapter(self) -> None:
        from unittest.mock import patch

        class _Adapter:
            backend_name = "mock"

            def transcribe(self, audio_path: str, config: object, context: object) -> dict[str, object]:
                del context
                self._audio_path = audio_path
                self._backend_name = getattr(config, "backend_name", "")
                return {
                    "segments": [
                        {"segment_id": "seg_0001", "start": 0.0, "end": 0.1, "text": "ok"}
                    ],
                    "meta": {"backend": "mock", "model": "fixture", "version": "1", "device": "cpu"},
                }

        adapter = _Adapter()
        with patch("src.asr.adapters.get_asr_adapter", return_value=adapter):
            result = dispatch_asr_transcription(
                audio_path="audio.wav",
                config=ASRConfig(backend_name="mock"),
                context=object(),
            )

        self.assertEqual(result["meta"]["backend"], "mock")
        self.assertEqual(adapter._audio_path, "audio.wav")
        self.assertEqual(adapter._backend_name, "mock")

    def test_dispatch_asr_transcription_runs_capability_preflight(self) -> None:
        from src.asr.registry import BackendCapabilities, BackendRegistration

        fake_registration = BackendRegistration(
            name="qwen3-forced-aligner",
            backend_class=object,
            capabilities=BackendCapabilities(
                supports_segment_timestamps=True,
                supports_word_timestamps=False,
                supports_alignment=True,
                supports_diarization=False,
                max_audio_duration=None,
            ),
        )

        with patch("src.asr.adapters.get_backend", return_value=fake_registration):
            with self.assertRaisesRegex(
                ValueError,
                "alignment backend and cannot be used with --asr-backend",
            ):
                dispatch_asr_transcription(
                    audio_path="audio.wav",
                    config=ASRConfig(backend_name="qwen3-forced-aligner"),
                    context=object(),
                )

    def test_asr_config_defaults(self) -> None:
        config = ASRConfig(backend_name="mock")

        self.assertEqual(config.backend_name, "mock")
        self.assertEqual(config.device, "auto")
        self.assertIsNone(config.model_path)
        self.assertIsNone(config.progress_callback)
        self.assertIsNone(config.cancel_check)


if __name__ == "__main__":
    unittest.main()
