from __future__ import annotations

import inspect
import unittest

from src.asr import ASRConfig
from src.asr.adapters import (
    ASRExecutionContext,
    _ADAPTER_REGISTRY,
    _BaseASRAdapter,
    apply_cross_chunk_continuity,
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


class ASRCrossChunkContinuityTests(unittest.TestCase):
    def test_converts_chunk_local_timestamps_to_absolute_and_preserves_order(self) -> None:
        result = apply_cross_chunk_continuity(
            asr_result={
                "segments": [
                    {"segment_id": "seg_b", "start": 0.3, "end": 0.7, "text": "line b", "chunk_index": 1},
                    {"segment_id": "seg_a", "start": 0.1, "end": 0.2, "text": "line a", "chunk_index": 0},
                ],
                "meta": {"backend": "mock", "model": "m", "version": "v", "device": "cpu"},
            },
            chunk_offsets_by_index={0: 0.0, 1: 300.0},
        )

        self.assertEqual([seg["segment_id"] for seg in result["segments"]], ["seg_a", "seg_b"])
        self.assertEqual(result["segments"][1]["start"], 300.3)
        self.assertEqual(result["segments"][1]["end"], 300.7)

    def test_merges_repeated_boundary_overlap_segments_deterministically(self) -> None:
        result = apply_cross_chunk_continuity(
            asr_result={
                "segments": [
                    {"segment_id": "seg_0001", "start": 299.6, "end": 300.0, "text": "Stay close.", "chunk_index": 0},
                    {"segment_id": "seg_0002", "start": 0.0, "end": 0.4, "text": "Stay close.", "chunk_index": 1},
                ],
                "meta": {"backend": "mock", "model": "m", "version": "v", "device": "cpu"},
            },
            chunk_offsets_by_index={0: 0.0, 1: 300.0},
        )

        self.assertEqual(len(result["segments"]), 1)
        merged = result["segments"][0]
        self.assertEqual(merged["segment_id"], "seg_0001")
        self.assertEqual(merged["start"], 299.6)
        self.assertEqual(merged["end"], 300.4)
        self.assertEqual(merged["text"], "Stay close.")

    def test_merges_split_utterance_and_tiny_boundary_fragment(self) -> None:
        result = apply_cross_chunk_continuity(
            asr_result={
                "segments": [
                    {"segment_id": "seg_0001", "start": 299.6, "end": 300.0, "text": "I can't", "chunk_index": 0},
                    {"segment_id": "seg_0002", "start": 0.0, "end": 0.5, "text": "can't lose you", "chunk_index": 1},
                    {"segment_id": "seg_0003", "start": 0.5, "end": 0.58, "text": "now", "chunk_index": 1},
                ],
                "meta": {"backend": "mock", "model": "m", "version": "v", "device": "cpu"},
            },
            chunk_offsets_by_index={0: 0.0, 1: 300.0},
        )

        self.assertEqual(len(result["segments"]), 1)
        merged = result["segments"][0]
        self.assertEqual(merged["text"], "I can't lose you now")
        self.assertEqual(merged["start"], 299.6)
        self.assertEqual(merged["end"], 300.58)


if __name__ == "__main__":
    unittest.main()
