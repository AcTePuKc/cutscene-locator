import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import json

from src.align import Qwen3ForcedAligner


def _load_fixture(name: str) -> dict[str, object]:
    return json.loads(Path("tests/fixtures", name).read_text(encoding="utf-8"))



class Qwen3ForcedAlignerTests(unittest.TestCase):
    def test_missing_dependency_has_clear_install_hint(self) -> None:
        aligner = Qwen3ForcedAligner(
            SimpleNamespace(model_path=Path("models/qwen3"), device="cpu")
        )

        with patch("src.align.qwen3_forced_aligner.import_module", side_effect=ModuleNotFoundError()):
            with self.assertRaisesRegex(ValueError, "cutscene-locator\\[asr_qwen3\\]"):
                aligner.align(
                    "in.wav",
                    [{"ref_id": "span_0001", "text": "you"}],
                )

    def test_alignment_uses_reference_text_and_defaults_invalid_confidence(self) -> None:
        aligner = Qwen3ForcedAligner(
            SimpleNamespace(model_path=Path("models/qwen3"), device="cpu")
        )
        fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))

        class _ConfidenceEdgePipeline:
            def __call__(self, audio_path: str, return_timestamps: str):
                del audio_path
                del return_timestamps
                return {
                    "chunks": [
                        {"timestamp": (10.0, 11.0), "text": "mismatch", "confidence": "bad"},
                    ]
                }

        fake_transformers = types.SimpleNamespace(
            pipeline=lambda **kwargs: _ConfidenceEdgePipeline(),
        )

        def _fake_import(name: str):
            if name == "torch":
                return fake_torch
            if name == "transformers":
                return fake_transformers
            raise ModuleNotFoundError(name)

        with patch("src.align.qwen3_forced_aligner.import_module", side_effect=_fake_import):
            result = aligner.align(
                "in.wav",
                [{"ref_id": "span_0001", "text": "reference text"}],
            )

        self.assertEqual(result["transcript_text"], "reference text")
        self.assertEqual(result["spans"][0]["text"], "reference text")
        self.assertEqual(result["spans"][0]["confidence"], 1.0)

    def test_forced_alignment_minimal_fixture_contract_shape(self) -> None:
        fixture = _load_fixture("qwen3_forced_alignment_minimal_spans.json")
        aligner = Qwen3ForcedAligner(SimpleNamespace(model_path=Path("models/qwen3"), device="cpu"))
        fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))

        class _FixturePipeline:
            def __call__(self, audio_path: str, return_timestamps: str):
                del audio_path
                del return_timestamps
                return {
                    "chunks": [
                        {"timestamp": tuple(chunk["timestamp"]), "text": chunk["text"], "confidence": chunk["confidence"]}
                        for chunk in fixture["raw_chunks"]
                    ]
                }

        fake_transformers = types.SimpleNamespace(pipeline=lambda **kwargs: _FixturePipeline())

        def _fake_import(name: str):
            if name == "torch":
                return fake_torch
            if name == "transformers":
                return fake_transformers
            raise ModuleNotFoundError(name)

        with patch("src.align.qwen3_forced_aligner.import_module", side_effect=_fake_import):
            result = aligner.align("in.wav", fixture["reference_spans"])

        self.assertEqual(result["transcript_text"], fixture["expected_transcript_text"])
        self.assertEqual(result["spans"], fixture["expected_spans"])


if __name__ == "__main__":
    unittest.main()
