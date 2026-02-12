import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.align import Qwen3ForcedAligner


class _FakePipeline:
    def __call__(self, audio_path: str, return_timestamps: str):
        del audio_path
        del return_timestamps
        return {
            "chunks": [
                {"timestamp": (10.0, 11.0), "text": "you", "confidence": 0.9},
                {"timestamp": (11.1, 12.0), "text": "know", "confidence": 0.8},
            ]
        }


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

    def test_backend_emits_alignment_contract(self) -> None:
        aligner = Qwen3ForcedAligner(
            SimpleNamespace(model_path=Path("models/qwen3"), device="cpu")
        )
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

        with patch("src.align.qwen3_forced_aligner.import_module", side_effect=_fake_import):
            with patch("src.align.qwen3_forced_aligner.version", return_value="9.9.9"):
                result = aligner.align(
                    "in.wav",
                    [
                        {"ref_id": "span_0001", "text": "you"},
                        {"ref_id": "span_0002", "text": "know"},
                    ],
                )

        self.assertEqual(result["meta"]["backend"], "qwen3-forced-aligner")
        self.assertEqual(result["meta"]["version"], "9.9.9")
        self.assertEqual(result["meta"]["device"], "cpu")
        self.assertEqual(result["transcript_text"], "you know")
        self.assertEqual(result["spans"][0]["span_id"], "span_0001")
        self.assertEqual(result["spans"][0]["start"], 10.0)
        self.assertEqual(result["spans"][0]["end"], 11.0)
        self.assertEqual(result["spans"][0]["confidence"], 0.9)


if __name__ == "__main__":
    unittest.main()
