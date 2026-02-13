import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.align import Qwen3ForcedAligner
from src.asr import ASRConfig
from src.asr.qwen3_asr_backend import Qwen3ASRBackend


class _TorchModuleWithTo:
    def __init__(self) -> None:
        self.to_calls: list[str] = []

    def to(self, device: str) -> None:
        self.to_calls.append(device)


class _FakeQwen3ASRModel:
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs: object):
        del model_path
        del kwargs

        class _FakeModel:
            def transcribe(self, audio_path: str, **transcribe_kwargs: object):
                del audio_path
                del transcribe_kwargs
                return {
                    "chunks": [
                        {"timestamp": (0.0, 1.0), "text": " hello "},
                        {"timestamp": (1.1, 2.0), "text": "world"},
                    ]
                }

        model = _FakeModel()
        model.model = _TorchModuleWithTo()
        return model


class _FakeAlignmentPipeline:
    def __call__(self, audio_path: str, return_timestamps: str):
        del audio_path
        del return_timestamps
        return {
            "chunks": [
                {"timestamp": (10.0, 11.0), "text": "you", "confidence": 0.9},
                {"timestamp": (11.1, 12.0), "text": "know", "confidence": 0.8},
            ]
        }


class BackendContractTests(unittest.TestCase):
    def test_asr_and_alignment_contracts_share_required_schema_guards(self) -> None:
        cases: list[dict[str, object]] = [
            {
                "backend": "qwen3-asr",
                "kind": "asr",
                "result": self._run_qwen3_asr_case(),
                "expected_segment_keys": {"segment_id", "start", "end", "text"},
            },
            {
                "backend": "qwen3-forced-aligner",
                "kind": "alignment",
                "result": self._run_qwen3_forced_aligner_case(),
                "expected_span_keys": {"span_id", "start", "end", "text", "confidence"},
            },
        ]

        for case in cases:
            backend = str(case["backend"])
            result = case["result"]
            self.assertIsInstance(result, dict, msg=f"[{backend}] result must be a mapping")
            self.assertIn("meta", result, msg=f"[{backend}] missing meta")
            meta = result["meta"]
            self.assertIsInstance(meta, dict, msg=f"[{backend}] meta must be an object")
            for required_key in ("backend", "version", "device"):
                self.assertIn(required_key, meta, msg=f"[{backend}] missing meta.{required_key}")

            if case["kind"] == "asr":
                self.assertIn("segments", result, msg=f"[{backend}] missing segments")
                segments = result["segments"]
                self.assertIsInstance(segments, list, msg=f"[{backend}] segments must be a list")
                self.assertGreater(len(segments), 0, msg=f"[{backend}] segments must be non-empty")
                for index, segment in enumerate(segments):
                    self.assertIsInstance(segment, dict, msg=f"[{backend}] segment[{index}] must be an object")
                    self.assertTrue(
                        set(case["expected_segment_keys"]).issubset(segment.keys()),
                        msg=f"[{backend}] segment[{index}] missing required keys",
                    )
                    self.assertLessEqual(
                        float(segment["start"]),
                        float(segment["end"]),
                        msg=f"[{backend}] segment[{index}] has start > end",
                    )
            else:
                self.assertIn("transcript_text", result, msg=f"[{backend}] missing transcript_text")
                self.assertIn("spans", result, msg=f"[{backend}] missing spans")
                spans = result["spans"]
                self.assertIsInstance(spans, list, msg=f"[{backend}] spans must be a list")
                self.assertGreater(len(spans), 0, msg=f"[{backend}] spans must be non-empty")
                for index, span in enumerate(spans):
                    self.assertIsInstance(span, dict, msg=f"[{backend}] span[{index}] must be an object")
                    self.assertTrue(
                        set(case["expected_span_keys"]).issubset(span.keys()),
                        msg=f"[{backend}] span[{index}] missing required keys",
                    )
                    self.assertLessEqual(
                        float(span["start"]),
                        float(span["end"]),
                        msg=f"[{backend}] span[{index}] has start > end",
                    )

    def _run_qwen3_asr_case(self) -> dict[str, object]:
        backend = Qwen3ASRBackend()
        fake_qwen_asr = types.SimpleNamespace(Qwen3ASRModel=_FakeQwen3ASRModel)
        with patch("src.asr.qwen3_asr_backend.import_module", return_value=fake_qwen_asr):
            with patch("src.asr.qwen3_asr_backend.version", return_value="9.9.9"):
                return backend.transcribe(
                    "in.wav",
                    ASRConfig(backend_name="qwen3-asr", model_path=Path("models/qwen3"), device="cpu"),
                )

    def _run_qwen3_forced_aligner_case(self) -> dict[str, object]:
        aligner = Qwen3ForcedAligner(SimpleNamespace(model_path=Path("models/qwen3"), device="cpu"))
        fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
        fake_transformers = types.SimpleNamespace(pipeline=lambda **kwargs: _FakeAlignmentPipeline())

        def _fake_import(name: str):
            if name == "torch":
                return fake_torch
            if name == "transformers":
                return fake_transformers
            raise ModuleNotFoundError(name)

        with patch("src.align.qwen3_forced_aligner.import_module", side_effect=_fake_import):
            with patch("src.align.qwen3_forced_aligner.version", return_value="9.9.9"):
                return aligner.align(
                    "in.wav",
                    [
                        {"ref_id": "span_0001", "text": "you"},
                        {"ref_id": "span_0002", "text": "know"},
                    ],
                )


if __name__ == "__main__":
    unittest.main()
