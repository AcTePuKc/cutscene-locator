from pathlib import Path

import unittest
import unittest.mock

from src.asr import (
    ASRConfig,
    MockASRBackend,
    resolve_device,
    parse_asr_result,
    resolve_device_with_details,
    validate_asr_result,
)
from src.asr.timestamp_normalization import normalize_asr_segments_for_contract
from src.asr.device import _cuda_probe_ctranslate2, select_cuda_probe


class MockASRBackendTests(unittest.TestCase):
    def test_mock_backend_loads_valid_contract(self) -> None:
        result = MockASRBackend(Path("tests/fixtures/mock_asr_valid.json")).transcribe(
            "unused.wav",
            ASRConfig(backend_name="mock", device="cpu"),
        )

        self.assertEqual(result["meta"]["backend"], "mock")
        self.assertEqual(result["meta"]["model"], "unknown")
        self.assertEqual(result["meta"]["device"], "cpu")
        self.assertEqual(len(result["segments"]), 2)
        self.assertEqual(result["segments"][0]["segment_id"], "seg_0001")
        self.assertEqual(result["segments"][0]["start"], 12.34)
        self.assertEqual(result["segments"][0]["text"], "You don't get it.")

    def test_mock_backend_rejects_invalid_start_end(self) -> None:
        with self.assertRaisesRegex(ValueError, "start must be less than end"):
            MockASRBackend(Path("tests/fixtures/mock_asr_invalid_start_end.json")).transcribe(
                "unused.wav",
                ASRConfig(backend_name="mock"),
            )

    def test_validate_rejects_non_numeric_timestamps(self) -> None:
        with self.assertRaisesRegex(ValueError, r"segments\[0\]\.start must be numeric"):
            validate_asr_result(
                {
                    "segments": [
                        {
                            "segment_id": "seg_0001",
                            "start": "12.34",
                            "end": 13.0,
                            "text": "hello",
                        }
                    ],
                    "meta": {
                        "backend": "mock",
                        "model": "unknown",
                        "version": "1.0",
                        "device": "cpu",
                    },
                },
                source="inline",
            )

    def test_parse_asr_result_returns_validated_contract(self) -> None:
        result = parse_asr_result(
            {
                "segments": [
                    {
                        "segment_id": "seg_0001",
                        "start": 0.0,
                        "end": 0.5,
                        "text": "hello",
                    }
                ],
                "meta": {
                    "backend": "mock",
                    "model": "unknown",
                    "version": "1.0",
                    "device": "cpu",
                },
            },
            source="inline",
        )

        self.assertIsInstance(result["segments"], list)
        self.assertEqual(result["segments"][0]["segment_id"], "seg_0001")

    def test_validate_rejects_empty_text(self) -> None:
        with self.assertRaisesRegex(ValueError, r"segments\[0\]\.text must be a non-empty string"):
            validate_asr_result(
                {
                    "segments": [
                        {
                            "segment_id": "seg_0001",
                            "start": 12.34,
                            "end": 13.0,
                            "text": "   ",
                        }
                    ],
                    "meta": {
                        "backend": "mock",
                        "model": "unknown",
                        "version": "1.0",
                        "device": "cpu",
                    },
                },
                source="inline",
            )


    def test_timestamp_normalization_handles_equal_segment_boundaries_and_overlap_order(self) -> None:
        normalized = normalize_asr_segments_for_contract(
            [
                {"segment_id": "seg_0002", "start": 2.0000004, "end": 3.0, "text": "second"},
                {"segment_id": "seg_0001", "start": 1.0, "end": 2.0000004, "text": "adjacent"},
                {"segment_id": "seg_0003", "start": 1.0000004, "end": 1.5000004, "text": "overlap"},
            ],
            source="inline",
        )

        self.assertEqual(len(normalized), 3)
        self.assertEqual([segment["segment_id"] for segment in normalized], ["seg_0003", "seg_0001", "seg_0002"])
        self.assertEqual(normalized[1]["end"], normalized[2]["start"])

    def test_timestamp_normalization_rejects_end_before_start(self) -> None:
        with self.assertRaisesRegex(ValueError, "end must be greater than or equal to start"):
            normalize_asr_segments_for_contract(
                [{"segment_id": "seg_0001", "start": 1.0, "end": 0.9, "text": "bad"}],
                source="inline",
            )

    def test_timestamp_normalization_drops_zero_length_segment(self) -> None:
        normalized = normalize_asr_segments_for_contract(
            [{"segment_id": "seg_0001", "start": 1.0, "end": 1.0, "text": "drop"}],
            source="inline",
        )
        self.assertEqual(normalized, [])

    def test_timestamp_normalization_supports_empty_segments(self) -> None:
        normalized = normalize_asr_segments_for_contract([], source="inline")
        self.assertEqual(normalized, [])

    def test_timestamp_normalization_rejects_pathological_timings(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be non-negative"):
            normalize_asr_segments_for_contract(
                [{"segment_id": "seg_0001", "start": -0.1, "end": 0.3, "text": "bad"}],
                source="inline",
            )


class CTranslate2CudaProbeTests(unittest.TestCase):
    def test_cuda_probe_accepts_numeric_return(self) -> None:
        class _CTranslate2Module:
            @staticmethod
            def get_cuda_device_count() -> int:
                return 2

        with unittest.mock.patch("src.asr.device.import_module", return_value=_CTranslate2Module):
            available, reason = _cuda_probe_ctranslate2()

        self.assertTrue(available)
        self.assertEqual(reason, "ctranslate2 detected 2 CUDA device(s)")

    def test_cuda_probe_accepts_numeric_string_return(self) -> None:
        class _CTranslate2Module:
            @staticmethod
            def get_cuda_device_count() -> str:
                return "3"

        with unittest.mock.patch("src.asr.device.import_module", return_value=_CTranslate2Module):
            available, reason = _cuda_probe_ctranslate2()

        self.assertTrue(available)
        self.assertEqual(reason, "ctranslate2 detected 3 CUDA device(s)")

    def test_cuda_probe_handles_non_convertible_return_deterministically(self) -> None:
        class _CTranslate2Module:
            @staticmethod
            def get_cuda_device_count() -> object:
                return object()

        with unittest.mock.patch("src.asr.device.import_module", return_value=_CTranslate2Module):
            available, reason = _cuda_probe_ctranslate2()

        self.assertFalse(available)
        self.assertIn("returned a non-numeric device count", reason)

    def test_cuda_probe_preserves_getter_exception_path(self) -> None:
        class _CTranslate2Module:
            @staticmethod
            def get_cuda_device_count() -> int:
                raise RuntimeError("probe failed")

        with unittest.mock.patch("src.asr.device.import_module", return_value=_CTranslate2Module):
            available, reason = _cuda_probe_ctranslate2()

        self.assertFalse(available)
        self.assertEqual(reason, "ctranslate2 CUDA check failed: probe failed")




class CudaProbeSelectionTests(unittest.TestCase):
    def test_select_cuda_probe_routes_torch_backends(self) -> None:
        for backend in ("qwen3-asr", "whisperx", "vibevoice"):
            checker, label = select_cuda_probe(backend)
            self.assertEqual(label, "torch")
            with unittest.mock.patch("src.asr.device.import_module", side_effect=ModuleNotFoundError()):
                self.assertFalse(checker())

    def test_select_cuda_probe_routes_faster_whisper_to_ctranslate2(self) -> None:
        checker, label = select_cuda_probe("faster-whisper")
        self.assertEqual(label, "ctranslate2")
        with unittest.mock.patch("src.asr.device.import_module", side_effect=ModuleNotFoundError()):
            self.assertFalse(checker())

    def test_qwen_cuda_uses_torch_probe_when_ctranslate2_missing(self) -> None:
        fake_torch = type("Torch", (), {"cuda": type("Cuda", (), {"is_available": staticmethod(lambda: True)})()})

        def _fake_import(name: str):
            if name == "ctranslate2":
                raise ModuleNotFoundError(name)
            if name == "torch":
                return fake_torch
            raise ModuleNotFoundError(name)

        with unittest.mock.patch("src.asr.device.import_module", side_effect=_fake_import):
            checker, label = select_cuda_probe("qwen3-asr")
            resolution = resolve_device_with_details(
                "cuda",
                cuda_available_checker=checker,
                cuda_probe_reason_label=label,
            )

        self.assertEqual(resolution.resolved, "cuda")
        self.assertIn("torch CUDA probe reported available", resolution.reason)

    def test_faster_whisper_cuda_still_uses_ctranslate2_probe(self) -> None:
        def _fake_import(name: str):
            raise ModuleNotFoundError(name)

        with unittest.mock.patch("src.asr.device.import_module", side_effect=_fake_import):
            checker, label = select_cuda_probe("faster-whisper")
            with self.assertRaisesRegex(ValueError, "Reason: ctranslate2 CUDA probe reported unavailable"):
                resolve_device_with_details(
                    "cuda",
                    cuda_available_checker=checker,
                    cuda_probe_reason_label=label,
                )



class DeviceResolutionTests(unittest.TestCase):
    def test_auto_prefers_cuda_when_available(self) -> None:
        resolved = resolve_device("auto", cuda_available_checker=lambda: True)
        self.assertEqual(resolved, "cuda")

    def test_auto_falls_back_to_cpu_when_cuda_unavailable(self) -> None:
        resolved = resolve_device("auto", cuda_available_checker=lambda: False)
        self.assertEqual(resolved, "cpu")

    def test_cuda_request_fails_when_unavailable(self) -> None:
        with self.assertRaisesRegex(ValueError, "Requested --device cuda"):
            resolve_device("cuda", cuda_available_checker=lambda: False)


    def test_auto_resolution_includes_reason(self) -> None:
        resolution = resolve_device_with_details("auto", cuda_available_checker=lambda: False)
        self.assertEqual(resolution.resolved, "cpu")
        self.assertIn("selected cpu", resolution.reason)

    def test_cuda_unavailable_error_mentions_cuda_doc(self) -> None:
        with self.assertRaisesRegex(ValueError, "docs/CUDA.md"):
            resolve_device_with_details("cuda", cuda_available_checker=lambda: False)

    def test_cuda_unavailable_error_includes_probe_reason_label(self) -> None:
        with self.assertRaisesRegex(ValueError, "Reason: torch CUDA probe reported unavailable"):
            resolve_device_with_details(
                "cuda",
                cuda_available_checker=lambda: False,
                cuda_probe_reason_label="torch",
            )

    def test_mock_backend_uses_config_model_and_resolved_device_meta(self) -> None:
        result = MockASRBackend(Path("tests/fixtures/mock_asr_valid.json")).transcribe(
            "unused.wav",
            ASRConfig(backend_name="mock", device="auto", model_path=Path("models/my-model.bin")),
        )

        self.assertEqual(result["meta"]["model"], "my-model.bin")
        self.assertEqual(result["meta"]["device"], "cpu")


if __name__ == "__main__":
    unittest.main()
