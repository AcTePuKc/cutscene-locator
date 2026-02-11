from pathlib import Path

import unittest

from src.asr import ASRConfig, MockASRBackend, resolve_device, validate_asr_result


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

    def test_mock_backend_uses_config_model_and_resolved_device_meta(self) -> None:
        result = MockASRBackend(Path("tests/fixtures/mock_asr_valid.json")).transcribe(
            "unused.wav",
            ASRConfig(backend_name="mock", device="auto", model_path=Path("models/my-model.bin")),
        )

        self.assertEqual(result["meta"]["model"], "my-model.bin")
        self.assertEqual(result["meta"]["device"], "cpu")


if __name__ == "__main__":
    unittest.main()
