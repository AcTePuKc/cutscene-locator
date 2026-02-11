from pathlib import Path

import unittest

from src.asr import MockASRBackend, validate_asr_result


class MockASRBackendTests(unittest.TestCase):
    def test_mock_backend_loads_valid_contract(self) -> None:
        result = MockASRBackend(Path("tests/fixtures/mock_asr_valid.json")).run()

        self.assertEqual(result["meta"]["backend"], "mock")
        self.assertEqual(len(result["segments"]), 2)
        self.assertEqual(result["segments"][0]["segment_id"], "seg_0001")
        self.assertEqual(result["segments"][0]["start"], 12.34)
        self.assertEqual(result["segments"][0]["text"], "You don't get it.")

    def test_mock_backend_rejects_invalid_start_end(self) -> None:
        with self.assertRaisesRegex(ValueError, "start must be less than end"):
            MockASRBackend(Path("tests/fixtures/mock_asr_invalid_start_end.json")).run()

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
                        "version": "1.0",
                        "language": "en",
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
                        "version": "1.0",
                        "language": "en",
                    },
                },
                source="inline",
            )


if __name__ == "__main__":
    unittest.main()
