import unittest

from src.align import validate_alignment_result


class AlignmentValidationTests(unittest.TestCase):
    def test_validate_alignment_result_accepts_valid_contract(self) -> None:
        result = validate_alignment_result(
            {
                "transcript_text": "You do not get it.",
                "spans": [
                    {
                        "span_id": "span_0001",
                        "start": 1.25,
                        "end": 2.75,
                        "text": "You do not get it.",
                        "confidence": 0.92,
                    }
                ],
                "meta": {
                    "backend": "forced-aligner-mock",
                    "version": "1.0.0",
                    "device": "cpu",
                },
            },
            source="test",
        )

        self.assertEqual(result["transcript_text"], "You do not get it.")
        self.assertEqual(result["spans"][0]["span_id"], "span_0001")
        self.assertEqual(result["spans"][0]["confidence"], 0.92)

    def test_validate_alignment_result_rejects_invalid_confidence(self) -> None:
        with self.assertRaisesRegex(ValueError, r"confidence must be within \[0, 1\]"):
            validate_alignment_result(
                {
                    "transcript_text": "Known transcript",
                    "spans": [
                        {
                            "span_id": "span_0001",
                            "start": 0.0,
                            "end": 1.0,
                            "text": "Known transcript",
                            "confidence": 1.5,
                        }
                    ],
                    "meta": {
                        "backend": "forced-aligner-mock",
                        "version": "1.0.0",
                        "device": "cpu",
                    },
                },
                source="test",
            )


if __name__ == "__main__":
    unittest.main()
