from pathlib import Path

import unittest

from src.asr.backends import ASRResult
from src.ingest.script_parser import load_script_table
from src.match.engine import match_segments_to_script


class MatchingEngineTests(unittest.TestCase):
    def test_matches_each_segment_and_keeps_best_score(self) -> None:
        script_table = load_script_table(Path("tests/fixtures/script_sample.tsv"))
        asr_result: ASRResult = {
            "segments": [
                {
                    "segment_id": "seg_0001",
                    "start": 1.0,
                    "end": 2.0,
                    "text": "hello world",
                },
                {
                    "segment_id": "seg_0002",
                    "start": 2.5,
                    "end": 3.5,
                    "text": "zzzzz",
                },
            ],
            "meta": {
                "backend": "mock",
                "version": "1.0",
                "language": "en",
            },
        }

        output = match_segments_to_script(asr_result=asr_result, script_table=script_table)

        self.assertEqual(len(output.matches), 2)

        first = output.matches[0]
        self.assertEqual(first.segment_id, "seg_0001")
        self.assertEqual(first.matched_id, "M01_001")
        self.assertEqual(first.matched_text, "  Hello,   WORLD! ")
        self.assertEqual(first.score, 1.0)
        self.assertFalse(first.low_confidence)

        second = output.matches[1]
        self.assertEqual(second.segment_id, "seg_0002")
        self.assertTrue(0.0 <= second.score <= 1.0)
        self.assertTrue(second.low_confidence)

    def test_threshold_is_configurable(self) -> None:
        script_table = load_script_table(Path("tests/fixtures/script_sample.tsv"))
        asr_result: ASRResult = {
            "segments": [
                {
                    "segment_id": "seg_0001",
                    "start": 1.0,
                    "end": 2.0,
                    "text": "hello world",
                }
            ],
            "meta": {
                "backend": "mock",
                "version": "1.0",
                "language": "en",
            },
        }

        output = match_segments_to_script(
            asr_result=asr_result,
            script_table=script_table,
            low_confidence_threshold=1.0,
        )

        self.assertFalse(output.matches[0].low_confidence)

    def test_invalid_threshold_raises(self) -> None:
        script_table = load_script_table(Path("tests/fixtures/script_sample.tsv"))
        asr_result: ASRResult = {
            "segments": [
                {
                    "segment_id": "seg_0001",
                    "start": 1.0,
                    "end": 2.0,
                    "text": "hello world",
                }
            ],
            "meta": {
                "backend": "mock",
                "version": "1.0",
                "language": "en",
            },
        }

        with self.assertRaisesRegex(ValueError, "low_confidence_threshold"):
            match_segments_to_script(
                asr_result=asr_result,
                script_table=script_table,
                low_confidence_threshold=1.1,
            )


if __name__ == "__main__":
    unittest.main()
