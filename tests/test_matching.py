from pathlib import Path

import unittest
from unittest.mock import Mock, patch

from src.asr.base import ASRResult
from src.ingest.script_parser import load_script_table
from src.match.engine import MatchingConfig, match_segments_to_script


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
                "model": "unknown",
                "version": "1.0",
                "device": "cpu",
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


    def test_uses_rapidfuzz_wratio_for_similarity(self) -> None:
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
                "model": "unknown",
                "version": "1.0",
                "device": "cpu",
            },
        }

        with patch("src.match.engine.fuzz.WRatio", return_value=100.0) as wratio_mock:
            output = match_segments_to_script(asr_result=asr_result, script_table=script_table)

        self.assertGreater(wratio_mock.call_count, 0)
        self.assertEqual(output.matches[0].score, 1.0)

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
                "model": "unknown",
                "version": "1.0",
                "device": "cpu",
            },
        }

        output = match_segments_to_script(
            asr_result=asr_result,
            script_table=script_table,
            config=MatchingConfig(low_confidence_threshold=1.0),
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
                "model": "unknown",
                "version": "1.0",
                "device": "cpu",
            },
        }

        with self.assertRaisesRegex(ValueError, "low_confidence_threshold"):
            match_segments_to_script(
                asr_result=asr_result,
                script_table=script_table,
                config=MatchingConfig(low_confidence_threshold=1.1),
            )

    def test_invalid_progress_every_raises(self) -> None:
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
                "model": "unknown",
                "version": "1.0",
                "device": "cpu",
            },
        }

        with self.assertRaisesRegex(ValueError, "progress_every"):
            match_segments_to_script(
                asr_result=asr_result,
                script_table=script_table,
                config=MatchingConfig(progress_every=0),
            )

    def test_progress_logger_reports_final_progress(self) -> None:
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
                "model": "unknown",
                "version": "1.0",
                "device": "cpu",
            },
        }
        progress_logger = Mock()

        match_segments_to_script(
            asr_result=asr_result,
            script_table=script_table,
            config=MatchingConfig(progress_every=10),
            progress_logger=progress_logger,
        )

        progress_logger.assert_called_once_with("Verbose: matching progress 2/2 segments")

    def test_monotonic_window_biases_following_rows(self) -> None:
        script_table = load_script_table(Path("tests/fixtures/script_sample.tsv"))
        asr_result: ASRResult = {
            "segments": [
                {"segment_id": "seg_0001", "start": 0.0, "end": 1.0, "text": "hello world"},
                {"segment_id": "seg_0002", "start": 1.2, "end": 2.0, "text": "general kenobi"},
            ],
            "meta": {"backend": "mock", "model": "unknown", "version": "1.0", "device": "cpu"},
        }

        output = match_segments_to_script(
            asr_result=asr_result,
            script_table=script_table,
            config=MatchingConfig(monotonic_window=1),
        )

        self.assertEqual(output.matches[0].matched_id, "M01_001")
        self.assertIn(output.matches[1].matched_id, {"M01_002", "M01_003"})


if __name__ == "__main__":
    unittest.main()
