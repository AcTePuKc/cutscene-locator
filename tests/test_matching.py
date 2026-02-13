from pathlib import Path

import unittest
from unittest.mock import Mock, patch

from src.asr.base import ASRResult
from src.ingest.script_parser import ScriptRow, ScriptTable, load_script_table
from src.match.engine import MatchingConfig, _similarity_score, match_segments_to_script


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


    def test_similarity_penalizes_prefix_only_overlap(self) -> None:
        prefix_score = _similarity_score("hello there general kenobi", "hello")
        fuller_score = _similarity_score(
            "hello there general kenobi",
            "hello there general kenobi indeed",
        )

        self.assertLess(prefix_score, fuller_score)

    def test_matching_prefers_fuller_sentence_over_first_word_only(self) -> None:
        script_table = ScriptTable(
            delimiter="\t",
            columns=["id", "original"],
            rows=[
                ScriptRow(
                    row_number=2,
                    values={"id": "S01", "original": "hello"},
                    normalized_original="hello",
                ),
                ScriptRow(
                    row_number=3,
                    values={"id": "S02", "original": "hello there general kenobi indeed"},
                    normalized_original="hello there general kenobi indeed",
                ),
            ],
        )
        asr_result: ASRResult = {
            "segments": [
                {
                    "segment_id": "seg_0001",
                    "start": 1.0,
                    "end": 2.0,
                    "text": "hello there general kenobi",
                }
            ],
            "meta": {
                "backend": "mock",
                "model": "unknown",
                "version": "1.0",
                "device": "cpu",
            },
        }

        output = match_segments_to_script(asr_result=asr_result, script_table=script_table)

        self.assertEqual(output.matches[0].matched_id, "S02")


    def test_quick_filter_rejects_first_word_only_overlap(self) -> None:
        script_table = ScriptTable(
            delimiter="	",
            columns=["id", "original"],
            rows=[
                ScriptRow(
                    row_number=2,
                    values={"id": "S01", "original": "hello"},
                    normalized_original="hello",
                ),
                ScriptRow(
                    row_number=3,
                    values={"id": "S02", "original": "hello there general kenobi indeed"},
                    normalized_original="hello there general kenobi indeed",
                ),
            ],
        )
        asr_result: ASRResult = {
            "segments": [
                {
                    "segment_id": "seg_0001",
                    "start": 0.0,
                    "end": 1.0,
                    "text": "hello there general kenobi",
                }
            ],
            "meta": {"backend": "mock", "model": "unknown", "version": "1.0", "device": "cpu"},
        }
        progress_logger = Mock()

        with patch("src.match.engine._similarity_score", wraps=_similarity_score) as similarity_mock:
            output = match_segments_to_script(
                asr_result=asr_result,
                script_table=script_table,
                progress_logger=progress_logger,
            )

        self.assertEqual(output.matches[0].matched_id, "S02")
        self.assertEqual(similarity_mock.call_count, 1)
        self.assertFalse(
            any(
                "quick-filter rejected all" in call.args[0]
                for call in progress_logger.call_args_list
            )
        )

    def test_quick_filter_keeps_legitimate_short_line_match(self) -> None:
        script_table = ScriptTable(
            delimiter="	",
            columns=["id", "original"],
            rows=[
                ScriptRow(
                    row_number=2,
                    values={"id": "S01", "original": "on me"},
                    normalized_original="on me",
                ),
                ScriptRow(
                    row_number=3,
                    values={"id": "S02", "original": "off target"},
                    normalized_original="off target",
                ),
            ],
        )
        asr_result: ASRResult = {
            "segments": [
                {
                    "segment_id": "seg_0001",
                    "start": 0.0,
                    "end": 1.0,
                    "text": "on me",
                }
            ],
            "meta": {"backend": "mock", "model": "unknown", "version": "1.0", "device": "cpu"},
        }

        output = match_segments_to_script(
            asr_result=asr_result,
            script_table=script_table,
        )

        self.assertEqual(output.matches[0].matched_id, "S01")

    def test_uses_rapidfuzz_wratio_for_similarity(self) -> None:
        script_table = load_script_table(Path("tests/fixtures/script_sample.tsv"))
        asr_result: ASRResult = {
            "segments": [
                {
                    "segment_id": "seg_0001",
                    "start": 1.0,
                    "end": 2.0,
                    "text": "hello world there",
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
        self.assertTrue(0.0 <= output.matches[0].score <= 1.0)

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

        self.assertEqual(progress_logger.call_args_list[-1].args[0], "Verbose: matching progress 2/2 segments")

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
