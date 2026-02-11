from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.asr.base import ASRResult
from src.export import (
    write_matches_csv,
    write_scenes_json,
    write_subs_qa_srt,
    write_subs_target_srt,
)
from src.ingest.script_parser import load_script_table
from src.match.engine import match_segments_to_script
from src.scene import reconstruct_scenes


class ExportWritersTests(unittest.TestCase):
    def test_matches_csv_uses_contract_columns(self) -> None:
        script_table = load_script_table(Path("tests/fixtures/script_sample.tsv"))
        asr_result: ASRResult = {
            "segments": [
                {
                    "segment_id": "seg_0001",
                    "start": 12.34,
                    "end": 15.56,
                    "text": "hello world",
                }
            ],
            "meta": {"backend": "mock", "model": "unknown", "version": "1.0", "device": "cpu"},
        }
        matching_output = match_segments_to_script(asr_result=asr_result, script_table=script_table)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "matches.csv"
            write_matches_csv(output_path=output_path, matching_output=matching_output)

            with output_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(
                rows[0].keys(),
                {
                    "segment_id",
                    "start_time",
                    "end_time",
                    "asr_text",
                    "matched_id",
                    "matched_text",
                    "score",
                },
            )
            self.assertEqual(rows[0]["segment_id"], "seg_0001")

    def test_scenes_json_is_deterministic_and_contract_shaped(self) -> None:
        script_table = load_script_table(Path("tests/fixtures/script_sample.tsv"))
        asr_result: ASRResult = {
            "segments": [
                {"segment_id": "seg_0002", "start": 16.1, "end": 16.8, "text": "yeah"},
                {"segment_id": "seg_0001", "start": 12.34, "end": 15.56, "text": "hello world"},
            ],
            "meta": {"backend": "mock", "model": "unknown", "version": "1.0", "device": "cpu"},
        }
        matching_output = match_segments_to_script(asr_result=asr_result, script_table=script_table)
        scene_output = reconstruct_scenes(matching_output=matching_output)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "scenes.json"
            write_scenes_json(output_path=output_path, scene_output=scene_output)
            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertIn("scenes", payload)
        self.assertEqual(payload["scenes"][0]["scene_id"], "scene_001")
        self.assertEqual(payload["scenes"][0]["lines"][0]["segment_id"], "seg_0001")

    def test_subtitle_writers_follow_translation_rules(self) -> None:
        script_table = load_script_table(Path("tests/fixtures/script_sample.tsv"))
        asr_result: ASRResult = {
            "segments": [
                {"segment_id": "seg_0001", "start": 12.34, "end": 15.56, "text": "hello world"},
                {"segment_id": "seg_0002", "start": 16.1, "end": 16.8, "text": "second line"},
            ],
            "meta": {"backend": "mock", "model": "unknown", "version": "1.0", "device": "cpu"},
        }
        matching_output = match_segments_to_script(asr_result=asr_result, script_table=script_table)

        with tempfile.TemporaryDirectory() as temp_dir:
            qa_path = Path(temp_dir) / "subs_qa.srt"
            target_path = Path(temp_dir) / "subs_target.srt"
            write_subs_qa_srt(
                output_path=qa_path,
                matching_output=matching_output,
                script_table=script_table,
            )
            write_subs_target_srt(
                output_path=target_path,
                matching_output=matching_output,
                script_table=script_table,
            )

            qa_text = qa_path.read_text(encoding="utf-8")
            target_text = target_path.read_text(encoding="utf-8")

        self.assertIn("[ORIG]", qa_text)
        self.assertIn("[TR] Здрасти", qa_text)
        self.assertEqual(qa_text.count("[TR]"), 1)
        self.assertIn("Здрасти", target_text)

    def test_target_srt_not_generated_without_translation_column(self) -> None:
        script_table = load_script_table(Path("tests/fixtures/script_sample.csv"))
        asr_result: ASRResult = {
            "segments": [
                {"segment_id": "seg_0001", "start": 1.0, "end": 2.0, "text": "hello world"}
            ],
            "meta": {"backend": "mock", "model": "unknown", "version": "1.0", "device": "cpu"},
        }
        matching_output = match_segments_to_script(asr_result=asr_result, script_table=script_table)

        with tempfile.TemporaryDirectory() as temp_dir:
            target_path = Path(temp_dir) / "subs_target.srt"
            write_subs_target_srt(
                output_path=target_path,
                matching_output=matching_output,
                script_table=script_table,
            )
            self.assertFalse(target_path.exists())


if __name__ == "__main__":
    unittest.main()
