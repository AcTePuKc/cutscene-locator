from __future__ import annotations

import csv
import json
import os
import tempfile
import unittest
from pathlib import Path

from src.asr.backends import parse_asr_result
from src.export import write_matches_csv, write_scenes_json
from src.ingest.script_parser import ScriptTable, load_script_table
from src.match.engine import match_segments_to_script
from src.scene import reconstruct_scenes


@unittest.skipUnless(
    os.environ.get("CUTSCENE_RUN_INTEGRATION") == "1",
    "Set CUTSCENE_RUN_INTEGRATION=1 to run offline integration-style deterministic pipeline checks.",
)
class DeterministicPipelineIntegrationTests(unittest.TestCase):
    SCRIPT_FIXTURE = Path("tests/fixtures/script_integration_sample.tsv")
    ASR_FIXTURES = [
        Path("tests/fixtures/asr_normalized_faster_whisper.json"),
        Path("tests/fixtures/asr_normalized_qwen3_asr.json"),
        Path("tests/fixtures/asr_normalized_whisperx_vibevoice.json"),
    ]

    def test_offline_pipeline_outputs_are_stable_and_timestamp_safe(self) -> None:
        script_table = load_script_table(self.SCRIPT_FIXTURE)
        for fixture in self.ASR_FIXTURES:
            first_run = self._run_pipeline(script_table=script_table, asr_fixture=fixture)
            second_run = self._run_pipeline(script_table=script_table, asr_fixture=fixture)

            self.assertEqual(first_run["matches_rows"], second_run["matches_rows"])
            self.assertEqual(first_run["scenes_payload"], second_run["scenes_payload"])

            self.assertEqual(
                [row["segment_id"] for row in first_run["matches_rows"]],
                first_run["input_segment_ids"],
            )
            self.assertEqual(first_run["scene_boundaries"], [(0.0, 2.2), (15.0, 17.0)])
            self._assert_no_fabricated_timestamps(first_run)

    def _run_pipeline(self, *, script_table: ScriptTable, asr_fixture: Path) -> dict[str, object]:
        raw_data = json.loads(asr_fixture.read_text(encoding="utf-8"))
        asr_result = parse_asr_result(raw_data, source=str(asr_fixture))

        matching_output = match_segments_to_script(asr_result=asr_result, script_table=script_table)
        scene_output = reconstruct_scenes(matching_output=matching_output)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            matches_path = temp_path / "matches.csv"
            scenes_path = temp_path / "scenes.json"
            write_matches_csv(output_path=matches_path, matching_output=matching_output)
            write_scenes_json(output_path=scenes_path, scene_output=scene_output)

            with matches_path.open("r", encoding="utf-8", newline="") as handle:
                matches_rows = list(csv.DictReader(handle))
            scenes_payload = json.loads(scenes_path.read_text(encoding="utf-8"))

        return {
            "input_segment_ids": [segment["segment_id"] for segment in asr_result["segments"]],
            "input_timestamps": {
                segment["segment_id"]: (float(segment["start"]), float(segment["end"]))
                for segment in asr_result["segments"]
            },
            "matches_rows": matches_rows,
            "scenes_payload": scenes_payload,
            "scene_boundaries": [
                (scene["start_time"], scene["end_time"])
                for scene in scenes_payload["scenes"]
            ],
        }

    def _assert_no_fabricated_timestamps(self, run_output: dict[str, object]) -> None:
        input_timestamps = run_output["input_timestamps"]
        assert isinstance(input_timestamps, dict)

        for row in run_output["matches_rows"]:
            assert isinstance(row, dict)
            segment_id = row["segment_id"]
            expected_start, expected_end = input_timestamps[segment_id]
            self.assertEqual(float(row["start_time"]), expected_start)
            self.assertEqual(float(row["end_time"]), expected_end)

        scenes_payload = run_output["scenes_payload"]
        assert isinstance(scenes_payload, dict)
        for scene in scenes_payload["scenes"]:
            line_times = [input_timestamps[line["segment_id"]] for line in scene["lines"]]
            self.assertEqual(scene["start_time"], min(start for start, _ in line_times))
            self.assertEqual(scene["end_time"], max(end for _, end in line_times))


if __name__ == "__main__":
    unittest.main()
