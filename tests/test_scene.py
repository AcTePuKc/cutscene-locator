from __future__ import annotations

import unittest

from src.match.engine import MatchResult, MatchingOutput
from src.scene import reconstruct_scenes


class SceneReconstructionTests(unittest.TestCase):
    def test_chronological_sorting_and_stable_scene_ids(self) -> None:
        matching_output = MatchingOutput(
            matches=[
                MatchResult(
                    segment_id="seg_0003",
                    start_time=25.0,
                    end_time=26.0,
                    asr_text="c",
                    matched_id="M03",
                    matched_text="C",
                    score=1.0,
                    low_confidence=False,
                ),
                MatchResult(
                    segment_id="seg_0001",
                    start_time=1.0,
                    end_time=2.0,
                    asr_text="a",
                    matched_id="M01",
                    matched_text="A",
                    score=1.0,
                    low_confidence=False,
                ),
                MatchResult(
                    segment_id="seg_0002",
                    start_time=5.0,
                    end_time=6.0,
                    asr_text="b",
                    matched_id="M02",
                    matched_text="B",
                    score=1.0,
                    low_confidence=False,
                ),
            ]
        )

        scene_output = reconstruct_scenes(matching_output=matching_output, scene_gap_seconds=10.0)

        self.assertEqual(len(scene_output["scenes"]), 2)
        self.assertEqual(scene_output["scenes"][0]["scene_id"], "scene_001")
        self.assertEqual(scene_output["scenes"][1]["scene_id"], "scene_002")
        self.assertEqual(
            [line["segment_id"] for line in scene_output["scenes"][0]["lines"]],
            ["seg_0001", "seg_0002"],
        )
        self.assertEqual(
            [line["line_id"] for line in scene_output["scenes"][0]["lines"]],
            ["M01", "M02"],
        )

    def test_overlap_does_not_split_scene(self) -> None:
        matching_output = MatchingOutput(
            matches=[
                MatchResult(
                    segment_id="seg_0001",
                    start_time=10.0,
                    end_time=15.0,
                    asr_text="line one",
                    matched_id="M01",
                    matched_text="Line one",
                    score=1.0,
                    low_confidence=False,
                ),
                MatchResult(
                    segment_id="seg_0002",
                    start_time=14.0,
                    end_time=18.0,
                    asr_text="line two",
                    matched_id="M02",
                    matched_text="Line two",
                    score=1.0,
                    low_confidence=False,
                ),
            ]
        )

        scene_output = reconstruct_scenes(matching_output=matching_output, scene_gap_seconds=10.0)

        self.assertEqual(len(scene_output["scenes"]), 1)
        scene = scene_output["scenes"][0]
        self.assertEqual(scene["start_time"], 10.0)
        self.assertEqual(scene["end_time"], 18.0)
        self.assertEqual(
            [line["segment_id"] for line in scene["lines"]],
            ["seg_0001", "seg_0002"],
        )


if __name__ == "__main__":
    unittest.main()
