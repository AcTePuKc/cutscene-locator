"""Deterministic scene reconstruction from matched ASR segments."""

from __future__ import annotations

from typing import Any, TypedDict

from src.match.engine import MatchResult, MatchingOutput


class SceneLine(TypedDict):
    """Line mapping inside a reconstructed scene."""

    segment_id: str
    line_id: str


class Scene(TypedDict):
    """Single scene entry conforming to docs/Data-contracts.md."""

    scene_id: str
    start_time: float
    end_time: float
    lines: list[SceneLine]


class SceneReconstruction(TypedDict):
    """In-memory scene reconstruction contract."""

    scenes: list[Scene]


def _sorted_matches(matches: list[MatchResult]) -> list[MatchResult]:
    """Sort matches by segment start time, preserving stable input order for ties."""

    return sorted(matches, key=lambda match: match.start_time)


def reconstruct_scenes(
    *,
    matching_output: MatchingOutput,
    scene_gap_seconds: float = 10.0,
) -> SceneReconstruction:
    """Reconstruct scenes from matches using deterministic chronological gap splitting."""

    if scene_gap_seconds < 0:
        raise ValueError("scene_gap_seconds must be >= 0")

    ordered_matches = _sorted_matches(matching_output.matches)
    scenes: list[Scene] = []

    for match in ordered_matches:
        if not scenes:
            scenes.append(
                {
                    "scene_id": "scene_001",
                    "start_time": match.start_time,
                    "end_time": match.end_time,
                    "lines": [
                        {
                            "segment_id": match.segment_id,
                            "line_id": match.matched_id,
                        }
                    ],
                }
            )
            continue

        current_scene = scenes[-1]
        gap = match.start_time - current_scene["end_time"]

        if gap > scene_gap_seconds:
            scene_index = len(scenes) + 1
            scenes.append(
                {
                    "scene_id": f"scene_{scene_index:03d}",
                    "start_time": match.start_time,
                    "end_time": match.end_time,
                    "lines": [
                        {
                            "segment_id": match.segment_id,
                            "line_id": match.matched_id,
                        }
                    ],
                }
            )
            continue

        current_scene["end_time"] = max(current_scene["end_time"], match.end_time)
        current_scene["lines"].append(
            {
                "segment_id": match.segment_id,
                "line_id": match.matched_id,
            }
        )

    return {"scenes": scenes}
