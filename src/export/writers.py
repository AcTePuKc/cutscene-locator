"""Export writers for matches/scenes/subtitles."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from src.ingest.script_parser import ScriptTable
from src.match.engine import MatchingOutput
from src.scene import SceneReconstruction


def write_matches_csv(*, output_path: Path, matching_output: MatchingOutput) -> None:
    """Write one CSV row per ASR segment with contract-defined columns."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "segment_id",
                "start_time",
                "end_time",
                "asr_text",
                "matched_id",
                "matched_text",
                "score",
            ]
        )
        for match in matching_output.matches:
            writer.writerow(
                [
                    match.segment_id,
                    _format_float(match.start_time),
                    _format_float(match.end_time),
                    match.asr_text,
                    match.matched_id,
                    match.matched_text,
                    _format_float(match.score),
                ]
            )


def write_scenes_json(*, output_path: Path, scene_output: SceneReconstruction) -> None:
    """Write deterministic scene JSON structure."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered_scenes = sorted(scene_output["scenes"], key=lambda scene: scene["start_time"])
    payload = {
        "scenes": [
            {
                "scene_id": scene["scene_id"],
                "start_time": scene["start_time"],
                "end_time": scene["end_time"],
                "lines": [
                    {
                        "segment_id": line["segment_id"],
                        "line_id": line["line_id"],
                    }
                    for line in scene["lines"]
                ],
            }
            for scene in ordered_scenes
        ]
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_subs_qa_srt(
    *,
    output_path: Path,
    matching_output: MatchingOutput,
    script_table: ScriptTable,
) -> None:
    """Write QA SRT with [ORIG] and optional [TR] lines."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    script_index = _build_script_index(script_table)
    has_translation_column = "translation" in script_table.columns
    entries: list[str] = []

    for idx, match in enumerate(matching_output.matches, start=1):
        row_values = script_index.get(match.matched_id, {})
        orig_text = row_values.get("original", match.matched_text)

        block = [
            str(idx),
            f"{_format_srt_timestamp(match.start_time)} --> {_format_srt_timestamp(match.end_time)}",
            f"[ORIG] {orig_text}",
        ]

        if has_translation_column:
            translation = row_values.get("translation", "")
            if translation:
                block.append(f"[TR] {translation}")

        entries.append("\n".join(block))

    output_path.write_text("\n\n".join(entries) + "\n", encoding="utf-8")


def write_subs_target_srt(
    *,
    output_path: Path,
    matching_output: MatchingOutput,
    script_table: ScriptTable,
) -> None:
    """Write target-language-only SRT (translation column required)."""

    if "translation" not in script_table.columns:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    script_index = _build_script_index(script_table)
    entries: list[str] = []

    for idx, match in enumerate(matching_output.matches, start=1):
        row_values = script_index.get(match.matched_id, {})
        translation = row_values.get("translation", "")
        block = [
            str(idx),
            f"{_format_srt_timestamp(match.start_time)} --> {_format_srt_timestamp(match.end_time)}",
            translation,
        ]
        entries.append("\n".join(block))

    output_path.write_text("\n\n".join(entries) + "\n", encoding="utf-8")


def _build_script_index(script_table: ScriptTable) -> dict[str, dict[str, str]]:
    return {row.values["id"]: row.values for row in script_table.rows}


def _format_float(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _format_srt_timestamp(seconds: float) -> str:
    total_ms = int(round(seconds * 1000.0))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    sec, millis = divmod(rem, 1_000)
    return f"{hours:02d}:{minutes:02d}:{sec:02d},{millis:03d}"
