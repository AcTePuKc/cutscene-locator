"""Deterministic fuzzy matching engine for ASR segments and script lines."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher

from src.asr.backends import ASRResult
from src.ingest.script_parser import ScriptTable
from src.match.normalization import normalize_text


@dataclass(frozen=True)
class MatchResult:
    """Best-match result for one ASR segment."""

    segment_id: str
    start_time: float
    end_time: float
    asr_text: str
    matched_id: str
    matched_text: str
    score: float
    low_confidence: bool


@dataclass(frozen=True)
class MatchingOutput:
    """In-memory matching output for downstream scene reconstruction/export."""

    matches: list[MatchResult]


def _similarity_score(left: str, right: str) -> float:
    """Return deterministic similarity score in [0.0, 1.0]."""

    return SequenceMatcher(None, left, right, autojunk=False).ratio()


def match_segments_to_script(
    *,
    asr_result: ASRResult,
    script_table: ScriptTable,
    low_confidence_threshold: float = 0.85,
) -> MatchingOutput:
    """Match each ASR segment against all script lines and keep best candidate."""

    if not 0.0 <= low_confidence_threshold <= 1.0:
        raise ValueError("low_confidence_threshold must be in [0.0, 1.0]")
    if not script_table.rows:
        raise ValueError("Script table must contain at least one row for matching")

    matches: list[MatchResult] = []
    for segment in asr_result["segments"]:
        normalized_asr = normalize_text(segment["text"])
        scored_candidates: list[tuple[float, str, int, str]] = []
        for row in script_table.rows:
            line_id = row.values["id"]
            score = _similarity_score(normalized_asr, row.normalized_original)
            scored_candidates.append((score, line_id, row.row_number, row.values["original"]))

        best_score, best_line_id, _, best_text = max(
            scored_candidates,
            key=lambda item: (item[0], item[1], -item[2]),
        )
        score = round(best_score, 6)

        matches.append(
            MatchResult(
                segment_id=segment["segment_id"],
                start_time=float(segment["start"]),
                end_time=float(segment["end"]),
                asr_text=segment["text"],
                matched_id=best_line_id,
                matched_text=best_text,
                score=score,
                low_confidence=score < low_confidence_threshold,
            )
        )

    return MatchingOutput(matches=matches)
