"""Deterministic fuzzy matching engine for ASR segments and script lines."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Callable

from src.asr.base import ASRResult
from src.ingest.script_parser import ScriptRow, ScriptTable
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


@dataclass(frozen=True)
class MatchingConfig:
    """Deterministic knobs for candidate reduction and scoring."""

    low_confidence_threshold: float = 0.85
    quick_filter_threshold: float = 0.25
    length_bucket_size: int = 4
    max_length_bucket_delta: int = 3
    monotonic_window: int = 0
    progress_every: int = 50


def _similarity_score(left: str, right: str) -> float:
    """Return deterministic similarity score in [0.0, 1.0]."""

    return SequenceMatcher(None, left, right, autojunk=False).ratio()


def _tokenize(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    return stripped.split()


def _quick_filter_score(asr_tokens: list[str], candidate_tokens: list[str]) -> float:
    if not asr_tokens and not candidate_tokens:
        return 1.0
    if not asr_tokens or not candidate_tokens:
        return 0.0

    asr_set = set(asr_tokens)
    candidate_set = set(candidate_tokens)
    overlap = len(asr_set.intersection(candidate_set))
    denom = max(1, min(len(asr_set), len(candidate_set)))
    return overlap / float(denom)


def _build_script_indexes(
    script_rows: list[ScriptRow],
    *,
    length_bucket_size: int,
) -> tuple[dict[str, set[int]], dict[str, set[int]], dict[int, set[int]], list[list[str]]]:
    ngram_index: dict[str, set[int]] = defaultdict(set)
    first_word_index: dict[str, set[int]] = defaultdict(set)
    length_bucket_index: dict[int, set[int]] = defaultdict(set)
    token_rows: list[list[str]] = []

    for row_idx, row in enumerate(script_rows):
        tokens = _tokenize(row.normalized_original)
        token_rows.append(tokens)

        if tokens:
            first_word_index[tokens[0]].add(row_idx)

        for token in tokens:
            ngram_index[token].add(row_idx)

        bucket = len(tokens) // length_bucket_size
        length_bucket_index[bucket].add(row_idx)

    return ngram_index, first_word_index, length_bucket_index, token_rows


def _gather_candidate_row_indexes(
    *,
    asr_tokens: list[str],
    asr_token_count: int,
    all_row_indexes: set[int],
    ngram_index: dict[str, set[int]],
    first_word_index: dict[str, set[int]],
    length_bucket_index: dict[int, set[int]],
    length_bucket_size: int,
    max_length_bucket_delta: int,
    monotonic_window: int,
    previous_best_row_idx: int | None,
) -> set[int]:
    candidate_indexes: set[int] = set()

    if asr_tokens:
        first_word_matches = first_word_index.get(asr_tokens[0], set())
        if first_word_matches:
            candidate_indexes.update(first_word_matches)

        for token in asr_tokens:
            token_matches = ngram_index.get(token, set())
            if token_matches:
                candidate_indexes.update(token_matches)

    base_bucket = asr_token_count // length_bucket_size
    for delta in range(-max_length_bucket_delta, max_length_bucket_delta + 1):
        bucket = base_bucket + delta
        if bucket in length_bucket_index:
            candidate_indexes.update(length_bucket_index[bucket])

    if monotonic_window > 0 and previous_best_row_idx is not None:
        monotonic_candidates = {
            idx
            for idx in all_row_indexes
            if previous_best_row_idx <= idx <= (previous_best_row_idx + monotonic_window)
        }
        candidate_indexes = candidate_indexes.intersection(monotonic_candidates) if candidate_indexes else monotonic_candidates

    if not candidate_indexes:
        return set(all_row_indexes)

    return candidate_indexes


def match_segments_to_script(
    *,
    asr_result: ASRResult,
    script_table: ScriptTable,
    config: MatchingConfig | None = None,
    progress_logger: Callable[[str], None] | None = None,
) -> MatchingOutput:
    """Match each ASR segment against indexed script candidates and keep best candidate."""

    effective_config = config or MatchingConfig()

    if not 0.0 <= effective_config.low_confidence_threshold <= 1.0:
        raise ValueError("low_confidence_threshold must be in [0.0, 1.0]")
    if not 0.0 <= effective_config.quick_filter_threshold <= 1.0:
        raise ValueError("quick_filter_threshold must be in [0.0, 1.0]")
    if effective_config.length_bucket_size <= 0:
        raise ValueError("length_bucket_size must be greater than 0")
    if effective_config.max_length_bucket_delta < 0:
        raise ValueError("max_length_bucket_delta must be greater than or equal to 0")
    if effective_config.monotonic_window < 0:
        raise ValueError("monotonic_window must be greater than or equal to 0")
    if effective_config.progress_every <= 0:
        raise ValueError("progress_every must be greater than 0")
    if not script_table.rows:
        raise ValueError("Script table must contain at least one row for matching")

    (
        ngram_index,
        first_word_index,
        length_bucket_index,
        token_rows,
    ) = _build_script_indexes(script_table.rows, length_bucket_size=effective_config.length_bucket_size)

    all_row_indexes = set(range(len(script_table.rows)))
    row_index_by_id = {row.values["id"]: idx for idx, row in enumerate(script_table.rows)}

    matches: list[MatchResult] = []
    total_segments = len(asr_result["segments"])
    previous_best_row_idx: int | None = None

    for index, segment in enumerate(asr_result["segments"], start=1):
        normalized_asr = normalize_text(segment["text"])
        asr_tokens = _tokenize(normalized_asr)
        candidate_indexes = _gather_candidate_row_indexes(
            asr_tokens=asr_tokens,
            asr_token_count=len(asr_tokens),
            all_row_indexes=all_row_indexes,
            ngram_index=ngram_index,
            first_word_index=first_word_index,
            length_bucket_index=length_bucket_index,
            length_bucket_size=effective_config.length_bucket_size,
            max_length_bucket_delta=effective_config.max_length_bucket_delta,
            monotonic_window=effective_config.monotonic_window,
            previous_best_row_idx=previous_best_row_idx,
        )

        scored_candidates: list[tuple[float, str, int, str]] = []
        for row_idx in sorted(candidate_indexes):
            row = script_table.rows[row_idx]
            quick_score = _quick_filter_score(asr_tokens, token_rows[row_idx])
            if quick_score < effective_config.quick_filter_threshold:
                continue

            line_id = row.values["id"]
            score = _similarity_score(normalized_asr, row.normalized_original)
            scored_candidates.append((score, line_id, row.row_number, row.values["original"]))

        if not scored_candidates:
            for row_idx in sorted(candidate_indexes):
                row = script_table.rows[row_idx]
                line_id = row.values["id"]
                score = _similarity_score(normalized_asr, row.normalized_original)
                scored_candidates.append((score, line_id, row.row_number, row.values["original"]))

        best_score, best_line_id, _, best_text = max(
            scored_candidates,
            key=lambda item: (item[0], item[1], -item[2]),
        )
        score = round(best_score, 6)

        previous_best_row_idx = row_index_by_id.get(best_line_id, previous_best_row_idx)

        matches.append(
            MatchResult(
                segment_id=segment["segment_id"],
                start_time=float(segment["start"]),
                end_time=float(segment["end"]),
                asr_text=segment["text"],
                matched_id=best_line_id,
                matched_text=best_text,
                score=score,
                low_confidence=score < effective_config.low_confidence_threshold,
            )
        )

        if progress_logger is not None and (
            index % effective_config.progress_every == 0 or index == total_segments
        ):
            progress_logger(f"Verbose: matching progress {index}/{total_segments} segments")

    return MatchingOutput(matches=matches)
