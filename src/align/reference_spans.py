"""Deterministic conversion from script rows to alignment reference spans."""

from __future__ import annotations

from collections.abc import Callable

from src.ingest import ScriptRow, ScriptTable

from .base import ReferenceSpan


def script_table_to_reference_spans(
    script_table: ScriptTable,
    *,
    log_callback: Callable[[str], None] | None = None,
) -> list[ReferenceSpan]:
    """Map script rows to alignment ``reference_spans[]`` in stable file order.

    Rows with empty/invalid ``id`` or ``original`` are skipped deterministically.
    When ``log_callback`` is provided, skip reasons are emitted for diagnostics.
    """

    reference_spans: list[ReferenceSpan] = []
    for row in script_table.rows:
        maybe_span = _row_to_reference_span(row, log_callback=log_callback)
        if maybe_span is not None:
            reference_spans.append(maybe_span)
    return reference_spans


def _row_to_reference_span(
    row: ScriptRow,
    *,
    log_callback: Callable[[str], None] | None,
) -> ReferenceSpan | None:
    raw_ref_id = row.values.get("id")
    if not isinstance(raw_ref_id, str) or not raw_ref_id.strip():
        _emit_skip(log_callback, row.row_number, "missing or empty 'id'")
        return None

    raw_text = row.values.get("original")
    if not isinstance(raw_text, str) or not raw_text.strip():
        _emit_skip(log_callback, row.row_number, "missing or empty 'original'")
        return None

    return {
        "ref_id": raw_ref_id,
        "text": raw_text,
    }


def _emit_skip(log_callback: Callable[[str], None] | None, row_number: int, reason: str) -> None:
    if log_callback is None:
        return
    log_callback(f"alignment reference span skip row {row_number}: {reason}")

