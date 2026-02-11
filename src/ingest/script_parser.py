"""Script ingestion for TSV/CSV files."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from src.match.normalization import normalize_text

_REQUIRED_COLUMNS = ("id", "original")


@dataclass(frozen=True)
class ScriptRow:
    row_number: int
    values: dict[str, str]
    normalized_original: str


@dataclass(frozen=True)
class ScriptTable:
    delimiter: str
    columns: list[str]
    rows: list[ScriptRow]


def detect_delimiter(header_line: str) -> str:
    """Detect whether header is tab-separated or comma-separated."""

    tab_count = header_line.count("\t")
    comma_count = header_line.count(",")
    if tab_count > comma_count:
        return "\t"
    return ","


def load_script_table(script_path: Path) -> ScriptTable:
    """Load script rows from CSV/TSV and validate required columns."""

    if not script_path.exists():
        raise ValueError(f"Script file not found: {script_path}")
    if script_path.suffix.lower() not in {".tsv", ".csv"}:
        raise ValueError("Script file must have .tsv or .csv extension.")

    with script_path.open("r", encoding="utf-8", newline="") as handle:
        first_line = handle.readline()
        if not first_line:
            raise ValueError("Script file is empty.")

        delimiter = detect_delimiter(first_line)
        handle.seek(0)
        reader = csv.DictReader(handle, delimiter=delimiter)
        fieldnames = [name.strip() for name in (reader.fieldnames or []) if name]

        if not fieldnames:
            raise ValueError("Script file is missing a header row.")

        missing_columns = [col for col in _REQUIRED_COLUMNS if col not in fieldnames]
        if missing_columns:
            raise ValueError(
                "Missing required script columns: " + ", ".join(missing_columns)
            )

        rows: list[ScriptRow] = []
        for row_idx, raw_row in enumerate(reader, start=2):
            if None in raw_row:
                raise ValueError(f"Row {row_idx} has too many fields for detected delimiter.")

            row_values: dict[str, str] = {}
            for column in fieldnames:
                value = raw_row.get(column)
                row_values[column] = "" if value is None else value

            rows.append(
                ScriptRow(
                    row_number=row_idx,
                    values=row_values,
                    normalized_original=normalize_text(row_values["original"]),
                )
            )

    return ScriptTable(delimiter=delimiter, columns=fieldnames, rows=rows)
