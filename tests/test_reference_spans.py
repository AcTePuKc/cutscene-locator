from pathlib import Path

import unittest

from src.align.reference_spans import script_table_to_reference_spans
from src.ingest.script_parser import load_script_table


class ReferenceSpanConverterTests(unittest.TestCase):
    def test_script_table_to_reference_spans_preserves_file_order_and_text(self) -> None:
        table = load_script_table(Path("tests/fixtures/script_alignment_reference_spans.tsv"))

        spans = script_table_to_reference_spans(table)

        self.assertEqual(
            spans,
            [
                {"ref_id": "A_002", "text": "  First line with spaces  "},
                {"ref_id": "A_001", "text": "Second line!"},
                {"ref_id": "A_003", "text": "Third line unchanged"},
            ],
        )

    def test_script_table_to_reference_spans_reports_skipped_rows_only_when_verbose(self) -> None:
        table = load_script_table(Path("tests/fixtures/script_alignment_reference_spans.tsv"))

        verbose_logs: list[str] = []
        spans = script_table_to_reference_spans(table, log_callback=verbose_logs.append)

        self.assertEqual([span["ref_id"] for span in spans], ["A_002", "A_001", "A_003"])
        self.assertEqual(
            verbose_logs,
            [
                "alignment reference span skip row 3: missing or empty 'id'",
                "alignment reference span skip row 5: missing or empty 'original'",
            ],
        )

        quiet_logs: list[str] = []
        quiet_spans = script_table_to_reference_spans(table)
        self.assertEqual(quiet_spans, spans)
        self.assertEqual(quiet_logs, [])


if __name__ == "__main__":
    unittest.main()
