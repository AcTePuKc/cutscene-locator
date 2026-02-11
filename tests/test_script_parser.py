from pathlib import Path

import unittest

from src.ingest.script_parser import detect_delimiter, load_script_table
from src.match.normalization import normalize_text


class ScriptParserTests(unittest.TestCase):
    def test_detect_delimiter_prefers_tab_when_present(self) -> None:
        self.assertEqual(detect_delimiter("id\toriginal\ttranslation"), "\t")
        self.assertEqual(detect_delimiter("id,original,translation"), ",")

    def test_load_tsv_preserves_columns_and_adds_normalized_form(self) -> None:
        table = load_script_table(Path("tests/fixtures/script_sample.tsv"))

        self.assertEqual(table.delimiter, "\t")
        self.assertEqual(
            table.columns,
            ["id", "original", "translation", "file", "mission", "speaker", "notes"],
        )
        self.assertEqual(table.rows[0].values["original"], "  Hello,   WORLD! ")
        self.assertEqual(table.rows[0].values["translation"], "Здрасти")
        self.assertEqual(table.rows[0].values["speaker"], "NPC")
        self.assertEqual(table.rows[0].normalized_original, "hello world")

    def test_load_csv_autodetects_comma(self) -> None:
        table = load_script_table(Path("tests/fixtures/script_sample.csv"))

        self.assertEqual(table.delimiter, ",")
        self.assertEqual(table.rows[0].values["id"], "C01_001")
        self.assertEqual(table.rows[0].normalized_original, "hi there")

    def test_missing_required_column_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "Missing required script columns"):
            load_script_table(Path("tests/fixtures/script_missing_required.csv"))


class NormalizationTests(unittest.TestCase):
    def test_normalize_text_pipeline(self) -> None:
        normalized = normalize_text("  What... IS   this?!  ")
        self.assertEqual(normalized, "what is this")


if __name__ == "__main__":
    unittest.main()
