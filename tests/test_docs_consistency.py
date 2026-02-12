from __future__ import annotations

from pathlib import Path
import unittest

from src.asr.registry import list_declared_backends


class DocsConsistencyTests(unittest.TestCase):
    def test_declared_registry_backend_names_are_documented(self) -> None:
        declared_backends = list_declared_backends()

        cli_doc = Path("docs/CLI.md").read_text(encoding="utf-8")
        status_doc = Path("docs/STATUS.md").read_text(encoding="utf-8")

        for backend_name in declared_backends:
            with self.subTest(backend_name=backend_name):
                self.assertIn(backend_name, cli_doc)
                self.assertIn(backend_name, status_doc)

    def test_milestone_backend_checkboxes_cover_completed_changelog_items(self) -> None:
        status_doc = Path("docs/STATUS.md").read_text(encoding="utf-8")

        milestone_section_start = status_doc.index("## Milestone 2")
        milestone_section_end = status_doc.index("## Milestone 3")
        milestone_two_section = status_doc[milestone_section_start:milestone_section_end]

        changelog_start = status_doc.index("## Change log (manual)")
        changelog_section = status_doc[changelog_start:]

        completed_item_checks = {
            "generic adapter": {
                "changelog_markers": ["generic adapter"],
                "milestone_checkbox": "- [x] ASR backend adapter (generic)",
            },
            "timestamp normalization": {
                "changelog_markers": ["timestamp normalization"],
                "milestone_checkbox": "- [x] Timestamp normalization across backends",
            },
            "whisperx": {
                "changelog_markers": ["whisperx asr backend"],
                "milestone_checkbox": "- [x] WhisperX backend",
            },
            "forced alignment": {
                "changelog_markers": ["forced alignment", "qwen3-forced-aligner"],
                "milestone_checkbox": "- [x] Forced alignment support",
            },
            "vibevoice": {
                "changelog_markers": ["vibevoice asr backend"],
                "milestone_checkbox": "- [x] VibeVoice backend via shared adapter path",
            },
        }

        changelog_lower = changelog_section.lower()
        for item_name, item in completed_item_checks.items():
            if any(marker in changelog_lower for marker in item["changelog_markers"]):
                with self.subTest(item=item_name):
                    self.assertIn(item["milestone_checkbox"], milestone_two_section)


class DocsCliFlagParityTests(unittest.TestCase):
    def test_cli_parser_flags_have_documented_sections(self) -> None:
        from cli import build_parser

        parser = build_parser()
        documented = Path("docs/CLI.md").read_text(encoding="utf-8")

        expected_headings: dict[str, str] = {
            "--input": "### `--input`",
            "--script": "### `--script`",
            "--out": "### `--out`",
            "--asr-backend": "### `--asr-backend <name>`",
            "--mock-asr": "### `--mock-asr <file>`",
            "--model-path": "### `--model-path <path>`",
            "--model-id": "### `--model-id <repo_id>`",
            "--revision": "### `--revision <revision>`",
            "--auto-download": "### `--auto-download <tiny|base|small>`",
            "--device": "### `--device <cpu|cuda|auto>`",
            "--compute-type": "### `--compute-type <float16|float32|auto>`",
            "--chunk": "### `--chunk <seconds>`",
            "--scene-gap": "### `--scene-gap <seconds>`",
            "--ffmpeg-path": "### `--ffmpeg-path <path>`",
            "--keep-wav": "### `--keep-wav`",
            "--verbose": "### `--verbose`",
            "--version": "### `--version`",
            "--match-threshold": "### `--match-threshold <float>`",
            "--match-quick-threshold": "### `--match-quick-threshold <float>`",
            "--match-length-bucket-size": "### `--match-length-bucket-size <int>`",
            "--match-max-length-bucket-delta": "### `--match-max-length-bucket-delta <int>`",
            "--match-monotonic-window": "### `--match-monotonic-window <int>`",
            "--match-progress-every": "### `--match-progress-every <int>`",
            "--asr-vad-filter": "### `--asr-vad-filter <on|off>`",
            "--asr-merge-short-segments": "### `--asr-merge-short-segments <seconds>`",
            "--asr-language": "### `--asr-language <language>`",
            "--asr-beam-size": "### `--asr-beam-size <int>`",
            "--asr-temperature": "### `--asr-temperature <float>`",
            "--asr-best-of": "### `--asr-best-of <int>`",
            "--asr-no-speech-threshold": "### `--asr-no-speech-threshold <float>`",
            "--asr-logprob-threshold": "### `--asr-logprob-threshold <float>`",
            "--progress": "### `--progress <on|off>`",
        }

        parser_flags = sorted(
            option
            for action in parser._actions
            for option in action.option_strings
            if option.startswith("--") and option not in {"--help"}
        )

        self.assertEqual(sorted(expected_headings), parser_flags)

        for flag, heading in expected_headings.items():
            with self.subTest(flag=flag):
                self.assertIn(heading, documented)


if __name__ == "__main__":
    unittest.main()
