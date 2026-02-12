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


if __name__ == "__main__":
    unittest.main()
