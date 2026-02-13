from __future__ import annotations

from pathlib import Path
import re
import unittest

from src.asr.registry import list_declared_backends


class DocsConsistencyTests(unittest.TestCase):
    @staticmethod
    def _status_sections(status_doc: str) -> tuple[str, str, str]:
        milestone_two_start = status_doc.index("## Milestone 2")
        milestone_three_start = status_doc.index("## Milestone 3")
        changelog_start = status_doc.index("## Change log (manual)")
        milestone_two = status_doc[milestone_two_start:milestone_three_start]
        milestone_three = status_doc[milestone_three_start:changelog_start]
        changelog = status_doc[changelog_start:]
        return milestone_two, milestone_three, changelog

    @staticmethod
    def _checklist_lines(section: str) -> list[str]:
        return [line.strip() for line in section.splitlines() if line.lstrip().startswith("- [")]

    @staticmethod
    def _entry_paths(line: str) -> list[str]:
        return re.findall(r"`([^`]+)`", line)

    @staticmethod
    def _normalized_entry_name(line: str) -> str:
        entry = re.sub(r"^- \[[x ~]\]\s*", "", line.strip())
        entry = re.sub(r"\s*\([^)]*\)\s*$", "", entry)
        entry = " ".join(entry.split())
        return entry.lower()

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

        milestone_two_section, _, changelog_section = self._status_sections(status_doc)

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

    def test_status_milestone_entries_with_paths_require_completed_checkbox(self) -> None:
        status_doc = Path("docs/STATUS.md").read_text(encoding="utf-8")
        milestone_two, milestone_three, _ = self._status_sections(status_doc)

        lines = self._checklist_lines(milestone_two) + self._checklist_lines(milestone_three)
        for line in lines:
            paths = self._entry_paths(line)
            has_impl_or_test_paths = any(
                path.startswith(("src/", "tests/", "cli.py")) for path in paths
            )
            if has_impl_or_test_paths:
                with self.subTest(line=line):
                    self.assertTrue(
                        line.startswith("- [x]"),
                        "Milestone checklist entries that cite implementation/test files must use "
                        "a completed checkbox `- [x]`. Update the milestone checkbox line in "
                        "docs/STATUS.md (not only changelog notes).",
                    )

    def test_status_completed_marker_includes_impl_and_test_paths(self) -> None:
        status_doc = Path("docs/STATUS.md").read_text(encoding="utf-8")
        milestone_two, milestone_three, _ = self._status_sections(status_doc)

        lines = self._checklist_lines(milestone_two) + self._checklist_lines(milestone_three)
        for line in lines:
            if not line.startswith("- [x]"):
                continue
            paths = self._entry_paths(line)
            if not paths:
                continue

            has_impl_path = any(path.startswith(("src/", "cli.py")) for path in paths)
            has_test_path = any(path.startswith("tests/") for path in paths)

            with self.subTest(line=line):
                self.assertTrue(
                    has_impl_path and has_test_path,
                    "Completed STATUS milestone entries with file markers must include at least "
                    "one implementation path (`src/` or `cli.py`) and one test path (`tests/`). "
                    "Update the milestone checkbox line in docs/STATUS.md, not only changelog "
                    "footnotes.",
                )

    def test_changelog_completed_notes_do_not_conflict_with_unchecked_milestones(self) -> None:
        status_doc = Path("docs/STATUS.md").read_text(encoding="utf-8")
        milestone_two, milestone_three, changelog = self._status_sections(status_doc)

        changelog_lower = changelog.lower()
        unchecked_lines = [
            line
            for line in (self._checklist_lines(milestone_two) + self._checklist_lines(milestone_three))
            if line.startswith("- [ ]")
        ]

        for line in unchecked_lines:
            item_name = self._normalized_entry_name(line)
            if len(item_name) < 12:
                continue

            with self.subTest(item=item_name):
                self.assertNotIn(
                    item_name,
                    changelog_lower,
                    "Feature appears in changelog as completed while milestone checkbox is still "
                    "unchecked. Mark the milestone line as `- [x]` in docs/STATUS.md (do not "
                    "record completion only in changelog notes).",
                )


class DocsDeclaredDisabledDefinitionTests(unittest.TestCase):
    def test_cli_docs_define_declared_but_disabled_diagnostics(self) -> None:
        cli_doc = Path("docs/CLI.md").read_text(encoding="utf-8")

        required_fragments = (
            "**Declared but disabled backend**",
            "missing optional dependencies",
            "feature flag",
            "experimental backend disabled by default",
            "Dependency-gated errors include actionable install-extra guidance when disabled due to missing optional dependencies",
        )

        for fragment in required_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, cli_doc)


class DocsCudaRerunWordingTests(unittest.TestCase):
    def test_cpu_rerun_wording_replaces_fallback_semantics(self) -> None:
        cli_doc = Path("docs/CLI.md").read_text(encoding="utf-8")
        integration_issues_doc = Path("docs/Integration-issues.md").read_text(encoding="utf-8")

        self.assertIn("CPU rerun policy (no autoswitch)", cli_doc)
        self.assertNotIn("CPU fallback semantics", cli_doc)

        context_docs = (
            "docs/CLI.md",
            "docs/Integration-issues.md",
        )
        for doc_path in context_docs:
            doc_text = Path(doc_path).read_text(encoding="utf-8")
            with self.subTest(doc=doc_path):
                self.assertNotRegex(
                    doc_text,
                    re.compile(r"CPU\s+fallback", re.IGNORECASE),
                    "Use explicit CPU rerun wording instead of fallback terminology in CUDA failure guidance.",
                )
                self.assertRegex(
                    doc_text,
                    re.compile(r"(?:No automatic backend/device switching|Do not perform automatic backend/device switching)", re.IGNORECASE),
                    "CUDA guidance must state that backend/device switching is never automatic.",
                )
                self.assertIn(
                    "--device cpu",
                    doc_text,
                    "CUDA guidance must include explicit rerun instruction with --device cpu.",
                )
                self.assertRegex(
                    doc_text,
                    re.compile(r"manual\s+rerun\s+with\s+`?--device\s+cpu`?", re.IGNORECASE),
                    "CUDA guidance must use manual rerun wording rather than fallback phrasing.",
                )
                self.assertRegex(
                    doc_text,
                    re.compile(r"Do not perform automatic backend/device switching", re.IGNORECASE),
                    "Readiness and CUDA guidance must keep the no-autoswitch policy explicit.",
                )


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
            "--asr-preflight-only": "### `--asr-preflight-only`",
            "--alignment-preflight-only": "### `--alignment-preflight-only`",
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
            "--qwen3-batch-size": "### `--qwen3-batch-size <int>`",
            "--qwen3-chunk-length-s": "### `--qwen3-chunk-length-s <float>`",
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


class DocsAsrPreflightInvocationTests(unittest.TestCase):
    def test_qwen3_mode_contracts_are_documented(self) -> None:
        integration_doc = Path("docs/Integration.md").read_text(encoding="utf-8")
        contracts_doc = Path("docs/Data-contracts.md").read_text(encoding="utf-8")

        required_fragments = (
            "`qwen3-asr` is explicitly `text-only`",
            "Forced alignment is a distinct pipeline path",
            "qwen3-forced-aligner",
            "alignment contract/input",
        )

        for fragment in required_fragments[:2]:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, integration_doc)

        for fragment in required_fragments[2:]:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, contracts_doc)

    def test_preflight_docs_cover_installed_and_source_invocation_contract(self) -> None:
        cli_doc = Path("docs/CLI.md").read_text(encoding="utf-8")

        required_fragments = (
            "cutscene-locator --asr-preflight-only",
            "py .\\cli.py --asr-preflight-only",
            "python ./cli.py --asr-preflight-only",
            "must emit the same single-line JSON stdout contract in preflight-only mode",
            "PowerShell",
            "shell-level formatting artifacts",
        )

        for fragment in required_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, cli_doc)

    def test_preflight_docs_include_supported_preflight_modes(self) -> None:
        cli_doc = Path("docs/CLI.md").read_text(encoding="utf-8")

        required_fragments = (
            "\"mode\":\"asr_preflight_only\"",
            "\"mode\":\"alignment_preflight_only\"",
            "`--asr-preflight-only`",
            "`--alignment-preflight-only`",
        )

        for fragment in required_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, cli_doc)


if __name__ == "__main__":
    unittest.main()
