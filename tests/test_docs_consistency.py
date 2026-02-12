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


if __name__ == "__main__":
    unittest.main()
