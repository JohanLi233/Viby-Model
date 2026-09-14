"""Check the no-GPU entry point and Markdown diagnostics with stdlib only."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/check_repo.py"
spec = importlib.util.spec_from_file_location("viby_check_repo", SCRIPT)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class TestRepoTools(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def document(self, text):
        path = self.root / "README.md"
        path.write_text(text, encoding="utf-8")
        return path

    def test_links_resolve_relative_to_document_and_decode_spaces(self):
        (self.root / "a file.md").write_text("# Target\n")
        doc = self.document(
            "[file](a%20file.md#target)\n[file](<a file.md>)\n"
            "[web](https://example.invalid/missing)\n[anchor](#local)\n"
        )
        self.assertEqual(checker.markdown_errors(doc), [])

    def test_missing_link_has_source_line(self):
        errors = checker.markdown_errors(self.document("# Title\n\n[bad](gone.md)\n"))
        self.assertEqual(len(errors), 1)
        self.assertIn("README.md:3:", errors[0])
        self.assertIn("gone.md", errors[0])

    def test_code_examples_are_not_links(self):
        doc = self.document("`[example](missing)`\n```md\n[example](missing)\n```\n")
        self.assertEqual(checker.markdown_errors(doc), [])

    def test_long_fence_needs_matching_close(self):
        doc = self.document("````md\n```\n[example](missing)\n````\n")
        self.assertEqual(checker.markdown_errors(doc), [])
        errors = checker.markdown_errors(self.document("~~~python\nprint(1)\n```\n"))
        self.assertEqual(len(errors), 1)
        self.assertIn("unclosed code fence", errors[0])

    def fixture_repo(self):
        tool = self.root / "scripts/check_repo.py"
        tool.parent.mkdir()
        tool.write_bytes(SCRIPT.read_bytes())
        for name in checker.ENTRY_DOCS:
            target = self.root / name
            target.parent.mkdir(exist_ok=True, parents=True)
            target.write_text("# Fixture\n")
        for files in checker.TEST_GROUPS.values():
            for name in files:
                target = self.root / "tests" / name
                target.parent.mkdir(exist_ok=True)
                # A dry-run or default check must never import these tests.
                target.write_text("raise RuntimeError('test imported unexpectedly')\n")
        return tool

    def run_cli(self, tool, *args):
        return subprocess.run(
            [sys.executable, str(tool), *args],
            cwd=self.root.parent,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_default_is_static_and_broken_link_fails(self):
        tool = self.fixture_repo()
        result = self.run_cli(tool)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.document("[bad](missing.md)\n")
        result = self.run_cli(tool)
        self.assertEqual(result.returncode, 1)
        self.assertIn("missing link target", result.stderr)

    def test_dry_run_does_not_import_tests_and_forwards_pytest_arguments(self):
        tool = self.fixture_repo()
        result = self.run_cli(
            tool, "test", "engine", "--dry-run", "--", "-k", "zero or cache"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("tests/test_engine_speculative.py", result.stdout)
        self.assertIn("'zero or cache'", result.stdout)

    def test_unknown_group_or_missing_test_fails(self):
        tool = self.fixture_repo()
        self.assertEqual(self.run_cli(tool, "test", "typo").returncode, 2)
        (self.root / "tests/test_engine_speculative.py").unlink()
        self.assertEqual(self.run_cli(tool, "test", "engine", "--dry-run").returncode, 2)

    def test_pytest_failure_exit_code_is_preserved(self):
        tool = self.fixture_repo()
        # A local stand-in exercises real process dispatch without any test/GPU imports.
        (self.root / "pytest.py").write_text(
            "from pathlib import Path\nimport sys\n"
            "Path('received.txt').write_text(repr(sys.argv[1:]))\n"
            "raise SystemExit(5)\n"
        )
        result = self.run_cli(
            tool, "test", "tools", "--python", sys.executable, "--", "-x"
        )
        self.assertEqual(result.returncode, 5, result.stderr)
        received = (self.root / "received.txt").read_text()
        self.assertIn("tests/test_repo_tools.py", received)
        self.assertIn("'-x'", received)


if __name__ == "__main__":
    unittest.main()
