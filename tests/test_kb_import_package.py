"""kb-import reports the crate version (T001's acceptance check).

This drives the real console entry point in a subprocess rather than calling
`build_parser` in process, so it also exercises `kb_import.__version__`, which
comes from `importlib.metadata` and therefore depends on the package actually
being installed -- a mock could not catch a metadata lookup that silently
failed (`REPO_INVARIANTS.md` ENG-003).
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parent.parent


def _crate_version() -> str:
    """Read the version out of Cargo.toml's `[package]` table.

    Returns:
        The version string, e.g. "6.2.0".
    """
    text = (REPOSITORY / "Cargo.toml").read_text(encoding="utf-8")
    match = re.search(r'(?sm)^\[package\]\n(?:(?!^\[).)*?^version = "(?P<version>[^"]+)"', text)
    assert match, "Cargo.toml's [package] table has no version key"
    return match.group("version")


def test_kb_import_version_matches_the_crate_version():
    result = subprocess.run(
        [sys.executable, "-m", "kb_import.cli", "--version"],
        capture_output=True,
        text=True,
        cwd=REPOSITORY,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"kb-import {_crate_version()}"
