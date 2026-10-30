"""
Tests for the scripts of the examples folder, which are shown in the
documentation.
"""

import ast
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"
SCRIPTS = sorted(EXAMPLES_DIR.glob("*.py"))


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: path.name)
def test_script_is_valid_python(script: Path) -> None:
    """Test that every example script can be parsed."""
    ast.parse(script.read_text(encoding="utf-8"), filename=str(script))


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: path.name)
def test_script_is_shown_in_the_documentation(script: Path) -> None:
    """Test that every example script is referenced by the documentation, so
    that none is left out."""
    pages = [
        REPO_ROOT / "docs" / "examples.md",
        REPO_ROOT / "docs" / "start" / "quickstart.md",
    ]
    text = "\n".join(page.read_text(encoding="utf-8") for page in pages)

    assert script.name in text


@pytest.mark.parametrize("name", ["config_in_python.py", "custom_components.py"])
def test_light_script_runs(tmp_path: Path, name: str) -> None:
    """Test that the scripts that run in a few seconds finish without error.

    The scripts use paths relative to the examples folder, so they are run
    from a copy of the folder and of the test data.
    """
    shutil.copytree(
        EXAMPLES_DIR,
        tmp_path / "examples",
        ignore=shutil.ignore_patterns("artifacts", "python_config_runs", "__pycache__"),
    )
    (tmp_path / "tests").mkdir()
    shutil.copy(REPO_ROOT / "tests" / "test_data.csv", tmp_path / "tests")

    result = subprocess.run(
        [sys.executable, name],
        cwd=tmp_path / "examples",
        env={**os.environ, "MPLBACKEND": "Agg"},
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )

    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip() != ""
