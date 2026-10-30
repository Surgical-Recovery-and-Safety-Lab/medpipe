"""
Tests for the metadata of the package in pyproject.toml.

PyPI rejects an upload when a classifier is not in its list, and that is only
found at the end of a release, so the classifiers are checked here.
"""

import tomllib
from pathlib import Path

import pytest

trove_classifiers = pytest.importorskip("trove_classifiers")

PYPROJECT = Path(__file__).parent.parent / "pyproject.toml"


def _classifiers() -> list[str]:
    with open(PYPROJECT, "rb") as file:
        return tomllib.load(file)["project"]["classifiers"]


@pytest.mark.parametrize("classifier", _classifiers())
def test_classifier_is_accepted_by_pypi(classifier: str) -> None:
    """Test that every classifier of the package is in the list of PyPI."""
    assert classifier in trove_classifiers.classifiers
