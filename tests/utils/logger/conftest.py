"""
Shared fixtures for the medpipe.utils.logger test suite.
"""

import logging
from collections.abc import Generator

import pytest


@pytest.fixture(autouse=True)
def reset_logger() -> Generator:
    """
    Fixture to reset the 'medpipe' logger handlers before and after each test.
    This prevents state leakage where handlers from one test pollute another.
    """
    logger = logging.getLogger("medpipe")
    logger.handlers.clear()
    yield
    logger.handlers.clear()
