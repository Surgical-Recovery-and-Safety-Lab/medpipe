"""
Tests for the medpipe.utils.logger module.
"""

import logging
import sys
from pathlib import Path
from typing import Generator

import pytest

from medpipe.utils.logger import add_file_handler, get_console_logger


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


# --- Tests for Console Logger ---


def test_get_console_logger_initialization() -> None:
    """Test successful initialization of the console logger with correct levels."""
    logger = get_console_logger(name="medpipe", level=logging.INFO)

    assert logger.name == "medpipe"
    # Base logger should be set to DEBUG to allow all traffic through
    assert logger.level == logging.DEBUG
    assert len(logger.handlers) == 1

    handler = logger.handlers[0]
    assert isinstance(handler, logging.StreamHandler)
    assert handler.stream == sys.stdout
    assert handler.level == logging.INFO


def test_get_console_logger_prevents_duplicates() -> None:
    """Test that multiple calls do not attach redundant stream handlers."""
    logger1 = get_console_logger()
    logger2 = get_console_logger()

    assert logger1 is logger2
    assert len(logger1.handlers) == 1  # Should still be exactly 1


def test_get_console_logger_custom_name_and_level() -> None:
    """Test initializing a logger with a custom name and a different threshold."""
    custom_logger = get_console_logger(name="custom_logger", level=logging.WARNING)

    assert custom_logger.name == "custom_logger"
    assert custom_logger.handlers[0].level == logging.WARNING

    # Cleanup for the custom logger
    custom_logger.handlers.clear()


# --- Tests for File Handler ---


def test_add_file_handler_creates_file_and_directory(tmp_path: Path) -> None:
    """Test that the file handler correctly creates nested parent directories."""
    logger = get_console_logger()

    # Intentionally use a nested directory that does not exist yet
    nested_log_dir = tmp_path / "artifacts" / "v1"

    add_file_handler(logger, log_dir=nested_log_dir, filename="test.log")

    expected_file = nested_log_dir / "test.log"
    assert expected_file.exists()
    assert len(logger.handlers) == 2  # StreamHandler + FileHandler


def test_add_file_handler_writes_correct_levels(tmp_path: Path) -> None:
    """
    Test edge case: file handler captures DEBUG, while console ignores it.
    Verifies that the format string is applied correctly to the file.
    """
    logger = get_console_logger(level=logging.INFO)
    log_dir = tmp_path / "v2"

    add_file_handler(
        logger, log_dir=log_dir, filename="execution.log", level=logging.DEBUG
    )

    # Emit logs
    logger.debug("This is a debug message.")
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")

    log_file = log_dir / "execution.log"
    with open(log_file, "r", encoding="utf-8") as f:
        log_contents = f.read()

    # The file should contain all three messages
    assert "DEBUG" in log_contents
    assert "This is a debug message." in log_contents
    assert "INFO" in log_contents
    assert "This is an info message." in log_contents
    assert "WARNING" in log_contents
    assert "This is a warning message." in log_contents


def test_add_file_handler_multiple_handlers(tmp_path: Path) -> None:
    """Test boundary: attaching multiple file handlers for different purposes."""
    logger = get_console_logger()
    log_dir = tmp_path / "v3"

    add_file_handler(logger, log_dir=log_dir, filename="debug.log", level=logging.DEBUG)
    add_file_handler(
        logger, log_dir=log_dir, filename="errors.log", level=logging.ERROR
    )

    assert len(logger.handlers) == 3

    logger.error("System failure")

    with (
        open(log_dir / "debug.log", "r") as f1,
        open(log_dir / "errors.log", "r") as f2,
    ):
        assert "System failure" in f1.read()
        assert "System failure" in f2.read()
