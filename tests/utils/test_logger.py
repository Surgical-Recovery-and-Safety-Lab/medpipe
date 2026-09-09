"""
Tests for the medpipe.utils.logger module.
"""

import logging
import sys
from pathlib import Path
from typing import Generator

import pytest

from medpipe.utils.logger import (
    CompactProgressFilter,
    add_file_handler,
    get_console_logger,
    set_verbosity,
)


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


# --- Tests for CompactProgressFilter ---


def test_compact_progress_filter_milestone_filtering() -> None:
    """Test that CompactProgressFilter allows milestones and warnings, but blocks non-milestone INFO logs."""
    progress_filter = CompactProgressFilter()

    # 1. Milestone INFO message -> Should PASS
    record_milestone = logging.LogRecord(
        name="medpipe",
        level=logging.INFO,
        pathname="pipeline.py",
        lineno=10,
        msg="Step 1/3: Ingesting and splitting dataset.",
        args=(),
        exc_info=None,
    )
    assert progress_filter.filter(record_milestone) is True

    # 2. Granular operational INFO message -> Should be FILTERED (False)
    record_granular = logging.LogRecord(
        name="medpipe.orchestrator",
        level=logging.INFO,
        pathname="orchestrator.py",
        lineno=20,
        msg="Filtered dataset from 11 down to 11 required columns.",
        args=(),
        exc_info=None,
    )
    assert progress_filter.filter(record_granular) is False

    # 3. WARNING level message -> Should PASS regardless of text content
    record_warning = logging.LogRecord(
        name="medpipe.evaluator",
        level=logging.WARNING,
        pathname="evaluator.py",
        lineno=30,
        msg="Subgroup is empty. Skipping.",
        args=(),
        exc_info=None,
    )
    assert progress_filter.filter(record_warning) is True


# --- Tests for Console Logger & Verbosity ---


def test_get_console_logger_initialization_default() -> None:
    """Test default initialization of the console logger with 'compact' verbosity."""
    logger = get_console_logger(name="medpipe", verbose="compact")

    assert logger.name == "medpipe"
    # Base root logger set to DEBUG to allow all traffic to pass to file handlers
    assert logger.level == logging.DEBUG
    assert len(logger.handlers) == 1

    handler = logger.handlers[0]
    assert isinstance(handler, logging.StreamHandler)
    assert handler.stream == sys.stdout
    assert handler.level == logging.INFO

    # Compact mode attaches CompactProgressFilter
    assert any(isinstance(f, CompactProgressFilter) for f in handler.filters)


def test_get_console_logger_prevents_duplicates() -> None:
    """Test that multiple calls do not attach redundant stream handlers."""
    logger1 = get_console_logger()
    logger2 = get_console_logger()

    assert logger1 is logger2
    assert len(logger1.handlers) == 1  # Should still be exactly 1


def test_get_console_logger_custom_name_and_verbosity() -> None:
    """Test initializing a sub-logger with custom name and 'quiet' threshold."""
    root_logger = logging.getLogger("medpipe")
    root_logger.handlers.clear()

    custom_logger = get_console_logger(name="medpipe.custom_logger", verbose="quiet")

    assert custom_logger.name == "medpipe.custom_logger"
    assert len(custom_logger.handlers) == 0
    assert len(root_logger.handlers) == 1
    assert root_logger.handlers[0].level == logging.WARNING

    # Quiet mode removes CompactProgressFilter
    assert not any(
        isinstance(f, CompactProgressFilter) for f in root_logger.handlers[0].filters
    )


@pytest.mark.parametrize(
    "verbose, expected_level, expects_filter",
    [
        ("quiet", logging.WARNING, False),
        (False, logging.WARNING, False),
        (0, logging.WARNING, False),
        ("compact", logging.INFO, True),
        (1, logging.INFO, True),
        ("info", logging.INFO, False),
        (True, logging.INFO, False),
        (logging.INFO, logging.INFO, False),
        ("debug", logging.DEBUG, False),
        (3, logging.DEBUG, False),
    ],
)
def test_set_verbosity_modes(
    verbose: str | bool | int, expected_level: int, expects_filter: bool
) -> None:
    """Test setting global verbosity across all valid mode representations."""
    get_console_logger()  # Initialize root stream handler
    set_verbosity(verbose)

    root_logger = logging.getLogger("medpipe")
    handler = root_logger.handlers[0]

    assert handler.level == expected_level
    has_filter = any(isinstance(f, CompactProgressFilter) for f in handler.filters)
    assert has_filter is expects_filter


# --- Tests for File Handler ---


def test_add_file_handler_creates_file_and_directory(tmp_path: Path) -> None:
    """Test that the file handler correctly creates nested parent directories."""
    logger = get_console_logger()

    nested_log_dir = tmp_path / "artifacts" / "v1"
    add_file_handler(logger, log_dir=nested_log_dir, filename="test.log")

    expected_file = nested_log_dir / "test.log"
    assert expected_file.exists()
    assert len(logger.handlers) == 2  # StreamHandler + FileHandler


def test_add_file_handler_writes_unfiltered_logs(tmp_path: Path) -> None:
    """
    Test edge case: file handler captures all DEBUG messages and does not
    inherit compact filtering applied to the console handler.
    """
    logger = get_console_logger(verbose="compact")
    log_dir = tmp_path / "v2"

    add_file_handler(
        logger, log_dir=log_dir, filename="execution.log", level=logging.DEBUG
    )

    # Emit logs
    logger.debug("Debug diagnostic info.")
    logger.info(
        "Filtered dataset from 11 down to 11 required columns."
    )  # Non-milestone INFO
    logger.warning("Warning message.")

    log_file = log_dir / "execution.log"
    with open(log_file, "r", encoding="utf-8") as f:
        log_contents = f.read()

    # File should contain all unfiltered logs
    assert "DEBUG" in log_contents
    assert "Debug diagnostic info." in log_contents
    assert "Filtered dataset from 11 down to 11 required columns." in log_contents
    assert "WARNING" in log_contents
    assert "Warning message." in log_contents


def test_add_file_handler_multiple_handlers(tmp_path: Path) -> None:
    """Test attaching multiple file handlers for different threshold levels."""
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
