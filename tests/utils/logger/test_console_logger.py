"""
Tests for the get_console_logger function of the medpipe.utils.logger module.
"""

import logging
import sys
from pathlib import Path

from medpipe.utils.logger import CompactProgressFilter, get_console_logger


class TestGetConsoleLogger:
    """Test class for the get_console_logger function"""

    def test_initialization_default(self) -> None:
        """Test default initialization of the console logger with 'compact'
        verbosity."""
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

    def test_verbose_none_skips_verbosity_configuration(self) -> None:
        """Test that omitting verbose leaves the new handler's level and
        filters untouched, since set_verbosity is only called when verbose
        is not None."""
        logger = get_console_logger()

        assert len(logger.handlers) == 1
        handler = logger.handlers[0]
        assert handler.level == logging.NOTSET
        assert not any(isinstance(f, CompactProgressFilter) for f in handler.filters)

    def test_root_logger_always_set_to_debug(self) -> None:
        """Test that the root 'medpipe' logger threshold is DEBUG regardless
        of the requested console verbosity."""
        get_console_logger(verbose="quiet")

        assert logging.getLogger("medpipe").level == logging.DEBUG

    def test_returns_named_child_logger(self) -> None:
        """Test that the returned logger is the exact instance for the
        requested name."""
        logger = get_console_logger(name="medpipe.pipeline")

        assert logger is logging.getLogger("medpipe.pipeline")

    def test_prevents_duplicate_handlers_across_calls(self) -> None:
        """Test that multiple calls do not attach redundant stream handlers."""
        logger1 = get_console_logger()
        logger2 = get_console_logger()

        assert logger1 is logger2
        assert len(logger1.handlers) == 1  # Should still be exactly 1

    def test_custom_name_and_verbosity(self) -> None:
        """Test initializing a sub-logger with custom name and 'quiet' threshold."""
        root_logger = logging.getLogger("medpipe")

        custom_logger = get_console_logger(
            name="medpipe.custom_logger", verbose="quiet"
        )

        assert custom_logger.name == "medpipe.custom_logger"
        assert len(custom_logger.handlers) == 0
        assert len(root_logger.handlers) == 1
        assert root_logger.handlers[0].level == logging.WARNING

        # Quiet mode removes CompactProgressFilter
        assert not any(
            isinstance(f, CompactProgressFilter)
            for f in root_logger.handlers[0].filters
        )

    def test_detects_preexisting_console_handler(self) -> None:
        """Test that a StreamHandler attached outside of get_console_logger
        is still recognised, so a second one is not added."""
        root_logger = logging.getLogger("medpipe")
        root_logger.addHandler(logging.StreamHandler(sys.stdout))

        get_console_logger(verbose="debug")

        assert len(root_logger.handlers) == 1
        assert root_logger.handlers[0].level == logging.DEBUG

    def test_file_handler_alone_does_not_count_as_console_handler(
        self, tmp_path: Path
    ) -> None:
        """Test that a FileHandler alone does not satisfy has_console_handler,
        since FileHandler is a StreamHandler subclass that must be excluded."""
        root_logger = logging.getLogger("medpipe")
        root_logger.addHandler(logging.FileHandler(tmp_path / "run.log"))

        get_console_logger(verbose="info")

        assert len(root_logger.handlers) == 2
        console_handlers = [
            h
            for h in root_logger.handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)
        ]
        assert len(console_handlers) == 1
