"""
Tests for the add_file_handler function of the medpipe.utils.logger module.
"""

import logging
import re
from pathlib import Path

from medpipe.utils.logger import add_file_handler, get_console_logger


class TestAddFileHandler:
    """Test class for the add_file_handler function"""

    def test_creates_file_and_nested_directory(self, tmp_path: Path) -> None:
        """Test that the file handler correctly creates nested parent directories."""
        logger = get_console_logger()

        nested_log_dir = tmp_path / "artifacts" / "v1"
        add_file_handler(logger, log_dir=nested_log_dir, filename="test.log")

        expected_file = nested_log_dir / "test.log"
        assert expected_file.exists()
        assert len(logger.handlers) == 2  # StreamHandler + FileHandler

    def test_accepts_string_log_dir(self, tmp_path: Path) -> None:
        """Test that log_dir may be passed as a plain string as well as a Path."""
        logger = get_console_logger()
        log_dir = str(tmp_path / "string_dir")

        add_file_handler(logger, log_dir=log_dir, filename="test.log")

        assert (tmp_path / "string_dir" / "test.log").exists()

    def test_default_filename_and_level(self, tmp_path: Path) -> None:
        """Test that omitting filename and level falls back to
        'execution.log' at DEBUG threshold."""
        logger = get_console_logger()

        add_file_handler(logger, log_dir=tmp_path)

        assert (tmp_path / "execution.log").exists()
        file_handlers = [
            h for h in logger.handlers if isinstance(h, logging.FileHandler)
        ]
        assert len(file_handlers) == 1
        assert file_handlers[0].level == logging.DEBUG

    def test_writes_unfiltered_logs(self, tmp_path: Path) -> None:
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
        with open(log_file, encoding="utf-8") as f:
            log_contents = f.read()

        # File should contain all unfiltered logs
        assert "DEBUG" in log_contents
        assert "Debug diagnostic info." in log_contents
        assert "Filtered dataset from 11 down to 11 required columns." in log_contents
        assert "WARNING" in log_contents
        assert "Warning message." in log_contents

    def test_log_line_format_includes_module_and_line_number(
        self, tmp_path: Path
    ) -> None:
        """Test that the file handler's formatter includes the logger name,
        module, and line number as documented."""
        logger = get_console_logger()

        add_file_handler(logger, log_dir=tmp_path, filename="format.log")
        logger.info("hello world")

        log_contents = (tmp_path / "format.log").read_text(encoding="utf-8")

        assert re.search(
            r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} \| medpipe \| INFO \| "
            r"test_file_handler:\d+ \| hello world",
            log_contents,
        )

    def test_multiple_handlers_different_levels(self, tmp_path: Path) -> None:
        """Test attaching multiple file handlers for different threshold levels."""
        logger = get_console_logger()
        log_dir = tmp_path / "v3"

        add_file_handler(
            logger, log_dir=log_dir, filename="debug.log", level=logging.DEBUG
        )
        add_file_handler(
            logger, log_dir=log_dir, filename="errors.log", level=logging.ERROR
        )

        assert len(logger.handlers) == 3

        logger.error("System failure")

        with (
            open(log_dir / "debug.log") as f1,
            open(log_dir / "errors.log") as f2,
        ):
            assert "System failure" in f1.read()
            assert "System failure" in f2.read()

    def test_duplicate_handler_for_same_path_is_not_added(self, tmp_path: Path) -> None:
        """Test that calling add_file_handler twice for the same resolved
        path does not attach a second FileHandler."""
        logger = get_console_logger()

        add_file_handler(logger, log_dir=tmp_path, filename="run.log")
        add_file_handler(logger, log_dir=tmp_path, filename="run.log")

        file_handlers = [
            h for h in logger.handlers if isinstance(h, logging.FileHandler)
        ]
        assert len(file_handlers) == 1

    def test_different_filenames_in_same_directory_both_attach(
        self, tmp_path: Path
    ) -> None:
        """Test that distinct filenames in the same directory are treated as
        distinct handlers, not deduplicated."""
        logger = get_console_logger()

        add_file_handler(logger, log_dir=tmp_path, filename="a.log")
        add_file_handler(logger, log_dir=tmp_path, filename="b.log")

        file_handlers = [
            h for h in logger.handlers if isinstance(h, logging.FileHandler)
        ]
        assert len(file_handlers) == 2

    def test_sub_logger_attaches_handler_to_top_level_logger(
        self, tmp_path: Path
    ) -> None:
        """Test that passing a sub-logger still attaches the FileHandler to
        the top-level 'medpipe' logger, not the sub-logger itself."""
        get_console_logger()
        sub_logger = logging.getLogger("medpipe.pipeline.orchestrator")

        add_file_handler(sub_logger, log_dir=tmp_path, filename="sub.log")

        root_logger = logging.getLogger("medpipe")
        assert any(
            isinstance(h, logging.FileHandler) for h in root_logger.handlers
        )
        assert not any(
            isinstance(h, logging.FileHandler) for h in sub_logger.handlers
        )
