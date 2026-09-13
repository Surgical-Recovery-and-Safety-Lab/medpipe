"""
Tests for the set_verbosity function of the medpipe.utils.logger module.
"""

import logging
import sys

import pytest

from medpipe.utils.logger import CompactProgressFilter, set_verbosity


@pytest.fixture
def console_handler() -> logging.Handler:
    """Attach a bare StreamHandler to the 'medpipe' logger, independent of
    get_console_logger, so set_verbosity can be tested in isolation."""
    handler = logging.StreamHandler(sys.stdout)
    logging.getLogger("medpipe").addHandler(handler)

    return handler


class TestSetVerbosity:
    """Test class for the set_verbosity function"""

    @pytest.mark.parametrize(
        "verbose, expected_level, expects_filter",
        [
            ("quiet", logging.WARNING, False),
            ("warning", logging.WARNING, False),
            (False, logging.WARNING, False),
            (0, logging.WARNING, False),
            (logging.WARNING, logging.WARNING, False),
            ("compact", logging.INFO, True),
            ("progress", logging.INFO, True),
            (1, logging.INFO, True),
            ("info", logging.INFO, False),
            ("detailed", logging.INFO, False),
            (True, logging.INFO, False),
            (logging.INFO, logging.INFO, False),
            (2, logging.INFO, False),
            ("debug", logging.DEBUG, False),
            (3, logging.DEBUG, False),
            (logging.DEBUG, logging.DEBUG, False),
        ],
    )
    def test_verbosity_modes(
        self,
        console_handler: logging.Handler,
        verbose: str | bool | int,
        expected_level: int,
        expects_filter: bool,
    ) -> None:
        """Test setting global verbosity across all valid mode representations."""
        set_verbosity(verbose)

        assert console_handler.level == expected_level
        has_filter = any(
            isinstance(f, CompactProgressFilter) for f in console_handler.filters
        )
        assert has_filter is expects_filter

    @pytest.mark.parametrize(
        "verbose, expected_level",
        [
            ("COMPACT", logging.INFO),
            ("Quiet", logging.WARNING),
            ("DEBUG", logging.DEBUG),
            ("Info", logging.INFO),
        ],
    )
    def test_string_mode_is_case_insensitive(
        self, console_handler: logging.Handler, verbose: str, expected_level: int
    ) -> None:
        """Test that mode strings are matched case-insensitively."""
        set_verbosity(verbose)

        assert console_handler.level == expected_level

    def test_unrecognised_string_mode_defaults_to_info(
        self, console_handler: logging.Handler
    ) -> None:
        """Test that an unrecognised string mode falls back to unfiltered INFO."""
        set_verbosity("nonsense_mode")

        assert console_handler.level == logging.INFO
        assert not any(
            isinstance(f, CompactProgressFilter) for f in console_handler.filters
        )

    def test_out_of_range_int_defaults_to_compact(
        self, console_handler: logging.Handler
    ) -> None:
        """Test that an integer outside the known mapping falls back to 'compact'."""
        set_verbosity(99)

        assert console_handler.level == logging.INFO
        assert any(
            isinstance(f, CompactProgressFilter) for f in console_handler.filters
        )

    def test_default_argument_is_compact(
        self, console_handler: logging.Handler
    ) -> None:
        """Test that calling set_verbosity with no arguments uses 'compact'."""
        set_verbosity()

        assert console_handler.level == logging.INFO
        assert any(
            isinstance(f, CompactProgressFilter) for f in console_handler.filters
        )

    def test_repeated_calls_do_not_stack_filters(
        self, console_handler: logging.Handler
    ) -> None:
        """Test that calling set_verbosity multiple times does not attach
        multiple CompactProgressFilter instances."""
        set_verbosity("compact")
        set_verbosity("compact")
        set_verbosity("compact")

        filter_count = sum(
            isinstance(f, CompactProgressFilter) for f in console_handler.filters
        )
        assert filter_count == 1

    def test_switching_away_from_compact_removes_filter(
        self, console_handler: logging.Handler
    ) -> None:
        """Test that switching from 'compact' to another mode detaches the filter."""
        set_verbosity("compact")
        assert any(
            isinstance(f, CompactProgressFilter) for f in console_handler.filters
        )

        set_verbosity("debug")

        assert not any(
            isinstance(f, CompactProgressFilter) for f in console_handler.filters
        )

    def test_file_handler_is_not_reconfigured(self, tmp_path) -> None:
        """Test that FileHandler instances on the logger are left untouched,
        since FileHandler is a StreamHandler subclass that must be excluded."""
        file_handler = logging.FileHandler(tmp_path / "run.log")
        file_handler.setLevel(logging.ERROR)
        logging.getLogger("medpipe").addHandler(file_handler)

        set_verbosity("debug")

        assert file_handler.level == logging.ERROR

    def test_no_handlers_does_not_raise(self) -> None:
        """Test that calling set_verbosity with no handlers attached is a no-op."""
        assert logging.getLogger("medpipe").handlers == []

        set_verbosity("debug")  # Should not raise

    def test_non_stream_handler_is_ignored(self) -> None:
        """Test that non-StreamHandler handlers (e.g. NullHandler) are left alone."""
        null_handler = logging.NullHandler()
        logging.getLogger("medpipe").addHandler(null_handler)

        set_verbosity("debug")  # Should not raise or mutate the NullHandler

        assert null_handler.level == logging.NOTSET
