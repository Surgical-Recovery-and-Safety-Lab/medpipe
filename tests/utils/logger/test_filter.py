"""
Tests for the CompactProgressFilter class of the medpipe.utils.logger module.
"""

import logging

import pytest

from medpipe.utils.logger import CompactProgressFilter


def _make_record(level: int, msg: str) -> logging.LogRecord:
    """Build a bare LogRecord for exercising the filter directly."""
    return logging.LogRecord(
        name="medpipe",
        level=level,
        pathname="module.py",
        lineno=1,
        msg=msg,
        args=(),
        exc_info=None,
    )


class TestCompactProgressFilter:
    """Test class for the CompactProgressFilter class"""

    def test_milestone_info_message_passes(self) -> None:
        """Test that an INFO message containing a progress keyword passes."""
        progress_filter = CompactProgressFilter()
        record = _make_record(logging.INFO, "Step 1/3: Ingesting and splitting dataset.")

        assert progress_filter.filter(record) is True

    def test_non_milestone_info_message_blocked(self) -> None:
        """Test that an INFO message without a progress keyword is filtered out."""
        progress_filter = CompactProgressFilter()
        record = _make_record(
            logging.INFO, "Filtered dataset from 11 down to 11 required columns."
        )

        assert progress_filter.filter(record) is False

    def test_non_milestone_debug_message_blocked(self) -> None:
        """Test that a DEBUG message without a progress keyword is filtered out."""
        progress_filter = CompactProgressFilter()
        record = _make_record(logging.DEBUG, "Fitting fold 3 of 5.")

        assert progress_filter.filter(record) is False

    def test_milestone_debug_message_passes(self) -> None:
        """Test that a DEBUG message containing a progress keyword still passes,
        since keyword matching is independent of level for sub-WARNING records."""
        progress_filter = CompactProgressFilter()
        record = _make_record(logging.DEBUG, "Ingesting data from cohort_a.csv")

        assert progress_filter.filter(record) is True

    @pytest.mark.parametrize(
        "level", [logging.WARNING, logging.ERROR, logging.CRITICAL]
    )
    def test_warning_and_above_always_pass(self, level: int) -> None:
        """Test that WARNING, ERROR, and CRITICAL messages pass regardless
        of their text content."""
        progress_filter = CompactProgressFilter()
        record = _make_record(level, "Some message with no keyword.")

        assert progress_filter.filter(record) is True

    @pytest.mark.parametrize(
        "keyword",
        CompactProgressFilter.PROGRESS_KEYWORDS,
    )
    def test_each_progress_keyword_passes(self, keyword: str) -> None:
        """Test that every declared progress keyword is individually
        recognized by the filter."""
        progress_filter = CompactProgressFilter()
        record = _make_record(logging.INFO, f"{keyword} some detail")

        assert progress_filter.filter(record) is True

    def test_empty_message_blocked(self) -> None:
        """Test that an empty INFO message is filtered out."""
        progress_filter = CompactProgressFilter()
        record = _make_record(logging.INFO, "")

        assert progress_filter.filter(record) is False

    def test_message_with_format_args_is_evaluated(self) -> None:
        """Test that %-style message args are resolved before keyword matching."""
        progress_filter = CompactProgressFilter()
        record = logging.LogRecord(
            name="medpipe",
            level=logging.INFO,
            pathname="module.py",
            lineno=1,
            msg="Ingesting data from %s",
            args=("cohort_a.csv",),
            exc_info=None,
        )

        assert progress_filter.filter(record) is True
