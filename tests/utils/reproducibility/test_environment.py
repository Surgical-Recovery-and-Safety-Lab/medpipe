"""
Tests for the get_git_commit_hash and capture_environment_state functions
of the medpipe.utils.reproducibility module.
"""

import subprocess
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from medpipe.utils.reproducibility import (
    capture_environment_state,
    compute_config_hash,
    get_git_commit_hash,
)


class TestGetGitCommitHash:
    """Tests for the get_git_commit_hash utility function."""

    @patch("subprocess.check_output")
    def test_get_git_commit_hash_success(self, mock_check_output) -> None:
        """Test git hash retrieval when running inside a valid repository."""
        mock_check_output.return_value = b"abcdef1234567890\n"

        assert get_git_commit_hash() == "abcdef1234567890"

    @patch("subprocess.check_output")
    def test_get_git_commit_hash_calls_expected_command(
        self, mock_check_output
    ) -> None:
        """Test that the correct git subcommand is invoked with stderr
        suppressed."""
        mock_check_output.return_value = b"abcdef1234567890\n"

        get_git_commit_hash()

        mock_check_output.assert_called_once_with(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )

    @patch("subprocess.check_output")
    def test_get_git_commit_hash_failure(self, mock_check_output) -> None:
        """Test git hash fallback when Git is unavailable or not in a repo."""
        mock_check_output.side_effect = subprocess.CalledProcessError(1, "git")

        assert get_git_commit_hash() is None

    @patch("subprocess.check_output")
    def test_get_git_commit_hash_git_executable_missing(
        self, mock_check_output
    ) -> None:
        """Test git hash fallback when git executable is not installed on
        system PATH."""
        mock_check_output.side_effect = FileNotFoundError("git binary not found")

        assert get_git_commit_hash() is None


class TestCaptureEnvironmentState:
    """Tests for the capture_environment_state utility function."""

    def test_capture_environment_state_no_dataset(self, sample_config: dict) -> None:
        """Test environment state capture without specifying a dataset path."""
        state = capture_environment_state(sample_config)

        assert "timestamp_utc" in state
        assert "config_hash" in state
        assert "python_version" in state
        assert "platform" in state
        assert "git_commit_hash" in state
        assert state["dataset_hash"] is None

    def test_capture_environment_state_config_hash_matches(
        self, sample_config: dict
    ) -> None:
        """Test that the embedded config_hash matches compute_config_hash
        applied to the same config."""
        state = capture_environment_state(sample_config)

        assert state["config_hash"] == compute_config_hash(sample_config)

    def test_capture_environment_state_timestamp_is_valid_iso8601(
        self, sample_config: dict
    ) -> None:
        """Test that timestamp_utc parses as a valid ISO-8601 timestamp."""
        state = capture_environment_state(sample_config)

        # Should not raise
        datetime.fromisoformat(state["timestamp_utc"])

    def test_capture_environment_state_python_version_and_platform_are_strings(
        self, sample_config: dict
    ) -> None:
        """Test that python_version and platform are non-empty strings."""
        state = capture_environment_state(sample_config)

        assert isinstance(state["python_version"], str) and state["python_version"]
        assert isinstance(state["platform"], str) and state["platform"]

    @patch("medpipe.utils.reproducibility.get_git_commit_hash")
    def test_capture_environment_state_uses_git_commit_hash(
        self, mock_get_git_commit_hash, sample_config: dict
    ) -> None:
        """Test that git_commit_hash in the resulting state is sourced from
        get_git_commit_hash, isolated from the real git environment."""
        mock_get_git_commit_hash.return_value = "deadbeef"

        state = capture_environment_state(sample_config)

        assert state["git_commit_hash"] == "deadbeef"
        mock_get_git_commit_hash.assert_called_once_with()

    @patch("medpipe.utils.reproducibility.get_git_commit_hash", return_value=None)
    def test_capture_environment_state_git_commit_hash_none_outside_repo(
        self, mock_get_git_commit_hash, sample_config: dict
    ) -> None:
        """Test that git_commit_hash is None when get_git_commit_hash
        reports no repository."""
        state = capture_environment_state(sample_config)

        assert state["git_commit_hash"] is None

    def test_capture_environment_state_with_dataset(
        self, sample_config: dict, sample_dataset: Path
    ) -> None:
        """Test environment state capture with a valid dataset file."""
        state = capture_environment_state(
            sample_config,
            dataset_path=sample_dataset,
        )

        assert isinstance(state["dataset_hash"], str)
        assert state["dataset_hash"] != "FILE_NOT_FOUND"

    def test_capture_environment_state_missing_dataset(
        self,
        tmp_path: Path,
        sample_config: dict,
    ) -> None:
        """Test environment state capture gracefully handles missing datasets."""
        missing_file = tmp_path / "does_not_exist.csv"
        state = capture_environment_state(
            sample_config,
            dataset_path=missing_file,
        )

        assert state["dataset_hash"] == "FILE_NOT_FOUND"
