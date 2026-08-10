"""
Tests for the medpipe.utils.reproducibility module.
"""

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from medpipe.utils.reproducibility import (
    ArtifactManager,
    capture_environment_state,
    compute_config_hash,
    compute_file_hash,
    get_git_commit_hash,
)


@pytest.fixture
def sample_config():
    """Fixture providing a standard mock configuration dictionary."""
    return {
        "model": {"name": "RandomForest", "params": {"n_estimators": 100}},
        "data": {"target": "outcome", "test_size": 0.2},
    }


@pytest.fixture
def sample_dataset(tmp_path):
    """Fixture creating a temporary dummy dataset file."""
    dataset_file = tmp_path / "dummy_data.csv"
    dataset_file.write_text("col1,col2,outcome\n1,2,1\n3,4,0")
    return dataset_file


# --- Tests for Hashing Utilities ---


class TestComputeFileHash:
    """Tests for the compute_file_hash utility function."""

    def test_compute_file_hash_success(self, sample_dataset):
        """Test that a file hashes correctly and consistently."""
        hash_val = compute_file_hash(sample_dataset)
        assert isinstance(hash_val, str)
        assert len(hash_val) == 64  # SHA-256 length

        hash_val_2 = compute_file_hash(sample_dataset)
        assert hash_val == hash_val_2

    def test_compute_file_hash_not_found(self):
        """Test that hashing a non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            compute_file_hash("non_existent_file.csv")

    def test_compute_file_hash_empty_file(self, tmp_path):
        """Test hashing a zero-byte empty file."""
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("")

        expected_hash = (
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        )
        assert compute_file_hash(empty_file) == expected_hash

    def test_compute_file_hash_directory_input(self, tmp_path):
        """Test that passing a directory path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            compute_file_hash(tmp_path)

    def test_compute_file_hash_invalid_algorithm(self, sample_dataset):
        """Test that passing an invalid algorithm name raises ValueError."""
        with pytest.raises(ValueError):
            compute_file_hash(sample_dataset, algorithm="invalid_algo_name")


class TestComputeConfigHash:
    """Tests for the compute_config_hash utility function."""

    def test_compute_config_hash_deterministic(self, sample_config):
        """Test that dictionaries with differently ordered keys produce the same hash."""
        shuffled_config = {
            "data": {"test_size": 0.2, "target": "outcome"},
            "model": {"params": {"n_estimators": 100}, "name": "RandomForest"},
        }

        hash1 = compute_config_hash(sample_config)
        hash2 = compute_config_hash(shuffled_config)

        assert isinstance(hash1, str)
        assert len(hash1) == 64
        assert hash1 == hash2

    def test_compute_config_hash_empty_dict(self):
        """Test hashing an empty configuration dictionary."""
        hash_val = compute_config_hash({})
        assert isinstance(hash_val, str)
        assert len(hash_val) == 64

    def test_compute_config_hash_non_json_serializable_types(self):
        """Test hashing configs containing non-native JSON types like
        Path and tuple."""
        config_with_custom_types = {
            "data_path": Path("/tmp/data.csv"),
            "dimensions": (100, 20),
        }
        hash_val = compute_config_hash(config_with_custom_types)
        assert isinstance(hash_val, str)
        assert len(hash_val) == 64


# --- Tests for Environment Capture ---


class TestGetGitCommitHash:
    """Tests for the get_git_commit_hash utility function."""

    @patch("subprocess.check_output")
    def test_get_git_commit_hash_success(self, mock_check_output):
        """Test git hash retrieval when running inside a valid repository."""
        mock_check_output.return_value = b"abcdef1234567890\n"
        assert get_git_commit_hash() == "abcdef1234567890"

    @patch("subprocess.check_output")
    def test_get_git_commit_hash_failure(self, mock_check_output):
        """Test git hash fallback when Git is unavailable or not in a repo."""
        mock_check_output.side_effect = subprocess.CalledProcessError(1, "git")
        assert get_git_commit_hash() is None

    @patch("subprocess.check_output")
    def test_get_git_commit_hash_git_executable_missing(self, mock_check_output):
        """Test git hash fallback when git executable is not installed on system PATH."""
        mock_check_output.side_effect = FileNotFoundError("git binary not found")
        assert get_git_commit_hash() is None


class TestCaptureEnvironmentState:
    """Tests for the capture_environment_state utility function."""

    def test_capture_environment_state_no_dataset(self, sample_config):
        """Test environment state capture without specifying a dataset path."""
        state = capture_environment_state(sample_config)

        assert "timestamp_utc" in state
        assert "config_hash" in state
        assert "python_version" in state
        assert "platform" in state
        assert "git_commit_hash" in state
        assert state["dataset_hash"] is None

    def test_capture_environment_state_with_dataset(
        self, sample_config, sample_dataset
    ):
        """Test environment state capture with a valid dataset file."""
        state = capture_environment_state(
            sample_config,
            dataset_path=sample_dataset,
        )

        assert isinstance(state["dataset_hash"], str)
        assert state["dataset_hash"] != "FILE_NOT_FOUND"

    def test_capture_environment_state_missing_dataset(
        self,
        tmp_path,
        sample_config,
    ):
        """Test environment state capture gracefully handles missing datasets."""
        missing_file = tmp_path / "does_not_exist.csv"
        state = capture_environment_state(
            sample_config,
            dataset_path=missing_file,
        )

        assert state["dataset_hash"] == "FILE_NOT_FOUND"


# --- Tests for ArtifactManager ---


class TestArtifactManagerInit:
    """Tests for ArtifactManager.__init__."""

    def test_initialization(self, tmp_path):
        """Test that the manager correctly creates the base directory."""
        base_dir = tmp_path / "test_artifacts"
        manager = ArtifactManager(base_artifact_dir=base_dir)

        assert manager.base_dir == base_dir
        assert base_dir.exists()
        assert base_dir.is_dir()


class TestArtifactManagerGetNextVersionNumber:
    """Tests for ArtifactManager._get_next_version_number."""

    def test_get_next_version_empty_directory(self, tmp_path):
        """Test next version calculation on an empty directory."""
        manager = ArtifactManager(base_artifact_dir=tmp_path)
        assert manager._get_next_version_number() == 1

    def test_get_next_version_existing_directories(self, tmp_path):
        """Test the auto-increment logic with existing version directories."""
        manager = ArtifactManager(base_artifact_dir=tmp_path)
        (tmp_path / "v1").mkdir()
        (tmp_path / "v2").mkdir()

        assert manager._get_next_version_number() == 3

    def test_get_next_version_noisy_directory(self, tmp_path):
        """Test auto-incrementing version calculation with extraneous
        files and folders."""
        manager = ArtifactManager(base_artifact_dir=tmp_path)

        (tmp_path / "readme.txt").write_text("notes")
        (tmp_path / ".DS_Store").touch()
        (tmp_path / "v_invalid_folder").mkdir()
        (tmp_path / "v1.5_beta").mkdir()

        (tmp_path / "v2").mkdir()
        (tmp_path / "v10").mkdir()

        assert manager._get_next_version_number() == 11


class TestArtifactManagerCreateRunDirectory:
    """Tests for ArtifactManager.create_run_directory."""

    def test_create_run_directory(self, tmp_path):
        """Test that the versioned run directory is generated correctly."""
        manager = ArtifactManager(base_artifact_dir=tmp_path)

        run_dir = manager.create_run_directory()

        assert run_dir.exists()
        assert run_dir.is_dir()
        assert run_dir.parent == tmp_path
        assert run_dir.name.startswith("v1")


class TestArtifactManagerSaveJson:
    """Tests for ArtifactManager.save_json."""

    def test_save_json_success(self, tmp_path):
        """Test saving a standard dictionary to disk using save_json."""
        manager = ArtifactManager(base_artifact_dir=tmp_path)
        run_dir = manager.create_run_directory()

        payload = {
            "metrics": {"auc": 0.92, "accuracy": 0.88},
            "model_name": "LogisticRegression",
            "iterations": [1, 2, 3],
        }

        saved_path = manager.save_json(payload, run_dir, "eval_results.json")

        assert isinstance(saved_path, Path)
        assert saved_path.exists()
        assert saved_path.name == "eval_results.json"
        assert saved_path.parent == run_dir

        with open(saved_path, "r", encoding="utf-8") as f:
            loaded_data = json.load(f)

        assert loaded_data == payload

    def test_save_json_custom_types(self, tmp_path):
        """Test save_json with non-native JSON types (Path, tuple, set)
        using default=str."""
        manager = ArtifactManager(base_artifact_dir=tmp_path)
        run_dir = manager.create_run_directory()

        custom_payload = {
            "path_obj": Path("/var/log/medpipe"),
            "tuple_data": (1, 2, 3),
        }

        saved_path = manager.save_json(
            custom_payload,
            run_dir,
            "custom_types.json",
        )
        assert saved_path.exists()

        with open(saved_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)

        assert loaded["path_obj"] == "/var/log/medpipe"
        assert loaded["tuple_data"] == [1, 2, 3] or loaded["tuple_data"] == "(1, 2, 3)"

    def test_save_json_string_path_destination(self, tmp_path):
        """Test save_json when passing the destination directory as a
        string rather than Path."""
        manager = ArtifactManager(base_artifact_dir=tmp_path)
        run_dir = manager.create_run_directory()

        data = {"status": "success"}
        saved_path = manager.save_json(data, str(run_dir), "output.json")

        assert saved_path.exists()
        assert saved_path.name == "output.json"


class TestArtifactManagerSaveResolvedConfig:
    """Tests for ArtifactManager.save_resolved_config."""

    def test_save_resolved_config(self, tmp_path, sample_config):
        """Test persistence of the configuration dictionary to JSON
        via save_resolved_config."""
        manager = ArtifactManager(base_artifact_dir=tmp_path)
        run_dir = manager.create_run_directory()

        saved_path = manager.save_resolved_config(sample_config, run_dir)

        assert saved_path.exists()
        assert saved_path.name == "resolved_config.json"

        with open(saved_path, "r", encoding="utf-8") as f:
            loaded_config = json.load(f)
        assert loaded_config == sample_config

    def test_save_config_with_custom_types(self, tmp_path):
        """Test saving a config containing non-standard serializable objects."""
        manager = ArtifactManager(base_artifact_dir=tmp_path)
        run_dir = manager.create_run_directory()

        complex_config = {
            "file_path": Path("/usr/bin/data.csv"),
            "tuple_param": (10, 20),
        }

        saved_path = manager.save_resolved_config(complex_config, run_dir)
        assert saved_path.exists()

        with open(saved_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)

        assert loaded["file_path"] == "/usr/bin/data.csv"
        assert loaded["tuple_param"] == [10, 20]


class TestArtifactManagerSaveEnvState:
    """Tests for ArtifactManager.save_env_state."""

    def test_save_env_state(self, tmp_path, sample_config, sample_dataset):
        """Test persistence of environment metadata to JSON via save_env_state."""
        manager = ArtifactManager(base_artifact_dir=tmp_path)
        run_dir = manager.create_run_directory()

        saved_path = manager.save_env_state(
            run_dir, sample_config, dataset_path=sample_dataset
        )

        assert saved_path.exists()
        assert saved_path.name == "env_state.json"

        with open(saved_path, "r", encoding="utf-8") as f:
            loaded_state = json.load(f)

        expected_keys = {
            "timestamp_utc",
            "config_hash",
            "python_version",
            "platform",
            "git_commit_hash",
            "dataset_hash",
        }
        assert expected_keys.issubset(loaded_state.keys())
        assert loaded_state["dataset_hash"] != "FILE_NOT_FOUND"
