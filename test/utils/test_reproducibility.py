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


def test_compute_file_hash_success(sample_dataset):
    """Test that a file hashes correctly and consistently."""
    hash_val = compute_file_hash(sample_dataset)
    assert isinstance(hash_val, str)
    assert len(hash_val) == 64  # SHA-256 length

    # Verify deterministic nature
    hash_val_2 = compute_file_hash(sample_dataset)
    assert hash_val == hash_val_2


def test_compute_file_hash_not_found():
    """Test that hashing a non-existent file raises the appropriate error."""
    with pytest.raises(FileNotFoundError):
        compute_file_hash("non_existent_file.csv")


def test_compute_config_hash_deterministic(sample_config):
    """Test that dictionaries with differently ordered keys produce the same hash."""
    # Create an identical config but insert keys in a different order
    shuffled_config = {
        "data": {"test_size": 0.2, "target": "outcome"},
        "model": {"params": {"n_estimators": 100}, "name": "RandomForest"},
    }

    hash1 = compute_config_hash(sample_config)
    hash2 = compute_config_hash(shuffled_config)

    assert isinstance(hash1, str)
    assert len(hash1) == 64
    assert hash1 == hash2


# --- Tests for Environment Capture ---


@patch("subprocess.check_output")
def test_get_git_commit_hash_success(mock_check_output):
    """Test git hash retrieval when running inside a valid repository."""
    mock_check_output.return_value = b"abcdef1234567890\n"
    assert get_git_commit_hash() == "abcdef1234567890"


@patch("subprocess.check_output")
def test_get_git_commit_hash_failure(mock_check_output):
    """Test git hash fallback when Git is unavailable or not in a repo."""
    mock_check_output.side_effect = subprocess.CalledProcessError(1, "git")
    assert get_git_commit_hash() is None


def test_capture_environment_state_no_dataset(sample_config):
    """Test environment state capture without specifying a dataset path."""
    state = capture_environment_state(sample_config)

    assert "timestamp_utc" in state
    assert "config_hash" in state
    assert "python_version" in state
    assert "platform" in state
    assert "git_commit_hash" in state
    assert state["dataset_hash"] is None


def test_capture_environment_state_with_dataset(sample_config, sample_dataset):
    """Test environment state capture with a valid dataset file."""
    state = capture_environment_state(sample_config, dataset_path=sample_dataset)

    assert isinstance(state["dataset_hash"], str)
    assert state["dataset_hash"] != "FILE_NOT_FOUND"


def test_capture_environment_state_missing_dataset(tmp_path, sample_config):
    """Test environment state capture gracefully handles missing datasets."""
    missing_file = tmp_path / "does_not_exist.csv"
    state = capture_environment_state(sample_config, dataset_path=missing_file)

    assert state["dataset_hash"] == "FILE_NOT_FOUND"


def test_compute_file_hash_empty_file(tmp_path):
    """Test hashing a zero-byte empty file."""
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("")

    # Standard SHA-256 hash for an empty string
    expected_hash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    assert compute_file_hash(empty_file) == expected_hash


def test_compute_file_hash_directory_input(tmp_path):
    """Test that passing a directory path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        compute_file_hash(tmp_path)


def test_compute_file_hash_invalid_algorithm(sample_dataset):
    """Test that passing an invalid algorithm name raises ValueError."""
    with pytest.raises(ValueError):
        compute_file_hash(sample_dataset, algorithm="invalid_algo_name")


def test_compute_config_hash_empty_dict():
    """Test hashing an empty configuration dictionary."""
    hash_val = compute_config_hash({})
    assert isinstance(hash_val, str)
    assert len(hash_val) == 64


def test_compute_config_hash_non_json_serializable_types():
    """Test hashing configs containing non-native JSON types like Path and tuple."""
    config_with_custom_types = {
        "data_path": Path("/tmp/data.csv"),
        "dimensions": (100, 20),
    }
    hash_val = compute_config_hash(config_with_custom_types)
    assert isinstance(hash_val, str)
    assert len(hash_val) == 64


@patch("subprocess.check_output")
def test_get_git_commit_hash_git_executable_missing(mock_check_output):
    """Test git hash fallback when git executable is not installed on system PATH."""
    mock_check_output.side_effect = FileNotFoundError("git binary not found")
    assert get_git_commit_hash() is None


# --- Tests for ArtifactManager ---


def test_artifact_manager_initialization(tmp_path):
    """Test that the manager correctly creates the base directory."""
    base_dir = tmp_path / "test_artifacts"
    manager = ArtifactManager(base_artifact_dir=base_dir)

    assert manager.base_dir == base_dir
    assert base_dir.exists()
    assert base_dir.is_dir()


def test_artifact_manager_get_next_version(tmp_path):
    """Test the auto-increment logic for experiment versions."""
    manager = ArtifactManager(base_artifact_dir=tmp_path)

    # Should be 1 on empty directory
    assert manager._get_next_version_number() == 1

    # Manually create mock version directories
    (tmp_path / "v1").mkdir()
    (tmp_path / "v2").mkdir()

    # Should correctly identify 3 as the next available version
    assert manager._get_next_version_number() == 3


def test_artifact_manager_create_run_directory(tmp_path):
    """Test that the versioned run directory is generated correctly."""
    manager = ArtifactManager(base_artifact_dir=tmp_path)

    run_dir = manager.create_run_directory()

    assert run_dir.exists()
    assert run_dir.is_dir()
    assert run_dir.parent == tmp_path
    assert run_dir.name.startswith("v1")


def test_artifact_manager_save_resolved_config(tmp_path, sample_config):
    """Test persistence of the configuration dictionary to JSON."""
    manager = ArtifactManager(base_artifact_dir=tmp_path)
    run_dir = manager.create_run_directory()

    saved_path = manager.save_resolved_config(sample_config, run_dir)

    assert saved_path.exists()
    assert saved_path.name == "resolved_config.json"

    # Verify contents match exactly
    with open(saved_path, "r", encoding="utf-8") as f:
        loaded_config = json.load(f)
    assert loaded_config == sample_config


def test_artifact_manager_save_env_state(tmp_path, sample_config, sample_dataset):
    """Test persistence of environment metadata to JSON."""
    manager = ArtifactManager(base_artifact_dir=tmp_path)
    run_dir = manager.create_run_directory()

    saved_path = manager.save_env_state(
        run_dir, sample_config, dataset_path=sample_dataset
    )

    assert saved_path.exists()
    assert saved_path.name == "env_state.json"

    # Verify required keys are present in the written file
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


def test_artifact_manager_get_next_version_noisy_directory(tmp_path):
    """Test auto-incrementing version calculation with extraneous files and folders."""
    manager = ArtifactManager(base_artifact_dir=tmp_path)

    # Create non-version files and directories
    (tmp_path / "readme.txt").write_text("notes")
    (tmp_path / ".DS_Store").touch()
    (tmp_path / "v_invalid_folder").mkdir()
    (tmp_path / "v1.5_beta").mkdir()

    # Create valid version directories (including double digit to test numeric sorting)
    (tmp_path / "v2").mkdir()
    (tmp_path / "v10").mkdir()

    # Max valid version is 10, so next version should be 11
    assert manager._get_next_version_number() == 11


def test_artifact_manager_save_config_with_custom_types(tmp_path):
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
