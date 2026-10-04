"""
Tests for the ArtifactManager class of the medpipe.utils.reproducibility module.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from medpipe.utils.reproducibility import ArtifactManager


class TestArtifactManagerInit:
    """Tests for ArtifactManager.__init__."""

    def test_initialization(self, tmp_path: Path) -> None:
        """Test that the manager correctly creates the base directory."""
        base_dir = tmp_path / "test_artifacts"
        manager = ArtifactManager(base_artifact_dir=base_dir)

        assert manager.base_dir == base_dir
        assert base_dir.exists()
        assert base_dir.is_dir()

    def test_initialization_default_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that omitting base_artifact_dir defaults to './artifacts'
        relative to the current working directory."""
        monkeypatch.chdir(tmp_path)

        manager = ArtifactManager()

        assert manager.base_dir == Path("artifacts")
        assert (tmp_path / "artifacts").is_dir()

    def test_initialization_accepts_string_path(self, tmp_path: Path) -> None:
        """Test that base_artifact_dir may be passed as a plain string."""
        base_dir = str(tmp_path / "string_artifacts")
        manager = ArtifactManager(base_artifact_dir=base_dir)

        assert manager.base_dir == Path(base_dir)
        assert manager.base_dir.is_dir()

    def test_initialization_is_idempotent_for_existing_directory(
        self, tmp_path: Path
    ) -> None:
        """Test that constructing a second manager over the same directory
        does not raise, since mkdir uses exist_ok=True."""
        base_dir = tmp_path / "shared_artifacts"
        ArtifactManager(base_artifact_dir=base_dir)

        # Should not raise even though the directory already exists.
        second_manager = ArtifactManager(base_artifact_dir=base_dir)

        assert second_manager.base_dir == base_dir


class TestArtifactManagerGetNextVersionNumber:
    """Tests for ArtifactManager._get_next_version_number."""

    def test_get_next_version_empty_directory(self, tmp_path: Path) -> None:
        """Test next version calculation on an empty directory."""
        manager = ArtifactManager(base_artifact_dir=tmp_path)
        assert manager._get_next_version_number() == 1

    def test_get_next_version_existing_directories(self, tmp_path: Path) -> None:
        """Test the auto-increment logic with existing version directories."""
        manager = ArtifactManager(base_artifact_dir=tmp_path)
        (tmp_path / "v1").mkdir()
        (tmp_path / "v2").mkdir()

        assert manager._get_next_version_number() == 3

    def test_get_next_version_with_gap_uses_max_not_fill(self, tmp_path: Path) -> None:
        """Test that a gap in the version sequence is not backfilled — the
        next version is always max + 1."""
        manager = ArtifactManager(base_artifact_dir=tmp_path)
        (tmp_path / "v1").mkdir()
        (tmp_path / "v5").mkdir()

        assert manager._get_next_version_number() == 6

    def test_get_next_version_ignores_files_named_like_versions(
        self, tmp_path: Path
    ) -> None:
        """Test that a file (not a directory) named like a version is ignored."""
        manager = ArtifactManager(base_artifact_dir=tmp_path)
        (tmp_path / "v5").write_text("not a directory")

        assert manager._get_next_version_number() == 1

    def test_get_next_version_noisy_directory(self, tmp_path: Path) -> None:
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

    def test_create_run_directory(self, tmp_path: Path) -> None:
        """Test that the versioned run directory is generated correctly."""
        manager = ArtifactManager(base_artifact_dir=tmp_path)

        run_dir = manager.create_run_directory()

        assert run_dir.exists()
        assert run_dir.is_dir()
        assert run_dir.parent == tmp_path
        assert run_dir.name.startswith("v1")

    def test_create_run_directory_sequential_calls_increment(
        self, tmp_path: Path
    ) -> None:
        """Test that repeated calls produce sequentially incrementing
        run directories."""
        manager = ArtifactManager(base_artifact_dir=tmp_path)

        run_dir_1 = manager.create_run_directory()
        run_dir_2 = manager.create_run_directory()
        run_dir_3 = manager.create_run_directory()

        assert [d.name for d in (run_dir_1, run_dir_2, run_dir_3)] == [
            "v1",
            "v2",
            "v3",
        ]


class TestArtifactManagerSaveJson:
    """Tests for ArtifactManager.save_json."""

    def test_save_json_success(self, tmp_path: Path) -> None:
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

        with open(saved_path, encoding="utf-8") as f:
            loaded_data = json.load(f)

        assert loaded_data == payload

    def test_save_json_custom_types(self, tmp_path: Path) -> None:
        """Test save_json with non-native JSON types (Path, tuple) using
        default=str for anything json can't natively serialize."""
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

        with open(saved_path, encoding="utf-8") as f:
            loaded = json.load(f)

        # Path is not natively JSON-serializable, so default=str stringifies it.
        assert loaded["path_obj"] == "/var/log/medpipe"
        # tuples ARE natively serialized by json as arrays, independent of default=str.
        assert loaded["tuple_data"] == [1, 2, 3]

    def test_save_json_string_path_destination(self, tmp_path: Path) -> None:
        """Test save_json when passing the destination directory as a
        string rather than Path."""
        manager = ArtifactManager(base_artifact_dir=tmp_path)
        run_dir = manager.create_run_directory()

        data = {"status": "success"}
        saved_path = manager.save_json(data, str(run_dir), "output.json")

        assert saved_path.exists()
        assert saved_path.name == "output.json"

    def test_save_json_creates_missing_destination_directory(
        self, tmp_path: Path
    ) -> None:
        """Test that save_json creates a destination directory that does
        not exist yet, including nested parents."""
        manager = ArtifactManager(base_artifact_dir=tmp_path)
        nested_dir = tmp_path / "not_yet_created" / "nested"

        saved_path = manager.save_json({"a": 1}, nested_dir, "data.json")

        assert nested_dir.is_dir()
        assert saved_path.exists()

    def test_save_json_overwrites_existing_file(self, tmp_path: Path) -> None:
        """Test that saving to the same filename twice overwrites rather
        than errors or appends."""
        manager = ArtifactManager(base_artifact_dir=tmp_path)
        run_dir = manager.create_run_directory()

        manager.save_json({"version": 1}, run_dir, "state.json")
        saved_path = manager.save_json({"version": 2}, run_dir, "state.json")

        with open(saved_path, encoding="utf-8") as f:
            loaded = json.load(f)

        assert loaded == {"version": 2}


class TestArtifactManagerSaveResolvedConfig:
    """Tests for ArtifactManager.save_resolved_config."""

    def test_save_resolved_config(self, tmp_path: Path, sample_config: dict) -> None:
        """Test persistence of the configuration dictionary to JSON
        via save_resolved_config."""
        manager = ArtifactManager(base_artifact_dir=tmp_path)
        run_dir = manager.create_run_directory()

        saved_path = manager.save_resolved_config(sample_config, run_dir)

        assert saved_path.exists()
        assert saved_path.name == "resolved_config.json"

        with open(saved_path, encoding="utf-8") as f:
            loaded_config = json.load(f)
        assert loaded_config == sample_config

    def test_save_config_with_custom_types(self, tmp_path: Path) -> None:
        """Test saving a config containing non-standard serializable objects."""
        manager = ArtifactManager(base_artifact_dir=tmp_path)
        run_dir = manager.create_run_directory()

        complex_config = {
            "file_path": Path("/usr/bin/data.csv"),
            "tuple_param": (10, 20),
        }

        saved_path = manager.save_resolved_config(complex_config, run_dir)
        assert saved_path.exists()

        with open(saved_path, encoding="utf-8") as f:
            loaded = json.load(f)

        assert loaded["file_path"] == "/usr/bin/data.csv"
        assert loaded["tuple_param"] == [10, 20]


class TestArtifactManagerSaveTomlConfig:
    """Tests for ArtifactManager.save_toml_config."""

    def test_save_toml_config(self, tmp_path: Path) -> None:
        """Test that the original TOML file is copied verbatim into the
        destination directory as config.toml."""
        manager = ArtifactManager(base_artifact_dir=tmp_path)
        run_dir = manager.create_run_directory()

        source_file = tmp_path / "original_config.toml"
        source_file.write_text('[meta]\nproject_name = "demo"\n')

        saved_path = manager.save_toml_config(source_file, run_dir)

        assert saved_path.exists()
        assert saved_path.name == "config.toml"
        assert saved_path.read_text() == source_file.read_text()

    def test_save_toml_config_creates_missing_destination_directory(
        self, tmp_path: Path
    ) -> None:
        """Test that the destination directory is created if missing."""
        manager = ArtifactManager(base_artifact_dir=tmp_path)

        source_file = tmp_path / "original_config.toml"
        source_file.write_text('[meta]\nproject_name = "demo"\n')

        missing_dir = tmp_path / "nested" / "env"
        saved_path = manager.save_toml_config(source_file, missing_dir)

        assert missing_dir.exists()
        assert saved_path.exists()


class TestArtifactManagerSaveEnvState:
    """Tests for ArtifactManager.save_env_state."""

    def test_save_env_state(
        self, tmp_path: Path, sample_config: dict, sample_dataset: Path
    ) -> None:
        """Test persistence of environment metadata to JSON via save_env_state."""
        manager = ArtifactManager(base_artifact_dir=tmp_path)
        run_dir = manager.create_run_directory()

        saved_path = manager.save_env_state(
            run_dir, sample_config, dataset_path=sample_dataset
        )

        assert saved_path.exists()
        assert saved_path.name == "env_state.json"

        with open(saved_path, encoding="utf-8") as f:
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

    def test_save_env_state_without_dataset(
        self, tmp_path: Path, sample_config: dict
    ) -> None:
        """Test that save_env_state works with no dataset_path, leaving
        dataset_hash as None in the persisted file."""
        manager = ArtifactManager(base_artifact_dir=tmp_path)
        run_dir = manager.create_run_directory()

        saved_path = manager.save_env_state(run_dir, sample_config)

        with open(saved_path, encoding="utf-8") as f:
            loaded_state = json.load(f)

        assert loaded_state["dataset_hash"] is None

    @patch("medpipe.utils.reproducibility.capture_environment_state")
    def test_save_env_state_delegates_to_capture_environment_state(
        self, mock_capture, tmp_path: Path, sample_config: dict
    ) -> None:
        """Test that save_env_state forwards config and dataset_path to
        capture_environment_state and persists exactly what it returns."""
        mock_capture.return_value = {"stub": "state"}
        manager = ArtifactManager(base_artifact_dir=tmp_path)
        run_dir = manager.create_run_directory()
        dataset_path = tmp_path / "data.csv"

        saved_path = manager.save_env_state(
            run_dir, sample_config, dataset_path=dataset_path
        )

        mock_capture.assert_called_once_with(
            sample_config, dataset_path=dataset_path
        )
        with open(saved_path, encoding="utf-8") as f:
            assert json.load(f) == {"stub": "state"}
