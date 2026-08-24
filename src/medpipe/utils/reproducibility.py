"""
medpipe.utils.reproducibility
-----------------------------

Provides utilities for capturing runtime environment metadata,
hashing configurations and datasets, and managing artifact serialization
for experiment reproducibility.
"""

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union


def compute_file_hash(file_path: Union[str, Path], algorithm: str = "sha256") -> str:
    """Compute the cryptographic hash of a file on disk.

    Parameters
    ----------
    file_path : str or Path
        Path to the file to be hashed.
    algorithm : str, default="sha256"
        Hashing algorithm to use (e.g., 'md5', 'sha256').

    Returns
    -------
    str
        Hexadecimal hash string representing the file contents.

    Raises
    ------
    FileNotFoundError
        If the specified `file_path` does not exist.

    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found for hashing: {path}")

    hasher = hashlib.new(algorithm)
    # Read in chunks to prevent memory overload on large datasets
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def compute_config_hash(config: Dict[str, Any]) -> str:
    """Generate a deterministic SHA-256 hash from a configuration dictionary.

    Parameters
    ----------
    config : dict
        Fully resolved configuration dictionary.

    Returns
    -------
    str
        The SHA-256 hash representing the config state.

    """
    # Canonicalize dictionary to ensure key ordering does not alter hash
    encoded_config = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded_config).hexdigest()


def get_git_commit_hash() -> Optional[str]:
    """Retrieve the current Git commit hash of the repository.

    Returns
    -------
    str or None
        The 40-character SHA-1 commit hash, or None if Git is not installed
        or the code is not running inside a Git repository.

    """
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return git_hash.decode("ascii").strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def capture_environment_state(
    config: Dict[str, Any],
    dataset_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Collect system, Python, package, and data metadata for reproducibility.

    Parameters
    ----------
    config : dict
        Fully resolved configuration to save.
    dataset_path : str or Path, optional
        Path to the dataset file to hash. If None, dataset_hash is omitted.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing environment and runtime metadata.

    """
    env_state: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config_hash": compute_config_hash(config),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit_hash": get_git_commit_hash(),
        "dataset_hash": None,
    }

    if dataset_path is not None:
        try:
            env_state["dataset_hash"] = compute_file_hash(dataset_path)
        except FileNotFoundError:
            env_state["dataset_hash"] = "FILE_NOT_FOUND"

    return env_state


class ArtifactManager:
    """Manages the creation of experiment artifact directories and
    metadata persistence.

    Parameters
    ----------
    base_artifact_dir : str or Path, default="artifacts"
        Root directory where experiment runs will be saved.

    Attributes
    ----------
    base_dir : Path
        Resolved base directory path.

    Methods
    -------
    create_run_directory()
        Create a new versioned directory for storing experiment
        artifacts.
    save_json(obj, destination_dir)
        Saves object as a JSON file.
    saved_resolved_config(config, destination_dir)
        Persist the resolved configuration dictionary as a JSON file.
    save_env_state(destination_dir, dataset_path)
        Capture and persist the environment state to `env_state.json`.

    """

    def __init__(self, base_artifact_dir: Union[str, Path] = "artifacts") -> None:
        self.base_dir = Path(base_artifact_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_next_version_number(self) -> int:
        """Scan the base directory and return the next auto-incremented run version.

        Returns
        -------
        int
            Next available experiment version number (e.g., 1 for 'v1').

        """
        existing_versions = []
        for path in self.base_dir.iterdir():
            if path.is_dir() and path.name.startswith("v"):
                version_str = path.name[1:].split("_")[
                    0
                ]  # Extracts number from v1_hash
                if version_str.isdigit():
                    existing_versions.append(int(version_str))

        return max(existing_versions, default=0) + 1

    def create_run_directory(self) -> Path:
        """Create a new versioned directory for storing experiment artifacts.

        Returns
        -------
        Path
            Path to the newly created run directory (e.g., `artifacts/v1`).

        """
        version_num = self._get_next_version_number()
        dir_name = f"v{version_num}"

        run_dir = self.base_dir / dir_name
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def save_json(
        self, obj: Any, destination_dir: Union[str, Path], filename: str
    ) -> Path:
        """Saves object as a JSON file.

        The destination directory is created if it does not exist.

        Parameters
        ----------
        obj : Any
            Object to save.
        destination_dir : str or Path
            Directory where the file will be written.
        filename : str
            Name of the file to save.

        Returns
        -------
        Path
            Path to the saved JSON file.

        """
        dest_dir = Path(destination_dir)
        dest_dir.mkdir(exist_ok=True, parents=True)

        dest = dest_dir / filename
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=4, default=str)
        return dest

    def save_resolved_config(
        self, config: Dict[str, Any], destination_dir: Union[str, Path]
    ) -> Path:
        """Persist the resolved configuration dictionary as a JSON file.

        Parameters
        ----------
        config : dict
            Fully resolved configuration to save.
        destination_dir : str or Path
            Directory where `resolved_config.json` will be written.

        Returns
        -------
        Path
            Path to the saved JSON file.

        """
        return self.save_json(config, destination_dir, "resolved_config.json")

    def save_env_state(
        self,
        destination_dir: Union[str, Path],
        config: Dict[str, Any],
        dataset_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Capture and persist the environment state to `env_state.json`.

        Parameters
        ----------
        destination_dir : str or Path
            Directory where `env_state.json` will be written.
        config : dict
            Fully resolved configuration dictionary.
        dataset_path : str or Path, optional
            Path to the dataset file to hash and log.

        Returns
        -------
        Path
            Path to the saved JSON file.

        """
        env_state = capture_environment_state(config, dataset_path=dataset_path)
        return self.save_json(env_state, destination_dir, "env_state.json")
