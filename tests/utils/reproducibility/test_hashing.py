"""
Tests for the compute_file_hash and compute_config_hash functions of the
medpipe.utils.reproducibility module.
"""

import hashlib
from pathlib import Path

import pytest

from medpipe.utils.reproducibility import compute_config_hash, compute_file_hash


class TestComputeFileHash:
    """Tests for the compute_file_hash utility function."""

    def test_compute_file_hash_success(self, sample_dataset: Path) -> None:
        """Test that a file hashes correctly and consistently."""
        hash_val = compute_file_hash(sample_dataset)
        assert isinstance(hash_val, str)
        assert len(hash_val) == 64  # SHA-256 length

        hash_val_2 = compute_file_hash(sample_dataset)
        assert hash_val == hash_val_2

    def test_compute_file_hash_known_value(self, sample_dataset: Path) -> None:
        """Test the hash against an independently computed reference value."""
        expected = hashlib.sha256(sample_dataset.read_bytes()).hexdigest()

        assert compute_file_hash(sample_dataset) == expected

    def test_compute_file_hash_not_found(self) -> None:
        """Test that hashing a non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            compute_file_hash("non_existent_file.csv")

    def test_compute_file_hash_empty_file(self, tmp_path: Path) -> None:
        """Test hashing a zero-byte empty file."""
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("")

        expected_hash = (
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        )
        assert compute_file_hash(empty_file) == expected_hash

    def test_compute_file_hash_directory_input(self, tmp_path: Path) -> None:
        """Test that passing a directory path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            compute_file_hash(tmp_path)

    def test_compute_file_hash_invalid_algorithm(self, sample_dataset: Path) -> None:
        """Test that passing an invalid algorithm name raises ValueError."""
        with pytest.raises(ValueError):
            compute_file_hash(sample_dataset, algorithm="invalid_algo_name")

    def test_compute_file_hash_non_default_algorithm(
        self, sample_dataset: Path
    ) -> None:
        """Test that the algorithm parameter is actually honoured, using md5
        as a non-default algorithm."""
        hash_val = compute_file_hash(sample_dataset, algorithm="md5")

        assert len(hash_val) == 32  # MD5 length
        assert hash_val == hashlib.md5(sample_dataset.read_bytes()).hexdigest()

    def test_compute_file_hash_multi_chunk_file(self, tmp_path: Path) -> None:
        """Test hashing a file larger than the internal 64KiB read chunk to
        exercise the chunked-reading loop across multiple iterations."""
        large_file = tmp_path / "large.bin"
        content = b"medpipe-reproducibility-chunk-test" * 20_000  # > 65536 bytes
        large_file.write_bytes(content)

        assert len(content) > 65536
        assert compute_file_hash(large_file) == hashlib.sha256(content).hexdigest()


class TestComputeConfigHash:
    """Tests for the compute_config_hash utility function."""

    def test_compute_config_hash_deterministic(self, sample_config: dict) -> None:
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

    def test_compute_config_hash_empty_dict(self) -> None:
        """Test hashing an empty configuration dictionary against a known
        reference value."""
        hash_val = compute_config_hash({})

        assert hash_val == (
            "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"
        )

    def test_compute_config_hash_different_configs_differ(
        self, sample_config: dict
    ) -> None:
        """Test that configs with different content produce different hashes."""
        other_config = {
            "model": {"name": "RandomForest", "params": {"n_estimators": 200}},
            "data": {"target": "outcome", "test_size": 0.2},
        }

        assert compute_config_hash(sample_config) != compute_config_hash(other_config)

    def test_compute_config_hash_non_json_serializable_types(self) -> None:
        """Test hashing configs containing non-native JSON types like
        Path and tuple."""
        config_with_custom_types = {
            "data_path": Path("/tmp/data.csv"),
            "dimensions": (100, 20),
        }
        hash_val = compute_config_hash(config_with_custom_types)
        assert isinstance(hash_val, str)
        assert len(hash_val) == 64
