"""
I/O utilities module.

This module provides helper functions for reading from and writing to files,
handling various common I/O tasks.

"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Callable, Dict, List, cast

import pandas as pd

from .config import MedpipeConfig
from .exceptions import file_checks


class DataLoaderRegistry:
    """Registry managing file extension mappings to DataFrame reader functions."""

    _registry: Dict[str, Callable[..., Any]] = {
        ".csv": pd.read_csv,
        ".tsv": lambda filepath, **kwargs: pd.read_csv(filepath, sep="\t", **kwargs),
        ".txt": pd.read_csv,
        ".parquet": pd.read_parquet,
        ".pq": pd.read_parquet,
        ".feather": pd.read_feather,
        ".xlsx": pd.read_excel,
        ".xls": pd.read_excel,
        ".json": pd.read_json,
        ".jsonl": lambda filepath, **kwargs: pd.read_json(
            filepath, lines=True, **kwargs
        ),
        ".pkl": pd.read_pickle,
        ".pickle": pd.read_pickle,
    }

    @classmethod
    def register(cls, extension: str) -> Callable[[Callable], Callable]:
        """Decorator or function to register a custom file reader for an extension."""

        def decorator(func: Callable[..., pd.DataFrame]) -> Callable[..., pd.DataFrame]:
            ext = cls._normalize_ext(extension)
            cls._registry[ext] = func
            return func

        return decorator

    @classmethod
    def get(cls, extension: str) -> Callable[..., pd.DataFrame]:
        """Retrieve the reader function for a given file extension."""
        ext = cls._normalize_ext(extension)
        if ext not in cls._registry:
            raise ValueError(
                f"Unsupported file extension '{ext}'. Registered extensions: {cls.list_registered()}"
            )
        return cls._registry[ext]

    @classmethod
    def list_registered(cls) -> List[str]:
        """List all supported file extensions."""
        return list(cls._registry.keys())

    @staticmethod
    def _normalize_ext(extension: str) -> str:
        """Ensure extensions are lowercased and prefixed with a dot."""
        ext = extension.lower().strip()
        return ext if ext.startswith(".") else f".{ext}"


def register_data_loader(extension: str, loader_func: Callable[..., Any]) -> None:
    """Register a custom file reader function for a specific file extension.

    Parameters
    ----------
    extension : str
        File extension starting with a dot (e.g., '.feather', '.rds').
    loader_func : Callable[..., Any]
        Function accepting a file path and returning a Pandas DataFrame.

    """
    DataLoaderRegistry.register(extension)(loader_func)


def load_data(data_file: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Reads a supported data file and returns its contents as a DataFrame.

    Parameters
    ----------
    data_file : str or Path
        Path to the data file to load.
    **kwargs : Any
        Additional keyword arguments passed to the underlying Pandas reader function
        (e.g., `sheet_name` for Excel, `sep` for CSV).

    Returns
    -------
    data : pd.DataFrame
        Loaded tabular data.

    Raises
    ------
    TypeError
        If data_file is not a str or Path.
    FileNotFoundError
        If data_file does not exist.
    IsADirectoryError
        If data_file is a directory rather than a file.
    ValueError
        If data_file has an unsupported file extension.

    """
    file_path = Path(data_file)
    supported_extensions = DataLoaderRegistry.list_registered()

    # Executes file existence, directory, and extension checks
    file_checks(file_path, supported_extensions)

    loader = DataLoaderRegistry.get(file_path.suffix)

    return cast(pd.DataFrame, loader(file_path, **kwargs))


def read_toml_configuration(config_file: str | Path) -> MedpipeConfig:
    """
    Reads a medpipe TOML configuration file.

    Parameters
    ----------
    config_file : str | Path
        Path to the configuration file.

    Returns
    -------
    config : MedpipeConfig
        Configuration for the pipeline.

    """
    file_checks(config_file, ".toml")

    with open(config_file, "rb") as file:
        raw_config = tomllib.load(file)

    config = MedpipeConfig.model_validate(raw_config)

    return config
