"""
I/O utilities module.

This module provides helper functions for reading from and writing to files,
handling various common I/O tasks.

"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict

import pandas as pd

from .config import MedpipeConfig
from .exceptions import file_checks

if TYPE_CHECKING:
    import pandas as pd

# Central registry mapping file extensions to Pandas loading callables
DATA_LOADER_REGISTRY: Dict[str, Callable[..., pd.DataFrame]] = {
    ".csv": pd.read_csv,
    ".tsv": lambda filepath, **kwargs: pd.read_csv(filepath, sep="\t", **kwargs),
    ".txt": pd.read_csv,
    ".parquet": pd.read_parquet,
    ".pq": pd.read_parquet,
    ".feather": pd.read_feather,
    ".xlsx": pd.read_excel,
    ".xls": pd.read_excel,
    ".json": pd.read_json,
    ".jsonl": lambda filepath, **kwargs: pd.read_json(filepath, lines=True, **kwargs),
    ".pkl": pd.read_pickle,
    ".pickle": pd.read_pickle,
}


def register_data_loader(
    extension: str, loader_func: Callable[..., pd.DataFrame]
) -> None:
    """Register a custom file reader for a specific extension.

    Parameters
    ----------
    extension : str
        File extension starting with a dot (e.g., '.feather').
    loader_func : Callable[..., pd.DataFrame]
        Function accepting a file path and returning a Pandas DataFrame.

    """
    ext = extension.lower().strip()
    if not ext.startswith("."):
        ext = f".{ext}"
    DATA_LOADER_REGISTRY[ext] = loader_func


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
    supported_extensions = list(DATA_LOADER_REGISTRY.keys())

    # Executes file existence, directory, and extension checks
    file_checks(file_path, supported_extensions)

    extension = file_path.suffix.lower()
    loader = DATA_LOADER_REGISTRY[extension]

    return loader(file_path, **kwargs)


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
