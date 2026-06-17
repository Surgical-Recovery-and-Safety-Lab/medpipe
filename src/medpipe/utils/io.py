"""
I/O utilities module.

This module provides helper functions for reading from and writing to files,
handling various common I/O tasks.

Functions:
- load_data_from_csv: Loads the data from a .csv file.
- read_toml_configuration: Parses the contents of a .TOML file.
"""

from __future__ import annotations

import tomllib
from typing import TYPE_CHECKING, Any

import pandas as pd

from medpipe._types import (
    DataConfig,
    HyperparameterConfig,
    MedpipeConfig,
    TopLevelConfig,
    WorkflowConfig,
)

from . import exceptions

if TYPE_CHECKING:
    import pandas as pd


def load_data_from_csv(data_file: str) -> pd.DataFrame:
    """
    Reads a .csv file and returns its contents.

    Parameters
    __________
    data_file : str
        Path to the .csv file to load.

    Returns
    _______
    data : pd.DataFrame
        Loaded data.

    Raises
    ______
    TypeError
        If data_file is not a str.
    FileNotFoundError
        If data_file does not exist.
    IsADirectoryError
        If data_file is not a file.
    ValueError
        If data_file extension is not .csv file.

    """
    try:
        exceptions.file_checks(data_file, ".csv")
    except (FileNotFoundError, IsADirectoryError, TypeError, ValueError):
        raise

    data = pd.read_csv(data_file)
    return data


def read_toml_configuration(config_file: str) -> dict[str, Any]:
    """
    Reads the top-level .TOML configuration file and returns contents with
    subconfiguration contents.

    Parameters
    ----------
    config_file : str
        Path to the configuration file.

    Returns
    -------
    config : dict[str, Any]
        Configuration contents as a dictionary.

    Raises
    ------
    TypeError
        If config_file is not a str.
    FileNotFoundError
        If config_file does not exist.
    IsADirectoryError
        If config_file is not a file.
    ValueError
        If data_file extension is not .csv file.
    tomllib.TOMLDecodeError
        If the file was not read properly.

    """
    try:
        exceptions.file_checks(config_file, ".toml")
    except (FileNotFoundError, IsADirectoryError, TypeError, ValueError):
        raise

    with open(config_file, "rb") as file:
        raw_config = tomllib.load(file)

    # Check top-level configuration is correct
    config: TopLevelConfig = TopLevelConfig.model_validate(raw_config)
    breakpoint()

    return config
