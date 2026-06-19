"""
I/O utilities module.

This module provides helper functions for reading from and writing to files,
handling various common I/O tasks.

Functions:
- load_data_from_csv: Loads the data from a .csv file.
- read_toml_configuration: Reads the top-level .TOML configuration file and
    returns contents with subconfiguration contents.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from .config import (
    SUBCONFIG_REGISTRY,
    MedpipeConfig,
    TopLevelConfig,
    parse_version_number,
    read_subconfiguration_file,
)
from .exceptions import file_checks

if TYPE_CHECKING:
    import pandas as pd

    from medpipe._types import Config


def load_data(data_file: str | Path) -> pd.DataFrame:
    """
    Reads a .csv or .parquet file and returns its contents.

    Parameters
    __________
    data_file : str
        Path to the .csv or .parquet file to load.

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
    file_checks(data_file, [".csv", ".parquet"])

    extension = Path(data_file).suffix

    if extension == ".csv":
        data = pd.read_csv(data_file)
    else:
        data = pd.read_parquet(data_file)
    return data


def read_toml_configuration(config_file: str | Path) -> MedpipeConfig:
    """
    Reads the top-level .TOML configuration file and returns contents with
    subconfiguration contents.

    Parameters
    ----------
    config_file : str | Path
        Path to the configuration file.

    Returns
    -------
    config : MedpipeConfig
        Configuration for the pipeline.

    Raises
    ------
    TypeError
        If config_file is not a str or Path.
    FileNotFoundError
        If config_file does not exist.
    IsADirectoryError
        If config_file is not a file.
    NotADirectoryError
        If subconfig_dir is not a directory.
    ValueError
        If config_file extension is not .toml file.
    tomllib.TOMLDecodeError
        If the file was not read properly.

    """
    file_checks(config_file, ".toml")

    with open(config_file, "rb") as file:
        raw_config = tomllib.load(file)

    # Check top-level configuration is correct
    top_level_config: TopLevelConfig = TopLevelConfig.model_validate(raw_config)

    subconfig_dir = Path(
        top_level_config.paths.config_dir
    )  # Create Path from config_dir
    subconfig_path = subconfig_dir.resolve()

    if not subconfig_dir.is_dir():
        raise NotADirectoryError(f"{subconfig_dir} is not a directory")

    v_list = parse_version_number(top_level_config.meta.version)

    parsed_configs: dict[str, Config] = {"top_level": top_level_config}

    for i, subtype in enumerate(SUBCONFIG_REGISTRY.keys()):
        sub_path = subconfig_path / subtype / (subtype + f"_v{v_list[i]}.toml")
        parsed_configs[subtype] = read_subconfiguration_file(sub_path, subtype)

    config = MedpipeConfig(**parsed_configs)  # type: ignore

    return config
