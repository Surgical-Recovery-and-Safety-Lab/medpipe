"""
I/O utilities module.

This module provides helper functions for reading from and writing to files,
handling various common I/O tasks.

Functions:
- load_data_from_csv: Loads the data from a csv file.
- read_toml_configuration: Parses the contents of a TOML file.
"""

import tomllib

import pandas as pd

from . import exceptions


def load_data_from_csv(data_path: str):
    """
    Reads a csv file and returns its contents.

    Parameters
    __________
    data_path
        Path to the csv file to load.

    Returns
    _______
    data : pd.DataFrame
        Loaded data.

    Raises
    ______
    TypeError
        If data_path is not a str.
    FileNotFoundError
        If data_path does not exist.
    ValueError
        If data_path extension is not .csv file.

    """
    if type(data_path) is not str:
        raise TypeError(f"{data_path} should be a string")

    try:
        exceptions.path_checks(data_path, ".csv")
    except (FileNotFoundError, TypeError):
        raise

    data = pd.read_csv(data_path)
    return data


def read_toml_configuration(config_path: str) -> dict:
    """
    Reads a .TOML configuration file and returns contents.

    Parameters:
    ----------
    config_path
        Path to the configuration file.

    Returns:
    -------
    config : dict
        Configuration contents as a dictionary.

    Raises:
    ------
    TypeError
        If data_path is not a str.
    FileNotFoundError
        If data_path does not exist.
    ValueError
        If data_path extension is not .csv file.
    tomllib.TOMLDecodeError
        If the file was not read properly.

    """
    if type(config_path) is not str:
        raise TypeError(f"{config_path} should be a string")

    try:
        exceptions.path_checks(config_path, ".toml")
    except (FileNotFoundError, TypeError):
        raise

    with open(config_path, "rb") as file:
        config = tomllib.load(file)
    return config
