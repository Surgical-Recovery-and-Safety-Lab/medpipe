"""
I/O utilities module.

This module provides helper functions for reading from and writing to files,
handling various common I/O tasks.

Functions:
- load_data_from_csv: Loads the data from a csv file.
- read_toml_configuration: Parses the contents of a TOML file.
"""

import pathlib
import tomllib

import pandas as pd


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

    path = pathlib.Path(data_path)  # Create a Path object
    if not path.exists():
        raise FileNotFoundError(f"{data_path} does not exist")
    if pathlib.Path(data_path).suffix != ".csv":
        raise ValueError(f"{data_path} should be a csv file")

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

    path = pathlib.Path(config_path)  # Create a Path object
    if not path.exists():
        raise FileNotFoundError(f"{config_path} does not exist")
    if pathlib.Path(config_path).suffix != ".toml":
        raise ValueError(f"{config_path} should be a toml file")

    with open(config_path, "rb") as file:
        config = tomllib.load(file)
    return config
