"""
I/O utilities module.

This module provides helper functions for reading from and writing to files,
handling various common I/O tasks.

Functions:
- load_data_from_csv: Loads the data from a .csv file.
- save_data_to_csv: Saves a DataFrame or Series to a .csv file.
- read_toml_configuration: Parses the contents of a .TOML file.
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
        Path to the .csv file to load.

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
    try:
        exceptions.path_checks(data_path, ".csv")
    except (FileNotFoundError, TypeError, ValueError):
        raise

    data = pd.read_csv(data_path)
    return data


def save_data_to_csv(df: pd.DataFrame, file_path: str) -> None:
    """
    Save data from a DataFrame to a .csv file.

    Parameters:
    ----------
    df
        DataFrame to save.
    file_path
        Path and file name to save to.

    Returns:
    -------
    None
        Nothing is returned.

    Raises:
    ------
    TypeError
        If file_path is not a str.
    TypeError
        If df is not a pd.DataFrame or pd.Series.
    ValueError
        If file_path extension is not .csv file.

    """
    try:
        exceptions.path_checks(file_path, ".csv")
    except (TypeError, ValueError):
        raise
    except FileNotFoundError:
        # File will be created so skip FileNotFoundError
        pass

    if type(df) is not pd.DataFrame:
        raise TypeError("df should be a pd.DataFrame")

    df.to_csv(file_path)


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
    try:
        exceptions.path_checks(config_path, ".toml")
    except (FileNotFoundError, TypeError, ValueError):
        raise

    with open(config_path, "rb") as file:
        config = tomllib.load(file)
    return config
