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


def load_data_from_csv(data_file: str):
    """
    Reads a .csv file and returns its contents.

    Parameters
    __________
    data_file
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


def save_data_to_csv(df: pd.DataFrame, save_file: str) -> None:
    """
    Save data from a DataFrame to a .csv file.

    Parameters
    ----------
    df
        DataFrame to save.
    save_file
        Path and file name to save to.

    Returns
    -------
    None
        Nothing is returned.

    Raises
    ------
    TypeError
        If save_file is not a str.
    TypeError
        If df is not a pd.DataFrame or pd.Series.
    IsADirectoryError
        If save_file is not a file.
    ValueError
        If save_file extension is not .csv file.

    """
    try:
        exceptions.file_checks(save_file, ".csv")
    except (TypeError, ValueError, IsADirectoryError):
        raise
    except FileNotFoundError:
        # File will be created so skip FileNotFoundError
        pass

    if type(df) is not pd.DataFrame:
        raise TypeError("df should be a pd.DataFrame")

    df.to_csv(save_file)


def read_toml_configuration(config_file: str) -> dict:
    """
    Reads a .TOML configuration file and returns contents.

    Parameters
    ----------
    config_file
        Path to the configuration file.

    Returns
    -------
    config : dict
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
        config = tomllib.load(file)
    return config
