"""
Execption functions module.

This module provides functions for execption handling and raising.

Functions:
- path_checks Checks if the path is correct.
"""

import pathlib


def file_checks(file: str, extension: str) -> None:
    """
    Performs checks to ensure that a file and extension are correct.

    Parameters
    ----------
    file
        File to check.
    extension
        Extension of the file to check.

    Returns
    -------
    None
        Nothing is returned.

    Raises
    ------
    TypeError
        If file is not a str.
    FileNotFoundError
        If file does not exist.
    IsADirectoryError
        If file is a directory.
    ValueError
        If file extension is not .extension file.

    """
    if type(file) is not str:
        raise TypeError(f"{file} should be a string")

    path_object = pathlib.Path(file)  # Create a Path object

    if not path_object.exists():
        raise FileNotFoundError(f"{file} does not exist")

    if not path_object.is_file():
        raise IsADirectoryError(f"{file} should be a file")

    if path_object.suffix != extension:
        raise ValueError(f"{file} should be a {extension} file")


def path_checks(path: str) -> None:
    """
    Performs checks to ensure that a path is correct.

    Parameters
    ----------
    path
        Path to check.

    Returns
    -------
    None
        Nothing is returned.

    Raises
    ------
    TypeError
        If path is not a str.
    FileNotFoundError
        If path does not exist.
    NotADirectoryError
        If path is not a directory.

    """
    if type(path) is not str:
        raise TypeError(f"{path} should be a string")

    path_object = pathlib.Path(path)  # Create a Path object

    if not path_object.exists():
        raise FileNotFoundError(f"{path} does not exist")

    if not path_object.is_dir():
        raise NotADirectoryError(f"{path} should be a directory")
