"""
Execption functions module.

This module provides functions for execption handling and raising.

Functions:
- path_checks Checks if the path is correct.
"""

import pathlib


def path_checks(path: str, extension: str) -> None:
    """
    Performs checks to ensure that a path and extension are correct.

    Parameters
    ----------
    path
        Path to the file to check.
    extension
        Extension of the file to check.

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
    ValueError
        If path extension is not .extension file.

    """
    if type(path) is not str:
        raise TypeError(f"{path} should be a string")

    path_object = pathlib.Path(path)  # Create a Path object

    if not path_object.exists():
        raise FileNotFoundError(f"{path} does not exist")

    if path_object.suffix != extension:
        raise ValueError(f"{path} should be a {extension} file")
