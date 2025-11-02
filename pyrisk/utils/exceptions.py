"""
Execption functions module.

This module provides functions for execption handling and raising.

Functions:
- path_checks: Checks if the path is correct.
"""

import pathlib


def path_checks(path: str, extension: str) -> None:
    """
    TODO: Add function description.

    Parameters:
    ----------
    arg
        arg description.

    Returns:
    -------
    return : return_type
        return description.

    Raises:
    ------
        FileNotFoundError
            If data_path does not exist.
        ValueError
            If data_path extension is not .extension file.

    """
    path_object = pathlib.Path(path)  # Create a Path object
    if not path_object.exists():
        raise FileNotFoundError(f"{path} does not exist")
    if path_object.suffix != extension:
        raise ValueError(f"{path} should be a {extension} file")
