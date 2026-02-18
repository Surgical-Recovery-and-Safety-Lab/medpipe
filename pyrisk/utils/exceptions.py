"""
Execption functions module.

This module provides functions for execption handling and raising.

Functions:
- file_checks: Checks if the file is correct.
- path_checks: Checks if the path is correct.
- array_check: Checks for an array-like.
- array_dim_check: Checks that the dimension of two arrays agree.
"""

import pathlib

from pandas.api.types import is_list_like


def file_checks(file: str, extension: str, exists: bool = True) -> None:
    """
    Performs checks to ensure that a file and extension are correct.

    Parameters
    ----------
    file
        File to check.
    extension
        Extension of the file to check.
    exists : default: True
        Flag to indicate if the file should exists.

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

    path_checks(str(path_object.parent))

    if not path_object.exists() and exists:
        raise FileNotFoundError(f"{file} does not exist")

    if not path_object.is_file() and exists:
        raise IsADirectoryError(f"{file} should be a file")

    if path_object.suffix != extension:
        raise ValueError(f"{file} should be a {extension} file")


def path_checks(path: str) -> None:
    """
    Performs checks to ensure that a path is correct and creates it
    if it does not exist.

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
    path_suffix = path_object.suffix
    dir = False

    if path_suffix != "":
        try:
            # If suffix ends with a digit because of version number
            int(path_suffix[1:])
            dir = True
        except ValueError:
            pass
    else:
        # If there is no suffix
        dir = True

    if not path_object.exists() and dir:
        path_object.mkdir(parents=True)

    if not path_object.exists():
        raise FileNotFoundError(f"{path} does not exist")

    if not path_object.is_dir():
        raise NotADirectoryError(f"{path} should be a directory")


def array_check(arr) -> None:
    """
    Checks that the input is an array-like.

    Parameters
    ----------
    arr : array-like
        Array to check.

    Returns
    -------
    None
        Nothing is returned.

    Raises
    ------
    TypeError
        If arr is not an array-like.

    """
    if not is_list_like(arr):
        raise TypeError(f"arr should be an array-like but instead got {type(arr)}")


def array_dim_check(arr1, arr2, dim=None) -> None:
    """
    Checks that the dimensions of the arrays match.

    Parameters
    ----------
    arr1 : array-like
        First array.
    arr2 : array-like
        Second array.
    dim : int or None, default: None
        Dimension to compare. If None shape is used.

    Returns
    -------
    None
        Nothing is returned.

    Raises
    ------
    TypeError
        If dim is not an integer.
    ValueError
        If the arrays do not have the same dimensions.

    """
    # Check arrays
    array_check(arr1)
    array_check(arr2)

    if dim is None:
        if arr1.shape != arr2.shape:
            raise ValueError("The dimensions do not agree")
    else:
        if type(dim) is not int:
            raise TypeError("dim should be an integer")
        if arr1.shape[dim] != arr2.shape[dim]:
            raise ValueError(f"The {dim} axis does not agree")
