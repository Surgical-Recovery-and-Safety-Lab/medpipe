"""
Execption functions module.

This module provides functions for execption handling and raising.

Functions:
- file_checks: Checks if the file is correct.
- path_checks: Checks if the path is correct.
- array_check: Checks for an array-like.
- array_dim_check: Checks that the dimension of two arrays agree.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from numpy import ndarray
from pandas import Series

if TYPE_CHECKING:
    import numpy.typing as npt
    import pandas as pd


def file_checks(
    file: str | Path, extension: str | list[str], exists: bool = True
) -> None:
    """
    Performs checks to ensure that a file and extension are correct.

    Parameters
    ----------
    file : str | Path
        File to check.
    extension : str | list[str]
        Extension or list of extensions of the file to check.
    exists : bool, default: True
        Flag to indicate if the file should exists.

    Returns
    -------
    None
        Nothing is returned.

    Raises
    ------
    TypeError
        If file is not a str or a Path.
        If extension is not a str or list[str].
    FileNotFoundError
        If file does not exist.
    IsADirectoryError
        If file is a directory.
    ValueError
        If file extension is not correct.

    """
    if not isinstance(file, (str, Path)):
        raise TypeError(f"File should be a string or Path")

    if not isinstance(extension, (str, list)):
        raise TypeError(f"Extension should be a string or list of strings")

    path_object = Path(file)  # Create a Path object

    path_checks(str(path_object.parent))

    if not path_object.exists() and exists:
        raise FileNotFoundError(f"{file} does not exist")

    if not path_object.is_file() and exists:
        raise IsADirectoryError(f"{file} should be a file")

    suffix = path_object.suffix
    if isinstance(extension, str):
        if suffix != extension:
            raise ValueError(f"File suffix should be {extension}, but got {suffix}")
    else:
        if suffix not in extension:
            raise ValueError(
                f"File suffix should be one of {extension}, but got {suffix}"
            )


def path_checks(path: str | Path) -> None:
    """
    Performs checks to ensure that a path is correct and creates it
    if it does not exist.

    Parameters
    ----------
    path : str | Path
        Path to check.

    Returns
    -------
    None
        Nothing is returned.

    Raises
    ------
    TypeError
        If path is not a str or Path.
    FileNotFoundError
        If path does not exist.
    NotADirectoryError
        If path is not a directory.

    """
    if not isinstance(path, (str, Path)):
        raise TypeError(f"Path should be a string or a Path")

    path_object = Path(path)  # Create a Path object

    if not path_object.exists() and dir:
        path_object.mkdir(parents=True)

    if not path_object.is_dir():
        raise NotADirectoryError(f"{path} should be a directory")


def array_check(arr: npt.NDArray | pd.Series | list[Any]) -> None:
    """
    Checks that the input is an array-like.

    Parameters
    ----------
    arr : npt.NDArray | pd.Series | list
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
    target_types = (list, ndarray, Series)
    if not isinstance(arr, target_types):
        raise TypeError(f"Input should be an array-like but instead got {type(arr)}")


def array_dim_check(
    arr1: npt.NDArray | pd.Series, arr2: npt.NDArray | pd.Series, dim: int | None = None
) -> None:
    """
    Checks that the dimensions of the arrays match.

    Parameters
    ----------
    arr1 : npt.NDArray | pd.Series
        First array.
    arr2 : npt.NDArray | pd.Series
        Second array.
    dim : int | None, default: None
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
            raise TypeError("Input dim should be an integer")
        if arr1.shape[dim] != arr2.shape[dim]:
            raise ValueError(f"The {dim} axis does not agree")
