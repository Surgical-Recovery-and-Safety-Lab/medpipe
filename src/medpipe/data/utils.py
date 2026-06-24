"""
Utility functions module.

This module provides utility functions for data manipulation.

Functions:
- get_split_idx: Returns the indices for the data splits.
- extract_labels: Extracts prediction labels from data.
- downcast_dtypes: Downcasts the float and int dtypes in data.
- convert_data: Convert data to a ndarray if possible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from medpipe._types import Labels, PredData
from medpipe.utils.exceptions import array_check, array_dim_check

if TYPE_CHECKING:
    import numpy.typing as npt


def get_split_idx(
    idx_list: npt.NDArray,
    column: pd.Series | npt.NDArray,
    values: list[str | int],
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Returns the indices for the data splits.

    Parameters
    ----------
    idx_list : npt.NDArray
        Indices of the set to split of shape (n_samples,).
    column : pd.Series | npt.NDArray
        Column of shape (n_samples,) used to split the data.
    values : list[str | int]
        Group values that should be in the test or recalibration set.

    Returns
    -------
    train_idx : npt.NDArray
        Train indices.
    other_idx : npt.NDArray
        Other indices for the test or recalibration set.

    Raises
    ------
    TypeError
        If column is not pd.Series or np.ndarray.
        If values is not a list or a np.ndarray.

    """
    array_check(idx_list)

    # Standardize groups to numpy
    if isinstance(column, pd.Series):
        column = column.to_numpy()
    elif not isinstance(column, np.ndarray):
        raise TypeError(f"column should be pd.Series or np.array, got {type(column)}")

    array_dim_check(idx_list, column, dim=0)  # Ensure dimension match

    # Type checking validation
    if not isinstance(values, (list, np.ndarray)):
        raise TypeError("values should be list or array")

    # Vectorized selection: Find where 'column' matches any value in 'values'
    val_mask = np.isin(column, values)
    other_idx = np.where(val_mask)[0]
    train_idx = np.where(~val_mask)[0]

    return train_idx, other_idx


        )

    return train_idx, val_idx


def extract_labels(
    data: pd.DataFrame, labels: list[str]
) -> tuple[pd.DataFrame, Labels]:
    """
    Extracts the prediction labels from the training data.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame to manipulate.
    labels : list[str]
        List of labels to extract from the data.

    Returns
    -------
    X : pd.DataFrame
        DataFrame containing the data.
    y : Labels
        Array containing the prediction labels.

    Raises
    ------
    TypeError
        If data is not a pd.DataFrame.
    TypeError
        If labels is not list(str).
    KeyError
        If a prediction label is not a valid key.

    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"data should be a pd.DataFrame, but got {type(data)}")
    if not isinstance(labels, list):
        raise TypeError(f"labels should be a list, but got {type(labels)}")
    if labels and not isinstance(labels[0], str):
        raise TypeError(f"labels should contain strings, but got {type(labels[0])}")

    # .drop() and column selection are already highly optimized in Pandas
    X = data.drop(columns=labels)
    y = data[labels].to_numpy()

    return X, y


def downcast_dtypes(data: pd.DataFrame) -> pd.DataFrame:
    """
    Downcasts the float64 and int64 dtypes in data.

    Parameters
    ----------
    data : pd.DataFrame
        Data to downcast of shape (n_samples, n_labels).

    Returns
    -------
    downcast_data : pd.DataFrame
        Downcast data of shape (n_samples, n_labels).

    """
    df = data.copy()

    # Process integers
    ints = df.select_dtypes(include=["integer"]).columns
    for col in ints:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    # Process floats
    floats = df.select_dtypes(include=["floating"]).columns
    for col in floats:
        df[col] = pd.to_numeric(df[col], downcast="float")

    return df


def convert_data(data: PredData) -> PredData:
    """
    Convert data to a ndarray if possible.

    The function checks if all columns are numeric to assess convertability.
    If the data can be converted, the pd.DataFrame is converted to ndarray.
    If the data is already ndarray the data is returned.

    Parameters
    ----------
    data : PredData
        Data to check.

    Returns
    -------
    converted_data : npt.NDArray
        Converted data of shape (n_samples, n_features)

    """
    if isinstance(data, np.ndarray):
        return data

    if isinstance(data, pd.DataFrame):
        # select_dtypes is faster than iterating through columns manually
        numeric_cols = data.select_dtypes(include=[np.number])

        # If the number of numeric columns equals total columns, convert all
        if numeric_cols.shape[1] == data.shape[1]:
            return data.to_numpy()

    return data


def get_data_from_idx(
    data: PredData, idx: npt.NDArray | list[int] = np.array([])
) -> PredData:
    """
    Returns the data at the given indices based on the data type.

    If no indices are provided, the full data is returned.

    Parameters
    ----------
    data : PredData
        Data of shape (n_samples, n_features) to query.
    idx : npt.NDArray | list[int], default: np.array([])
        Array of shape (n_indices,) indices to extract data at.

    Returns
    -------
    indexed_data : PredData
        Data of shape (n_indices, n_features).

    Raises
    ------
    TypeError
        If idx is not array-like.
        If idx does not contain integers.
        If data is not PredData type.

    """
    # Early exit for empty/None
    if idx is None or (isinstance(idx, (np.ndarray, list)) and len(idx) == 0):
        return data

    # Convert once without copying if already an array
    idx_arr = np.asanyarray(idx)

    # Boolean masks are allowed (must match data length)
    # Integer indices are allowed (positional)
    is_int = np.issubdtype(idx_arr.dtype, np.integer)
    is_bool = np.issubdtype(idx_arr.dtype, np.bool_)

    if not (is_int or is_bool):
        raise TypeError(f"idx must be integers or booleans, but got {idx_arr.dtype}")

    # Handle DataFrames
    if isinstance(data, pd.DataFrame):
        return data.iloc[idx_arr]

    # Handle NumPy
    if isinstance(data, np.ndarray):
        return data[idx_arr]

    raise TypeError(f"Expected PredData (NDArray/DataFrame), but got {type(data)}")
