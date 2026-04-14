"""
Utility functions module.

This module provides utility functions for data manipulation.

Functions:
- get_validation_idx: Removes some of the indices to create a validation set.
- extract_labels: Extracts prediction labels from data.
- downcast_dtypes: Downcasts the float and int dtypes in data.
- convert_data: Convert data to a ndarray if possible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import sklearn as skl

from medpipe._types import Labels, PredData
from medpipe.utils.exceptions import array_check, array_dim_check

if TYPE_CHECKING:
    import numpy.typing as npt


def get_validation_idx(
    idx_list: npt.NDArray,
    groups: pd.Series | npt.NDArray = np.array([]),
    group_vals: list[Any] | None = None,
    val_size: float = 0.1,
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Removes some of the indices to create a validation set.

    If groups are provided and group_vals is None, all the indices of
    the group with the largest value are selected as the validation set.
    If group_vals is specified with groups, then the group_vals are
    selected as the validation/test set.

    Parameters
    ----------
    idx_list : npt.NDArray
        Indices of the set to split of shape (n_samples,).
    groups : pd.Series | npt.NDArray, default: np.array([])
        Groups of shape (n_samples,) to which the train indices belong or empty.
    group_vals : list[Any] | None, default: None
        Group values that should be in the test set.
    val_size : float, default: 0.1
        Size of the validation set if groups are None.

    Returns
    -------
    train_idx : npt.NDArray
        Train indices.
    val_idx : npt.NDArray
        Validation indices.

    Raises
    ------
    TypeError
        If groups is not pd.Series or np.ndarray.
        If group_vals is not iterable.
        If group_vals is not a list or a np.ndarray.
        If val_size is not a float.
    ValueError
        If val_size < 0 or val_size > 1.

    """
    array_check(idx_list)

    # Standardize groups to numpy
    if isinstance(groups, pd.Series):
        groups = groups.to_numpy()
    elif not isinstance(groups, np.ndarray):
        raise TypeError(f"groups should be pd.Series or np.array, got {type(groups)}")

    if groups.size != 0:
        array_dim_check(idx_list, groups, dim=0)

        if group_vals is not None:
            # Type checking validation
            if not isinstance(group_vals, (list, np.ndarray)):
                raise TypeError("group_vals should be list or array")

            # Vectorized selection: Find where 'groups' matches any value in 'group_vals'
            val_mask = np.isin(groups, group_vals)
            val_idx = np.where(val_mask)[0]
            train_idx = np.where(~val_mask)[0]
        else:
            # Default: Take the largest group ID as validation
            group_max = np.max(groups)
            val_mask = groups == group_max
            val_idx = np.where(val_mask)[0]
            train_idx = np.where(~val_mask)[0]

    else:
        if not isinstance(val_size, float):
            raise TypeError(f"val_size should be a float, but got {type(val_size)}")
        if not (0.0 <= val_size <= 1.0):
            raise ValueError(f"val_size should be between 0 and 1, but got {val_size}")

        train_idx, val_idx = skl.model_selection.train_test_split(
            idx_list, test_size=val_size, random_state=42
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
