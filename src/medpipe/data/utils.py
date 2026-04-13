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

from medpipe._types import Labels, PData
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
    if isinstance(groups, pd.Series):
        groups = groups.to_numpy()  # Convert to array
    elif not isinstance(groups, np.ndarray):
        raise TypeError(
            f"groups should be a pd.Series or np.array, but got {type(groups)}"
        )

    if groups.size != 0:
        # If groups are provided
        array_check(idx_list)
        array_dim_check(idx_list, groups, dim=0)

        if group_vals is not None:
            if not hasattr(group_vals, "__iter__"):
                raise TypeError("group_vals should be iterable")
            if (
                isinstance(group_vals, dict)
                or isinstance(group_vals, str)
                or isinstance(group_vals, tuple)
            ):
                raise TypeError("group_vals should be list or array")
            val_idx = np.array([], dtype=np.int64)
            train_idx = []
            for group in group_vals:
                val_idx = np.concatenate((val_idx, np.where(groups == group)[0]))

            train_idx = np.setdiff1d(np.arange(len(groups)), val_idx)

        else:
            group_max = np.max(groups)
            val_idx = np.where(groups == group_max)[0]
            train_idx = np.where(groups != group_max)[0]

    else:
        if not isinstance(val_size, float):
            raise TypeError(f"val_size should be a float, but got {type(val_size)}")
        if val_size < 0.0 or val_size > 1.0:
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
    if type(data) is not type(pd.DataFrame()):
        raise TypeError(f"data should be a pd.DataFrame, but got {type(data)}")

    if type(labels) is not type([]):
        raise TypeError(f"labels should be a list, but got {type(labels)}")

    if type(labels[0]) is not type(""):
        raise TypeError(f"labels should be a list(str), but got {type(labels[0])}")

    X = data.drop(labels, axis=1)
    y = data[labels]

    return X, y.to_numpy()


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
    for col in data.columns:
        col_type = data[col].dtype

        if pd.api.types.is_integer_dtype(col_type):
            data[col] = pd.to_numeric(data[col], downcast="integer")
        elif pd.api.types.is_float_dtype(col_type):
            data[col] = pd.to_numeric(data[col], downcast="float")
    return data


def convert_data(data: PData) -> PData:
    """
    Convert data to a ndarray if possible.

    The function checks if all columns are numeric to assess convertability.
    If the data can be converted, the pd.DataFrame is converted to ndarray.
    If the data is already ndarray the data is returned.

    Parameters
    ----------
    data : PData
        Data to check.

    Returns
    -------
    converted_data : npt.NDArray
        Converted data of shape (n_samples, n_features)

    """
    convertable = True
    if isinstance(data, np.ndarray):
        # Data does not need to be converted
        return data
    elif isinstance(data, pd.DataFrame):
        for col in data.columns:
            if not pd.api.types.is_numeric_dtype(data[col].dtype):
                # If one of the columns is not numeric return False
                convertable = False
        if convertable:
            # Convert data to a ndarray
            return data.to_numpy()
    return data


def get_data_from_idx(
    data: PData, idx: npt.NDArray | list[int] = np.array([])
) -> PData:
    """
    Returns the data at the given indices based on the data type.

    If no indices are provided, the full data is returned.

    Parameters
    ----------
    data : PData
        Data of shape (n_samples, n_features) to query.
    idx : npt.NDArray | list[int], default: np.array([])
        Array of shape (n_indices,) indices to extract data at.

    Returns
    -------
    indexed_data : PData
        Data of shape (n_indices, n_features).

    Raises
    ------
    TypeError
        If idx is not array-like.
        If idx does not contain integers.
        If data is not PData type.

    """
    array_check(idx)
    if len(idx) == 0:
        # If no indices were passed return all the data
        return data

    if type(idx[0]) is not int:
        # Check for integers
        raise TypeError(f"idx should contain int, but got {type(idx[0])}")

    if isinstance(data, np.ndarray):
        return data[idx]
    elif isinstance(data, pd.DataFrame):
        return data.values[idx]
    else:
        raise TypeError(f"data should be PData type, but got {type(data)}")
