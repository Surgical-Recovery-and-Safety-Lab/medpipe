"""
Utility functions module.

This module provides utility functions for data manipulation.

Functions:
- get_split_idx: Returns the indices for the data splits.
- split_data: Split data into train and test or train and recalibration sets.
- extract_labels: Extracts prediction labels from data.
- convert_dtypes: Converts data types to category in a pd.DataFrame.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from medpipe._types import Labels
from medpipe.utils.exceptions import array_check, array_dim_check

if TYPE_CHECKING:
    import numpy.typing as npt


def get_split_idx(
    idx_list: npt.NDArray,
    column: pd.Series | npt.NDArray,
    values: list[str] | list[int],
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Returns the indices for the data splits.

    Parameters
    ----------
    idx_list : npt.NDArray
        Indices of the set to split of shape (n_samples,).
    column : pd.Series | npt.NDArray
        Column of shape (n_samples,) used to split the data.
    values : list[str] | list[int]
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
        raise TypeError(f"values should be list or np.array, got {type(values)}")

    # Vectorized selection: Find where 'column' matches any value in 'values'
    val_mask = np.isin(column, values)
    other_idx = np.where(val_mask)[0]
    train_idx = np.where(~val_mask)[0]

    if len(other_idx) == 0:
        # Other indices are empty because value was not in column
        raise ValueError(f"{values} not present in column")

    return train_idx, other_idx


def split_data(
    features: pd.DataFrame,
    labels: Labels,
    strategy: Literal["random", "group"],
    group_column: str | None = None,
    values: list[str] | list[int] | None = None,
    test_size: float | None = None,
    recalibration_size: float | None = None,
) -> tuple[pd.DataFrame, Labels, pd.DataFrame, Labels]:
    """
    Split data into train and test or train and recalibration sets.

    If strategy is group then group_column and values must be specified.
    If strategy is random then test_size or recalibration_size must be
    specified. If both are specified, the selected value is test_size.

    Parameters
    ----------
    features : pd.DataFrame
        Features to split.
    labels : Labels
        Labels to split.
    strategy : {"random", "group"}
        Strategy used to split the data.
    group_column : str | None, default: None
        Name of the column used to split with if strategy is group.
    values : list[str] | list[int] | None, default: None
        Values of the group column that do not belong to the train set.
    test_size : float | None, default: None
        Test set size if the strategy is random.
    recalibration_size : float | None, default: None
        Recalibration set size if the strategy is random.

    Returns
    -------
    X_train, X_test : pd.DataFrame
        Train and test / recalibration set.
    y_train, y_test : Labels
        Train and test / recalibration labels.

    Raises
    ------
    TypeError
        If features is not a pd.DataFrame.
        If labels is not a np.ndarray.
    ValueError
        If group_colum and values not specified with group strategy.
        If test_size or recalibration_size not specified with random strategy.

    """
    if not isinstance(features, pd.DataFrame):
        raise TypeError(f"features should be a pd.DataFrame, but got {type(features)}")
    if not isinstance(labels, np.ndarray):
        raise TypeError(f"labels should be a np.array, but got {type(labels)}")

    if strategy == "group":
        if not group_column or not values:
            raise ValueError(
                "group_column and values must be specified with group strategy"
            )

        train_idx, test_idx = get_split_idx(
            np.arange(len(features)),
            features[group_column],  # type: ignore
            values,  # type: ignore
        )

        X_train = features.iloc[train_idx]
        y_train = labels[train_idx]
        X_test = features.iloc[test_idx]
        y_test = labels[test_idx]

    elif strategy == "random":
        if test_size:
            size = test_size
        elif recalibration_size:
            size = recalibration_size
        else:
            raise ValueError(
                "test_size or recalibration_size must be specified with random strategy"
            )

        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=size
        )

    else:
        raise ValueError(f"strategy should be random or group, but got {strategy}")

    return (
        cast(pd.DataFrame, X_train),
        cast(Labels, y_train),
        cast(pd.DataFrame, X_test),
        cast(Labels, y_test),
    )


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
        If labels is not list[str].
    ValueError
        If a prediction label is not a valid key.

    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"data should be a pd.DataFrame, but got {type(data)}")
    if not isinstance(labels, list):
        raise TypeError(f"labels should be a list, but got {type(labels)}")
    if labels and not isinstance(labels[0], str):
        raise TypeError(f"labels should contain strings, but got {type(labels[0])}")

    # .drop() and column selection are already highly optimized in Pandas
    for label in labels:
        if label not in data.columns:
            raise ValueError(f"{label} was not found in data")

    X = data.drop(columns=labels)
    y = data[labels].to_numpy()

    return X, y


def convert_dtypes(X: pd.DataFrame) -> pd.DataFrame:
    """
    Converts data object types to category in a pd.DataFrame
    to avoid errors when cross-validate is called.

    Data is checked to see if it can be converted to numeric
    before converting to categorical. Timedeltas are converted
    to days.

    Parameters
    ----------
    X : pd.DataFrame
        Data to convert.

    Returns
    -------
    X_converted  : pd.DataFrame
        Data with converted dtypes.

    Raises
    ------
    TypeError
        If X is not a pd.DataFrame.

    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"Input X should be a pd.DataFrame, but got {type(X)}")
    obj_cols = X.select_dtypes(include=["object"]).columns
    for col in obj_cols:
        try:
            X[col] = pd.to_numeric(X[col])
        except ValueError:
            X[col] = X[col].astype("category")

    return X
