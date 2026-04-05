"""
Preprocessing functions module.

This module provides functions to preprocess data before training.

Functions:
- train_test_it: Creates a KFold iterator to split data into test and train sets.
- get_validation_idx: Removes some of the indices to create a validation set.
- convert_object_to_categorical: Converts object columns to categoricals.
- fit_preprocess_operations: Fits processing operations to data.
- bin_score: Bins the M3 score into 5 categories.
- extract_labels: Extracts prediction labels from data.
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np
import pandas as pd
import sklearn as skl
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, PowerTransformer, StandardScaler

from medpipe._types import Labels, PreprocessOp, PreprocessOpConfig
from medpipe.utils.exceptions import array_check, array_dim_check

if TYPE_CHECKING:
    import numpy.typing as npt


def train_test_it(
    group_k_fold: bool = False, **kwargs: Any
) -> StratifiedKFold | GroupKFold:
    """
    Creates a KFold iterator to split data into test and train sets.

    Parameters
    ----------
    group_k_fold : bool, default: False
        If True, the data will be split using a group and a
        GroupKFold iterator is returned.
    **kwargs: Any
        Extra arguments for the StratifiedKFold or GroupKFold class.

    Returns
    -------
    kfold_it : StratifiedKFold | GroupKFold
        KFold iterator.

    Raises
    ------
    ValueError
        If n_splits is less than 2.

    """
    # Create the correct argument dict for StratifiedKFold
    args_dict = dict()
    for key, value in kwargs.items():
        match key:
            case "random_state":
                if value == -1:
                    value = None
                args_dict.update({key: value})

            case "shuffle":
                args_dict.update({key: value})

            case "n_splits":
                if value < 2:
                    raise ValueError(
                        f"n_splits should be greater than 2, but got {value}"
                    )

                args_dict.update({key: value})

    if not group_k_fold:
        kfold_it = StratifiedKFold(**args_dict)
    else:
        kfold_it = GroupKFold(**args_dict)

    return kfold_it


def get_validation_idx(
    idx_list: npt.NDArray,
    groups: pd.Series | npt.NDArray | None = None,
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
    groups : pd.Series | npt.NDArray | None, default: None
        Groups of shape (n_samples,) to which the train indices belong.
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
    if groups is not None:
        # If groups are provided
        if isinstance(groups, pd.Series):
            groups = groups.to_numpy()  # Convert to array
        elif not isinstance(groups, np.ndarray):
            raise TypeError(
                f"groups should be a pd.Series or np.array, but got {type(groups)}"
            )
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


def convert_object_to_categorical(data: pd.DataFrame) -> pd.DataFrame:
    """
    Converts all object columns of a DataFrame to categoricals.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame to manipulate.

    Returns
    -------
    processed_data : pd.DataFrame
        Processed DataFrame.

    Raises
    ------
    TypeError
        If data is not a pd.DataFrame.

    """
    if type(data) is not type(pd.DataFrame()):
        raise TypeError(f"data should be a pd.DataFrame, but got {type(data)}")

    # Create a copy of data to work on
    processed_data = data

    for column in data.select_dtypes(include=["object"]).columns:
        processed_data[column] = data[column].astype("category")

    return processed_data


def fit_preprocess_operations(
    data: pd.DataFrame, preprocessing_dict: PreprocessOpConfig
) -> Mapping[str, PreprocessOp | str]:
    """
    Fits processing operations to data.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame to manipulate.
    preprocessing_dict : PreprocessOpConfig
        Dictionary of the operations and the features on which to operate.

    Returns
    -------
    operation_dict : Mapping[str, PreprocessOp | str]
        Dictionary of the different preprocessing objects.

    Raises
    ------
    TypeError
        If data is not a pd.DataFrame.
        If features is not a list[str].
    KeyError
        If a features is not a valid key.
    ValueError
        If preprocess is not a valid preprocessing function.

    """
    if type(data) is not type(pd.DataFrame()):
        raise TypeError(f"data should be a pd.DataFrame, but got {type(data)}")

    # Operation dictionary to store fitted operations
    data_copy = deepcopy(data)
    operation_dict = dict()

    for preprocess in preprocessing_dict.keys():
        features = preprocessing_dict[preprocess]["feature_list"]

        if type(features) is not type([]):
            raise TypeError(f"features should be a list, but got {type(features)}")

        if type(features[0]) is not type(""):
            raise TypeError(
                f"features should be a list(str), but got list({type(features[0])}"
            )

        match preprocess:
            case "ordinal_encoder":
                operation_dict[preprocess] = OrdinalEncoder().fit(data_copy[features])
                data_copy[features] = operation_dict[preprocess].transform(
                    data_copy[features]
                )
            case "standardise":
                operation_dict[preprocess] = StandardScaler().fit(data_copy[features])
                data_copy[features] = operation_dict[preprocess].transform(
                    data_copy[features]
                )
            case "power_transform":
                operation_dict[preprocess] = PowerTransformer().fit(data_copy[features])
                data_copy[features] = operation_dict[preprocess].transform(
                    data_copy[features]
                )
            case "bin":
                operation_dict[preprocess] = "bin"
            case _:
                raise ValueError(f"{preprocess} invalid preprocessing function")

    return operation_dict


def bin_score(data: npt.NDArray) -> npt.NDArray:
    """
    Bins the M3 score into 5 categories.

    Parameters
    ----------
    data : npt.NDArray
        M3 score data.

    Returns
    -------
    binned_data : npt.NDArray
        Binned data.

    """
    binned_data = np.ceil(data)
    binned_data[binned_data > 4] = 4
    return binned_data


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
