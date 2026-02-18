"""
Preprocessing functions module.

This module provides functions to preprocess data before training.

Functions:
- train_test_it: Creates a KFold iterator to split data into
    test and train sets.
- get_validation_idx: Removes some of the indices to create a validation set.
- convert_object_to_categorical: Converts object columns to categoricals.
- fit_preprocess_operations: Fits processing operations to data.
- bin_score: Bins the M3 score into 5 categories.
- extract_labels: Extracts prediction labels from data.
"""

from copy import deepcopy

import numpy as np
import pandas as pd
import sklearn as skl
from sklearn.preprocessing import OrdinalEncoder, PowerTransformer, StandardScaler

from medpipe.utils.exceptions import array_check, array_dim_check


def train_test_it(temporal_k_fold=False, **kwargs):
    """
    Creates a KFold iterator to split data into test and train sets.

    Parameters
    ----------
    temporal_k_fold : bool, default: False
        If True, the data will be split using a group and a
        GroupKFold iterator is returned.
    **kwargs
        Extra arguments for the StratifiedKFold or GroupKFold class.

    Returns
    -------
    kfold_it : StratifiedKFold or GroupKFold
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

    if not temporal_k_fold:
        kfold_it = skl.model_selection.StratifiedKFold(**args_dict)
    else:
        kfold_it = skl.model_selection.GroupKFold(**args_dict)

    return kfold_it


def get_validation_idx(idx_list, groups=None, val_size=0.1):
    """
    Removes some of the indices to create a validation set.

    If groups are provided, all the indices of the group with the largest
    value are selected as the validation set.

    Parameters
    ----------
    idx_list : np.array(n_samples,)
        Indices of the set to split.
    groups : pd.Series(n_samples,) or None, default: None
        Groups to which the train indices belong. Must be numeric.
    val_size : float, default: 0.1
        Size of the validation set if groups are None.

    Returns
    -------
    train_idx : np.array
        Train indices.
    val_idx : np.array
        Validation indices.

    """
    array_check(idx_list)
    if groups is not None:
        # If groups are provided
        groups = groups.to_numpy()  # Convert to array
        array_check(idx_list)
        array_dim_check(idx_list, groups, dim=0)

        if not np.isscalar(groups[0]):
            raise ValueError(f"groups should be scalar but instead got {groups.dtype}")
        group_max = np.max(groups)
        val_idx = np.where(groups == group_max)[0]
        train_idx = np.where(groups != group_max)[0]

    else:
        train_idx, val_idx = skl.model_selection.train_test_split(
            idx_list, test_size=val_size, random_state=42
        )

    return train_idx, val_idx


def convert_object_to_categorical(data: pd.DataFrame) -> pd.DataFrame:
    """
    Converts all object columns of a DataFrame to categoricals.

    Parameters
    ----------
    data
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


def fit_preprocess_operations(data, preprocessing_dict):
    """
    Fits processing operations to data.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame to manipulate.
    preprocessing_dict : dict[str, list[str]]
        Dictionary of the operations and the features on which to operate.

    Returns
    -------
    operation_dict : dict[]
        Dictionary of the different preprocessing objects.

    Raises
    ------
    TypeError
        If data is not a pd.DataFrame.
        If features is not a list(str).
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


def bin_score(data):
    """
    Bins the M3 score into 5 categories.

    Parameters
    ----------
    data : pd.DataFrame
        M3 score data.

    Returns
    -------
    binned_data : pd.DataFrame
        Binned data.

    """
    binned_data = np.ceil(data)
    binned_data[binned_data > 4] = 4
    return binned_data


def extract_labels(data, labels):
    """
    Extracts the prediction labels from the training data.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame to manipulate.
    labels : list(str)
        List of labels to extract from the data.

    Returns
    -------
    X : pd.DataFrame
        DataFrame containing the data.
    y : array-like
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
