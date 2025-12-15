"""
Preprocessing functions module.

This module provides functions to preprocess data before training.

Functions:
- test_train_it: Creates a KFold iterator to split data into
    test and train sets.
- get_validation_idx: Removes some of the indices to create a validation set.
- convert_object_to_categorical: Converts object columns to categoricals.
- preprocess_data: Processed select data columns based on preprocessing function.
- extract_labels: Extracts prediction labels from data.
"""

from copy import deepcopy

import numpy as np
import pandas as pd
import sklearn as skl
from sklearn.preprocessing import LabelEncoder, PowerTransformer, StandardScaler

from pyrisk.utils.exceptions import array_check, array_dim_check


def test_train_it(temporal_k_fold=False, **kwargs):
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

    Raises
    ------
    Error
        Add exceptions that might be raised.

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


def preprocess_data(data, features, preprocess):
    """
    Processed select data columns based on preprocessing function.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame to manipulate.
    features : list(str)
        List of features to encode from the data.
    preprocess : {"label_encoder", "standardise", "power_transform"}
        Preprocessing function.

    Returns
    -------
    processed_data : pd.DataFrame
        Processed DataFrame.

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

    if type(features) is not type([]):
        raise TypeError(f"features should be a list, but got {type(features)}")

    if type(features[0]) is not type(""):
        raise TypeError(
            f"features should be a list(str), but got list({type(features[0])}"
        )

    # Create a copy of data to work on
    processed_data = deepcopy(data)

    for feature in features:
        match preprocess:
            case "label_encoder":
                processed_data[feature] = LabelEncoder().fit_transform(data[feature])
            case "standardise":
                processed_data[feature] = StandardScaler().fit_transform(
                    np.expand_dims(data[feature].to_numpy(), axis=1)
                )
            case "power_transform":
                processed_data[feature] = PowerTransformer().fit_transform(
                    np.expand_dims(data[feature].to_numpy(), axis=1)
                )
            case _:
                raise ValueError(f"{preprocess} invalid preprocessing function")

    return processed_data


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
