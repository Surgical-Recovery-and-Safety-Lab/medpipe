"""
Preprocessing functions module.

This module provides functions to preprocess data before training.

Functions:
- split_test_train: Splits the data indices into train and test sets.
- convert_object_to_categorical: Converts object columns to categoricals.
- label_encode_data: Encodes data columns with a label encoder.
- extract_labels: Extracts prediction labels from data.
"""

import numpy as np
import pandas as pd
import sklearn as skl

from pyrisk.utils.exceptions import array_check


def split_test_train(data_idx, n_folds=1, **kwargs):
    """
    Split data indices into train and test sets.

    Parameters
    ----------
    data_ind : array-like
        Data indices to split.
    n_folds : int, default: 1
        Number of folds to compute.
    **kwargs
        Extra arguments for the train_test_split function or KFold class.

    Returns
    -------
    fold_indices : dict[int, tuple(array-like, array-like)]
        Dictionary containing the train and test indices.
        The key is the fold number and the value is a tuple with the first
        element the train indices and the second element is the test indices.

    Raises
    ------
    TypeError
        If data_idx is not an array-like.
    TypeError
        If n_folds is not an integer.
    ValueError
        If n_folds is less than 1.

    """
    array_check(data_idx)

    if type(n_folds) is not type(1):
        raise TypeError(f"n_folds should be an int, but got {type(n_folds)}")
    if n_folds < 1:
        raise ValueError(f"n_folds should be greater than 1, but got {n_folds}")

    fold_indices = dict()

    if n_folds == 1:
        # Split into a single test and train set
        train_idx, test_idx = skl.model_selection.train_test_split(data_idx, **kwargs)
        fold_indices.update({n_folds: (train_idx, test_idx)})
        return fold_indices

    # Create the correct argument dict for KFold
    args_dict = dict()
    for key, value in kwargs.items():
        if key == "random_state" or key == "shuffle":
            if value == -1:
                # If random_state is -1 then convert to None
                value = None

            args_dict.update({key: value})

    # Split into n_folds
    kf = skl.model_selection.KFold(n_splits=n_folds, **args_dict)
    for i, fold_idx in enumerate(kf.split(data_idx)):
        fold_indices.update({i: fold_idx})

    return fold_indices


def temporal_k_fold(
    data: pd.Series, n_folds: int = 5, feature_name: str = "OP_YEAR"
) -> dict[int, list[int]]:
    """
    Splits the data to create temporal based train/test sets.

    The function returns the index of examples of all years but one for
    the training set and the other year as the testing set.
    The newest data will be the first testing fold.

    Parameters
    ----------
    data
        Subset of the data with the temporal feature to split with.
    n_folds : default: 5
        Number of folds to create.
    feature_name : default: "OP_YEAR"
        Name of the feature used to split the data temporally.

    Returns
    -------
    fold_indices : dict[int, tuple(array-like, array-like)]
        Dictionary containing the train and test indices.
        The key is the year number and the value is a tuple with the first
        element the train indices and the second element is the test indices.

    Raises
    ------
    TypeError
        If data is not a pd.Series.
    TypeError
        If n_folds is not an integer.
    ValueError
        If n_folds is less than 1.
    ValueError
        If n_folds is greater than number of unique years.
    KeyError
        If feature_name is not in data.

    """
    if type(data) is not type(pd.Series()):
        raise TypeError(f"data should be a pd.Series, but got {type(data)}")

    if type(n_folds) is not type(1):
        raise TypeError(f"n_folds should be an integer, but got {type(n_folds)}")

    if n_folds < 1:
        raise ValueError(f"n_folds should be greater than 0, but got {n_folds}")

    if feature_name != data.name:
        raise KeyError(f"{feature_name} is not a column in data")

    years = data.unique()  # List of all the years

    if n_folds > len(years):
        raise ValueError(
            f"Too many folds, n_folds is {n_folds} and number of years is {len(years)}"
        )

    fold_years = reversed(years[len(years) - n_folds :])  # Years to create folds with
    fold_indices = {}  # Empty dict for the fold indices

    for year in fold_years:
        test_idx = data.loc[data == year].index.to_numpy()
        train_idx = np.delete(data.index.to_numpy(), test_idx)
        fold_indices.update({year: (train_idx, test_idx)})

    return fold_indices


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


def label_encode_data(data, features):
    """
    Encodes data columns with a label encoder.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame to manipulate.
    features : list(str)
        List of features to encode from the data.

    Returns
    -------
    processed_data : pd.DataFrame
        Processed DataFrame.

    Raises
    ------
    TypeError
        If data is not a pd.DataFrame.
    TypeError
        If features is not a list(str)
    KeyError
        If a features is not a valid key.

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
    processed_data = data

    for feature in features:
        processed_data[feature] = skl.preprocessing.LabelEncoder().fit_transform(
            data[feature]
        )

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
