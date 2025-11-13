"""
Preprocessing functions module.

This module provides functions to preprocess data before training.

Functions:
- split_test_train: Splits the data into train and test sets.
- convert_object_to_categorical: Converts object columns to categoricals.
- label_encode_data: Encodes data columns with a label encoder.
- extract_labels: Extracts prediction labels from data.
"""

import pandas as pd
import sklearn as skl


def split_test_train(
    data: pd.DataFrame, train_size: float = 0.8, random_state: int = 42
):
    """
    Split data into train and test sets.

    Parameters
    ----------
    data
        Data to split.
    train_size : float, default: 0.8
        Size of the training set, between 0.0 and 1.0.
    random_state : int, default: 42
        Random seed to use, for reproducibility.

    Returns
    -------
    train_set : pd.DataFrame
        Train subset for the data.
    test_set : pd.DataFrame
        Test subset for the data.

    Raises
    ------
    TypeError
        If train_size is not a float.
    TypeError
        If data is not a pd.DataFrame.
    ValueError
        If train_size is not between 0.0 and 1.0.

    """
    # Check that train_size is correct
    if type(train_size) is not type(0.0):
        raise TypeError(f"train_size should be a float, but got {type(train_size)}")
    if train_size < 0.0 or train_size > 1.0:
        raise ValueError(
            f"train_size should be between 0.0 and 1.0, but got {train_size}"
        )
    if type(data) is not type(pd.DataFrame()):
        raise TypeError(f"data should be a pd.DataFrame, but got {type(data)}")

    train_set, test_set = skl.model_selection.train_test_split(
        data, train_size=train_size, random_state=random_state
    )

    return train_set, test_set


def temporal_k_fold(
    data: pd.Series, n_folds: int = 5, feature_name: str = "OP_YEAR"
) -> dict[int, list[int]]:
    """
    Splits the data to create temporal based validation sets.

    The function returns the index of examples for a specific year.
    The newest data will be the first fold.

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
    fold_indices
        Temporal fold indices.

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
    fold_indices = {}  # Empyt dict for the fold indices

    for year in fold_years:
        indices = data.loc[data == year].index
        fold_indices.update({year: indices})

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
