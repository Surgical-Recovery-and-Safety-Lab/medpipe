"""
Preprocessing functions module.

This module provides functions to preprocess data before training.

Functions:
- test_train_it: Creates a KFold iterator to split data into
    test and train sets.
- convert_object_to_categorical: Converts object columns to categoricals.
- label_encode_data: Encodes data columns with a label encoder.
- extract_labels: Extracts prediction labels from data.
"""

import pandas as pd
import sklearn as skl


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
