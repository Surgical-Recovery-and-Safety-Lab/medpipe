"""
Preprocessing functions module.

This module provides functions to preprocess data before training.

Functions:
- split_test_train: Splits the data into train and test sets.
"""

import pandas as pd
import sklearn as skl


def split_test_train(
    data: pd.DataFrame, train_size: float = 0.2, random_state: int = 42
):
    """
    Split data into train and test sets.

    Parameters
    ----------
    data
        Data to split.
    train_size : float, default: 0.2
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
        raise TypeError("train_size should be a float")
    if train_size < 0.0 or train_size > 1.0:
        raise ValueError("train_size should be between 0.0 and 1.0")
    if type(data) is not type(pd.DataFrame()):
        raise TypeError("data should be a pd.DataFrame")

    train_set, test_set = skl.model_selection.train_test_split(
        data, train_size=train_size, random_state=random_state
    )

    return train_set, test_set
