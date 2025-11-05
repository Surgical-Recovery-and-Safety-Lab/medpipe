"""
Models functions module.

This module provides functions to create, train, and test models.

Functions:
- create_model: Creates a new model.
- train_model: Trains a given model on some train data.
- test_model: Tests a model on some test data.
- save_model: Pickles a model.
- load_model: Loads a pickled model.
"""

import pickle

import sklearn as skl

from pyrisk.utils.exceptions import array_check, array_dim_check, file_checks


def create_model(model_type: str, **config_params: dict):
    """
    Creates a AI model.

    Parameters
    ----------
    model_type : {"hgb", "svm"}
        Type of model to create.
            hgb: histogram gradient boosting.
            svm: support vector machine.
    **config_params
        Configuration parameters for the model.

    Returns
    -------
    model : skl.ensemble.HistGradBoostingClassifier or skl.svm.SVC
        Created model.

    Raises
    ------
    TypeError
        If model_type is not a str.
        If an unexpected keyword argument is present.
    ValueError
        If model_type is not "hgb" or "svm".

    """
    if type(model_type) is not str:
        raise TypeError(f"{model_type} shoud be a string")

    match model_type:
        case "hgb":
            print("Creating a Histogram Gradient Boosting model")
            model = skl.ensemble.HistGradientBoostingClassifier(**config_params)

        case "svm":
            print("Creating a Support Vector Machine model")
            model = skl.svm.SVC(**config_params)

        case _:
            raise ValueError(f"{model_type} invalid model type. See function docstring")

    return model


def train_model(model, X, y, sample_weight=None) -> None:
    """
    Trains an AI model.

    Parameters
    ----------
    model : skl.ensemble.HistGradBoostingClassifier or skl.svm.SVC
        Model to train.
    X : array-like of shape (n_samples, n_features)
        Training data.
    y : array-like of shape (n_samples, n_classes)
        Prediction labels.
    sample_weight : array-like of shape (n_samples, n_classes), default: None
        Weight of each sample to help address class imbalance.

    Returns
    -------
    None
        Nothing is returned.

    Raises
    ------
    TypeError
        If X, y, or sample_weight are not an array-like.
    ValueError
        If the X and y do not have the same dimensions.
        If the y and sample_weight do not have the same dimensions.

    """
    # Check that inputs are correct
    array_check(X)
    array_check(y)
    array_dim_check(X, y, 0)

    if sample_weight:
        array_check(sample_weight)
        array_dim_check(y, sample_weight)

    model.fit(X, y, sample_weight=sample_weight)


def test_model(model, X, y, sample_weight=None) -> float:
    """
    Trains an AI model.

    Parameters
    ----------
    model : skl.ensemble.HistGradBoostingClassifier or skl.svm.SVC
        Model to test.
    X : array-like of shape (n_samples, n_features)
        Testing data.
    y : array-like of shape (n_samples, n_classes)
        Prediction labels.
    sample_weight : array-like of shape (n_samples, n_classes), default: None
        Weight of each sample to help address class imbalance.

    Returns
    -------
    score : float
        Accuracy of the model.

    Raises
    ------
    TypeError
        If X, y, or sample_weight are not an array-like.
    ValueError
        If the X and y do not have the same dimensions.
        If the y and sample_weight do not have the same dimensions.

    """
    # Check that inputs are correct
    array_check(X)
    array_check(y)
    array_dim_check(X, y, 0)

    if sample_weight:
        array_check(sample_weight)

    return model.score(X, y, sample_weight=sample_weight)


def save_model(model, save_file) -> None:
    """
    Saves an AI model to a pkl file.

    Parameters
    ----------
    model : skl.ensemble.HistGradBoostingClassifier or skl.svm.SVC
        Model to save.
    save_file : str
        Path to the file to save the model.

    Returns
    -------
    None
        Nothing is returned.

    Raises
    ------
    TypeError
        If save_file is not a str.
    FileNotFoundError
        If save_file does not exist.
    IsADirectoryError
        If save_file is a directory.
    ValueError
        If save_file extension is not .pkl file.

    """
    file_checks(save_file, ".pkl")

    with open(save_file, "wb") as f:
        pickle.dump(model, f)


def load_model(load_file: str):
    """
    Loads an AI model from a .pkl file.

    Parameters
    ----------
    load_file : str
        Path to the file to load the model from.

    Returns
    -------
    model : skl.ensemble.HistGradBoostingClassifier or skl.svm.SVC
        Loaded model.

    Raises
    ------
    TypeError
        If load_file is not a str.
    FileNotFoundError
        If load_file does not exist.
    IsADirectoryError
        If load_file is a directory.
    ValueError
        If load_file extension is not .pkl file.

    """
    file_checks(load_file, ".pkl")

    with open(load_file, "rb") as f:
        model = pickle.load(f)

    return model
