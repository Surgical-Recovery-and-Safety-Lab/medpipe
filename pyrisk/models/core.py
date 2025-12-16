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
from copy import deepcopy

import numpy as np
import pandas as pd
import sklearn as skl
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from torch.accelerator import current_accelerator, is_available

import pyrisk.data.weighting as weight
from pyrisk.data.preprocessing import extract_labels, get_validation_idx
from pyrisk.data.sampler import data_sampler
from pyrisk.metrics.core import (
    compute_pred_metrics,
    compute_score_metrics,
    print_metrics,
)
from pyrisk.models.AIRiskNN import AIRiskNN
from pyrisk.utils.exceptions import array_check, array_dim_check, file_checks
from pyrisk.utils.logger import print_message

SCRIPT_NAME = "models/core"


def create_model(
    model_type: str,
    n_features: int = -1,
    n_classes: int = 1,
    logger=None,
    **config_params,
):
    """
    Creates a AI model.

    Parameters
    ----------
    model_type : {"hgb", "svm", "nn", "logistic", "isotonic"}
        Type of model to create.
            hgb: histogram gradient boosting.
            svm: support vector machine.
            nn: AIRiskNN neural network.
            logistic: logistic regression.
            isotonic: isotonic regression.
    n_features : int, default: -1
        Number of features in the data, only needed for NN models.
    n_classes : int, default: 1
        Number of classes. Used to call MultiOutputClassifier.
    logger : logging.Logger, default: None
        Logger object to log prints. If None print to terminal.
    **config_params
        Configuration parameters for the model.

    Returns
    -------
    model : HistGradBoostingClassifier, SVC, AIRiskNN,
            LogisticRegression, IsotonicRegression,
            MultiOutputClassifier or MultiOuputRegressor.
        Created model.
        If n_classes > 1 and not AIRiskNN then MultiOutput models are created.

    Raises
    ------
    TypeError
        If model_type is not a str.
        If an unexpected keyword argument is present.
    ValueError
        If model_type is not "hgb", "svm", "nn", "logistic" or "isotonic".

    """
    if type(model_type) is not str:
        raise TypeError(f"{model_type} shoud be a string")

    match model_type:
        case "hgb":
            print_message(
                "Creating a Histogram Gradient Boosting model", logger, SCRIPT_NAME
            )
            model = skl.ensemble.HistGradientBoostingClassifier(**config_params)

            if n_classes > 1:
                model = MultiOutputClassifier(model)

        case "svm":
            print_message(
                "Creating a Support Vector Machine model", logger, SCRIPT_NAME
            )
            model = skl.svm.SVC(**config_params)

            if n_classes > 1:
                model = MultiOutputClassifier(model)

        case "nn":
            print_message("Creating a Neural Network model", logger, SCRIPT_NAME)

            if n_features == -1:
                raise ValueError("For nn models, please specify feature number")

            device = current_accelerator().type if is_available() else "cpu"
            print_message(f"Using {device} device", logger, SCRIPT_NAME)
            model = AIRiskNN(n_features, logger, **config_params).to(device)

        case "logistic":
            print_message(
                "Creating a Logistic Regression calibrator", logger, SCRIPT_NAME
            )
            model = LogisticRegression(**config_params)

            if n_classes > 1:
                model = MultiOutputRegressor(model)

        case "isotonic":
            print_message(
                "Creating an Isotonic Regression calibrator", logger, SCRIPT_NAME
            )
            model = IsotonicRegression(**config_params)

            if n_classes > 1:
                model = MultiOutputRegressor(model)
        case _:
            raise ValueError(f"{model_type} invalid model type. See function docstring")

    return model


def test_model(y_test, y_pred, y_pred_proba):
    """
    Computes different metrics to test the model.

    Parameters
    ----------
    model : HistGradBoostingClassifier, SVC, or AIRiskNN
        Model to test.
    X_test : array-like of shape (n_samples, n_features)
        Test data to make predictions on.
    y_test : array-like of shape (n_samples, n_classes)
        Ground truth test labels.

    Returns
    -------
    metric_dict : dict[str, dict[str, list[float or tuple(array-like)]]
        Dictionary of the model performance for one fold.
        Keys are the metric name and values are the metric value.
        The test metrics used are:
         - accuracy
         - f1
         - precision
         - recall
         - roc (Receiver Operator Characteristic)
         - auroc (Area Under Receiver Operator Characteristic)
         - prc (Precision-Recall Curve)
         - ap (Average Precision)

    Raises
    ------
    TypeError
        If X_test or y_test are not an array-like.
    ValueError
        If X_test and y_test do not have the same dimensions.

    """
    # Check that inputs are correct
    array_check(X_test)
    array_check(y_test)
    array_dim_check(X_test, y_test, 0)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    metric_dict = compute_pred_metrics(
        ["accuracy", "f1", "recall", "precision"], y_test, y_pred
    )
    metric_dict.update(
        compute_score_metrics(["roc", "auroc", "prc", "ap"], y_test, y_pred_proba)
    )
    return metric_dict


def save_pipeline(pipeline, save_file, extension=".pkl") -> None:
    """
    Saves a Pipeline to file.

    Parameters
    ----------
    pipeline : Pipeline
        Pipeline to save.
    save_file : str
        Path to the file to save the model.
    extension : str, default: ".pkl"
        Extension of the save file.

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
        If save_file extension is not extension.

    """
    file_checks(save_file, extension, exists=False)

    with open(save_file, "wb") as f:
        pickle.dump(pipeline, f)


def load_pipeline(load_file: str):
    """
    Loads a saved Pipeline from a .pkl file.

    Parameters
    ----------
    load_file : str
        Path to the file to load the Pipeline from.

    Returns
    -------
    pipeline : Pipeline
        Loaded pipeline.

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
        pipeline = pickle.load(f)

    return pipeline


def get_positive_proba(probabilities):
    """
    Returns just the positive label probabilities of the each class.

    Parameters
    ----------
    probabilities : array-like of shape (n_classes, (n_samples, 2))
        Probabilities for each class.

    Returns
    -------
    pos_proba : array-like of shape (n_samples, n_classes)
        Probabilities of the positive labels for each class.

    """
    if type(probabilities) is type(np.array([])):
        return probabilities

    pos_proba = np.zeros((probabilities[0].shape[0], len(probabilities)))
    for i, proba in enumerate(probabilities):
        pos_proba[:, i] = proba[:, 1]

    return pos_proba


def get_full_proba(pos_proba):
    """
    Returns probabilities for both labels.

    Parameters
    ----------
    pos_proba : array-like of shape (n_samples, n_classes)
        Probabilities of the positive labels for each class.

    Returns
    -------
    probabilities : array-like of shape (n_classes, (n_samples, 2))
        Probabilities for each class.

    """
    probabilities = []  # Empty list for the probabilities

    for i in range(pos_proba.shape[1]):
        probabilities.append(np.array([1 - pos_proba[:, i], pos_proba[:, i]]).T)

    return probabilities
