"""
Models functions module.

This module provides functions to core functions for models and pipelines.

Functions:
- create_model: Creates a new model.
- test_model: Tests a model on some test data.
- save_pipeline: Pickles a pipeline.
- load_pipeline: Loads a pickled pipeline.
- get_positive_proba: Returns just the positive label probabilities of the each class.
- get_full_proba: Returns probabilities for both labels.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from medpipe._types import FullProba, Labels, MetricDict, Model, PosProba
from medpipe.metrics.core import compute_pred_metrics, compute_score_metrics
from medpipe.utils.exceptions import array_check, file_checks
from medpipe.utils.logger import print_message

SCRIPT_NAME = "models/core"

if TYPE_CHECKING:
    import logging

    from medpipe.pipeline.pipeline import Pipeline


def create_model(
    model_type: str,
    logger: logging.Logger | None = None,
    quiet: bool = False,
    **config_params: Any,
) -> Model:
    """
    Creates a AI model.

    Parameters
    ----------
    model_type : {"hgb-c", "logistic", "isotonic"}
        Type of model to create.
            hgb-c: histogram gradient boosting classifier.
            logistic: logistic regression.
            isotonic: isotonic regression.
    quiet : bool, default: False
        Flag to create a model without printing.
    **config_params : dict[str, Any]
        Configuration parameters for the model.

    Returns
    -------
    model : Model
        Created model.

    Raises
    ------
    TypeError
        If model_type is not a str.
        If an unexpected keyword argument is present.
    ValueError
        If model_type is not "hgb-c", "logistic" or "isotonic".

    """
    if type(model_type) is not str:
        raise TypeError(f"{model_type} shoud be a string")

    match model_type:
        case "hgb-c":
            if not quiet:
                print_message(
                    "Creating a Histogram Gradient Boosting Classifier",
                    logger,
                    SCRIPT_NAME,
                )
            model = HistGradientBoostingClassifier(**config_params)
        case "logistic":
            if not quiet:
                print_message(
                    "Creating a Logistic Regression calibrator", logger, SCRIPT_NAME
                )
            model = LogisticRegression(**config_params)

        case "isotonic":
            if not quiet:
                print_message(
                    "Creating an Isotonic Regression calibrator", logger, SCRIPT_NAME
                )
            model = IsotonicRegression(**config_params)

        case _:
            raise ValueError(f"{model_type} invalid model type. See function docstring")

    return model


def test_model(y_test: Labels, y_pred: Labels, y_pred_proba: FullProba) -> MetricDict:
    """
    Computes different metrics to test the model.

    Parameters
    ----------
    y_test : Labels
        Ground truth test labels of shape (n_samples, n_classes).
    y_pred : Labels
        Predicted labels of shape (n_samples, n_classes).
    y_pred_proba : FullProba
        Predicted probabilities of shape (n_samples, 2).

    Returns
    -------
    metric_dict : MetricDict
        Dictionary of the model performance for one fold.
        Keys are the metric name and values are the metric value.
        The test metrics used are:
         - accuracy
         - f1
         - precision
         - recall
         - log_loss
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
    array_check(y_pred)
    array_check(y_pred_proba)

    metric_dict = compute_pred_metrics(
        ["accuracy", "f1", "recall", "precision"], y_test, y_pred
    )
    metric_dict.update(
        compute_score_metrics(
            ["roc", "auroc", "prc", "ap", "log_loss"], y_test, y_pred_proba
        )
    )
    return metric_dict


def save_pipeline(
    pipeline: Pipeline, save_file: str, extension: str = ".joblib"
) -> None:
    """
    Saves a Pipeline to file.

    Parameters
    ----------
    pipeline : Pipeline
        Pipeline to save.
    save_file : str
        Path to the file to save the model.
    extension : str, default: ".joblib"
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
        joblib.dump(pipeline, f, compress=3)


def load_pipeline(load_file: str) -> Pipeline:
    """
    Loads a saved Pipeline from a .joblib file.

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
        If load_file extension is not .joblib file.

    """
    file_checks(load_file, ".joblib")

    with open(load_file, "rb") as f:
        pipeline = joblib.load(f)

    return pipeline


def get_positive_proba(probabilities: FullProba) -> PosProba:
    """
    Returns just the positive label probabilities of the each class.

    Parameters
    ----------
    probabilities : FullProba
        Full probabilities for each class.

    Returns
    -------
    pos_proba : PosProba
        Probabilities of the positive labels for each class.

    """
    if type(probabilities) is type(np.array([])):
        return np.expand_dims(probabilities[:, 1], 1)

    pos_proba = np.zeros((probabilities[0].shape[0], len(probabilities)))
    for i, proba in enumerate(probabilities):
        pos_proba[:, i] = proba[:, 1]

    return pos_proba


def get_full_proba(pos_proba: PosProba) -> FullProba:
    """
    Returns probabilities for both labels.

    Parameters
    ----------
    pos_proba : PosProba
        Probabilities of the positive labels for each class.

    Returns
    -------
    probabilities : FullProba
        Full probabilities for each class.

    """
    probabilities = []  # Empty list for the probabilities

    for i in range(pos_proba.shape[1]):
        probabilities.append(np.array([1 - pos_proba[:, i], pos_proba[:, i]]).T)

    if len(probabilities) == 1:
        return probabilities[0]
    else:
        return probabilities
