"""
Models functions module.

This module provides functions to core functions for models and pipelines.

Functions:
- create_estimator: Creates an AI model.
- save_pipeline: Saves a pipeline with joblib.
- load_pipeline: Loads a pipeline with joblib.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Type

import joblib
import ngboost
import sklearn
from sklearn.base import is_regressor
from sklearn.compose import TransformedTargetRegressor

from medpipe.data.transformers import BoundedLogitTransformer
from medpipe.utils.exceptions import file_checks

SCRIPT_NAME = "models/core"

if TYPE_CHECKING:

    from sklearn.base import BaseEstimator

    from medpipe.pipeline.pipeline import MedpipePipeline


def create_estimator(model_type: str, **hyperparameters) -> BaseEstimator:
    """
    Creates an AI model based on the input model type.

    Parameters
    ----------
    model_type : str
        Type of estimator to create.
    **hyperparameters
        Configuration parameters for the estimator.

    Returns
    -------
    estimator : Model
        Created estimator.

    Raises
    ------
    TypeError
        If model_type is not a str.
        If an unexpected keyword argument is present.

    """
    if type(model_type) is not str:
        raise TypeError(f"{model_type} should be a string")

    estimator_cls = _check_model_type(model_type)  # Estimator class
    estimator = estimator_cls(**hyperparameters)

    if is_regressor(estimator) or isinstance(estimator, ngboost.NGBRegressor):
        return TransformedTargetRegressor(
            regressor=estimator,
            transformer=BoundedLogitTransformer(),
            check_inverse=False,  # Avoid error because of clipping
        )

    return estimator


def _check_model_type(model_type: str) -> Type:
    """
    Internal function that checks if the model type is correct.

    Currently checks the sklearn.ensemble, sklearn.isotonic, and
    sklearn.linear_model.

    Parameters
    ----------
    model_type : str
        Estimator name.

    Returns
    -------
    estimator : Type
        Estimator class.

    Raises
    ------
    ValueError
        If the model type is not a valid class in one of the modules.

    """
    if hasattr(sklearn.ensemble, model_type):
        return getattr(sklearn.ensemble, model_type)

    if hasattr(sklearn.linear_model, model_type):
        return getattr(sklearn.linear_model, model_type)

    if hasattr(sklearn.isotonic, model_type):
        return getattr(sklearn.isotonic, model_type)

    if hasattr(ngboost, model_type):
        return getattr(ngboost, model_type)

    raise ValueError(
        f"{model_type} is not found in sklearn.ensemble, sklearn.linear_model, "
        "sklearn.isotonic, or ngboost, please check that the operation matches"
    )


def save_pipeline(pipeline: MedpipePipeline, save_file: str | Path) -> None:
    """
    Saves a MedpipePipeline to a .joblib file.

    Parameters
    ----------
    pipeline : Pipeline
        Pipeline to save.
    save_file : str | Path
        Path to the file to save the model.

    Returns
    -------
    None
        Nothing is returned.

    """
    file_checks(save_file, ".joblib", exists=False)
    with open(save_file, "wb") as f:
        joblib.dump(pipeline, f, compress=3)


def load_pipeline(load_file: str | Path) -> MedpipePipeline:
    """
    Loads a saved MedpipePipeline from a .joblib file.

    Parameters
    ----------
    load_file : str | Path
        Path to the .joblib file to load the Pipeline from.

    Returns
    -------
    pipeline : Pipeline
        Loaded pipeline.

    """
    file_checks(load_file, ".joblib")

    with open(load_file, "rb") as f:
        pipeline = joblib.load(f)

    return pipeline
