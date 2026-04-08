"""
Predictor class.

This class creates a Predictor to train and make predictions.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from medpipe._types import Classifier, FProbas, Labels

from .core import create_model

SCRIPT_NAME = "models/Predictor"
if TYPE_CHECKING:
    import logging

    import numpy.typing as npt
    import pandas as pd


class Predictor:
    """
    Class that creates a Predictor.

    Attributes
    ----------
    model : Classifier
        Predictor model. The key is the predicted label, the model
        is a HistGradientBoostingClassifier.
    model_type : {"hgb-c"}
        Model type.
    hyperparameters : dict[str, Any]
        Model hyperparameter dictionary.
    logger : logging.Logger | None, default: None
        Logger object to log prints. If None print to terminal.

    Methods
    -------
    __init__(model_type, hyperparameters, logger=None)
        Initialise a Predictor class instance.
    _set_model()
        Set the model to default parameters.
    fit(X_train, y_train, weights=None):
        Fits the predictor based on input data.
    predict_proba(X)
        Predicts probabilities from input data.
    predict(X)
        Predicts labels from input data.
    """

    def __init__(
        self,
        model_type: str,
        hyperparameters: dict[str, Any],
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialise a Predictor class instance.

        Parameters
        ----------
        model_type : {"hgb-c"}
            Model type.
        hyperparameters : dict[str, Any]
            Model hyperparameter dictionary.
        logger : logging.Logger | None, default: None
            Logger object to log prints. If None print to terminal.

        Returns
        -------
        None
            Nothing is returned.

        """
        self.model_type = model_type
        self.hyperparameters = hyperparameters
        self.logger = logger

        # Create model based on attributes
        self._set_model()

    def _set_model(self, quiet: bool = False) -> None:
        """
        Set the model to default parameters.

        Parameters
        ----------
        quiet : bool, default: False
            Flag to create a model without printing.

        Returns
        -------
        None
            Nothing is returned.

        """
        self.model: Classifier = create_model(
            self.model_type,
            self.logger,
            quiet=quiet,
            **self.hyperparameters,
        )

    def fit(
        self, X_train: pd.DataFrame, y_train: Labels, weights: npt.NDArray | None = None
    ) -> None:
        """
        Fits the predictor to the training data.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training data.
        y_train : Labels
            Prediction labels.
        weights : npt.NDArray | None, default: None
            Weights to address class imbalance.

        Returns
        -------
        None
            Nothing is returned.

        """
        self.model.fit(X_train, y_train.squeeze(), sample_weight=weights)

    def predict_proba(self, X: pd.DataFrame) -> FProbas:
        """
        Predicts probabilities from input data.

        Parameters
        ----------
        X : pd.DataFrame
            Training data.

        Returns
        -------
        probabilities : FProbas
            Predicted probabilities.

        """
        return self.model.predict_proba(X)

    def predict(self, X: pd.DataFrame) -> Labels:
        """
        Predicts labels from input data.

        Parameters
        ----------
        X : pd.DataFrame
            Training data.

        Returns
        -------
        labels : Labels
            Predicted labels.

        """
        return self.model.predict(X)
