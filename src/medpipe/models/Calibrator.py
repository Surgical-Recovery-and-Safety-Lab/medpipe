"""
Calibrator class.

This class creates a Calibrator to calibrate predictions.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from numpy import array, expand_dims, round
from sklearn.isotonic import IsotonicRegression

from medpipe._types import FProbas, Labels, Model, PProbas

from .core import create_model, get_full_proba

SCRIPT_NAME = "models/Calibrator"
if TYPE_CHECKING:
    import logging


class Calibrator:
    """
    Class that creates a Calibrator.

    Attributes
    ----------
    model : Regressor
        Calibrator model.
    model_type : {"logistic", "isotonic"}
        Model type.
    logger : logging.Logger | None, default: None
        Logger object to log prints. If None print to terminal.

    Methods
    -------
    __init__(model_type, hyperparameters={}, logger=None):
        Initialise a Calibrator class instance.
    _set_model()
        Set the model to default parameters.
    fit(X, y)
        Fits the predictor based on input data.
    predict_proba(X)
        Predicts probabilities from input data.
    predict(X)
        Predicts labels from input data.
    """

    def __init__(
        self,
        model_type: str,
        hyperparameters: dict[str, Any] = {},
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialise a Calibrator class instance.

        Parameters
        ----------
        model_type : {"logistic", "isotonic"}
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
        self.model: Model = create_model(
            self.model_type,
            logger=self.logger,
            quiet=quiet,
            **self.hyperparameters,
        )

    def fit(self, X: PProbas, y: Labels) -> None:
        """
        Fits the predictor to the training data.

        Parameters
        ----------
        X : PProbas
            Predicted probabilities from model.
        y : Labels
            Labels to predict.

        Returns
        -------
        None
            Nothing is returned.

        """
        self.model.fit(X, y)

    def predict_proba(self, X: PProbas) -> FProbas:
        """
        Predicts probabilities from input data.

        Parameters
        ----------
        X : PProbas
            Positive predicted probabilities from a Predictor.

        Returns
        -------
        probabilities : FProbas
            Predicted probabilities.

        """
        if isinstance(self.model, IsotonicRegression):
            predictions = self.model.predict(X)
            return get_full_proba(expand_dims(predictions, 1))
        else:
            return self.model.predict_proba(X)

    def predict(self, X: PProbas) -> Labels:
        """
        Predicts labels from input data.

        Parameters
        ----------
        X : PProbas
            Positive predicted probabilities from a Predictor.

        Returns
        -------
        predictions : PProbas
            Predicted labels.

        """
        labels = round(self.model.predict(X))
        return array(labels, dtype=int).T
