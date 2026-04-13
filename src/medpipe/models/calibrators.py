"""
Implements the calibrator classes.

These classes create a Calibrator to calibrate predictions.

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, cast

from numpy import array, expand_dims, round
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from medpipe._types import Calibrator, FProbas, Labels, PProbas, R

from .core import create_model, get_full_proba

SCRIPT_NAME = "models/calibrators"
if TYPE_CHECKING:
    import logging


def create_calibrator(
    model_type: str,
    hyperparameters: dict[str, Any] = {},
    logger: logging.Logger | None = None,
) -> Calibrator:
    """
    Creates a calibrator instance.

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
    calibrator : Calibrator
        Calibrator instance.

    Raises
    ------
    ValueError
        If model_type is not valid.

    """

    match model_type:
        case "isotonic":
            return IsotonicCalibrator(hyperparameters, logger)
        case "logistic":
            return LogisticCalibrator(hyperparameters, logger)
        case _:
            raise ValueError(
                f"model type should be isotonic or logistic, but got {model_type}"
            )


class BaseCalibrator(ABC, Generic[R]):
    """
    Class that creates an abstract BaseCalibrator.

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
    _set_model(quiet=False)
        Set the model to default parameters.
    fit(X, y)
        Fits the predictor based on input data.
    predict_proba(X)
        Predicts probabilities from input data.
    predict(X)
        Predicts labels from input data.
    """

    model: R  # Define model as a generic Regressor

    def __init__(
        self,
        model_type: str,
        hyperparameters: dict[str, Any] = {},
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialise a BaseCalibrator class instance.

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
        self.model = cast(
            R,
            create_model(
                self.model_type,
                logger=self.logger,
                quiet=quiet,
                **self.hyperparameters,
            ),
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
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        self.model.fit(X, y)

    @abstractmethod
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
        pass

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
        X = X.reshape(-1, 1) if X.ndim == 1 else X  # Reshape if 1 dimension only
        labels = round(self.model.predict(X))
        return array(labels, dtype=int).T


class LogisticCalibrator(BaseCalibrator[LogisticRegression]):
    """
    Class that creates an IsotonicCalibrator.

    Attributes
    ----------
    model : LogitsticRegression
        Calibrator model.
    model_type : "logistic"
        Model type.
    logger : logging.Logger | None, default: None
        Logger object to log prints. If None print to terminal.

    Methods
    -------
    __init__(hyperparameters={}, logger=None):
        Initialise a LogisticCalibrator class instance.
    _set_model(quiet=False)
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
        hyperparameters: dict[str, Any] = {},
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialise a LogisticCalibrator class instance.

        Parameters
        ----------
        hyperparameters : dict[str, Any]
            Model hyperparameter dictionary.
        logger : logging.Logger | None, default: None
            Logger object to log prints. If None print to terminal.

        Returns
        -------
        None
            Nothing is returned.

        """
        super().__init__("logistic", hyperparameters, logger)

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
        X = X.reshape(-1, 1) if X.ndim == 1 else X  # Reshape if 1 dimension only
        return self.model.predict_proba(X)


class IsotonicCalibrator(BaseCalibrator[IsotonicRegression]):
    """
    Class that creates an IsotonicCalibrator.

    Attributes
    ----------
    model : Regressor
        Calibrator model.
    model_type : "isotonic"
        Model type.
    logger : logging.Logger | None, default: None
        Logger object to log prints. If None print to terminal.

    Methods
    -------
    __init__(model_type, hyperparameters={}, logger=None):
        Initialise an IsotonicCalibrator class instance.
    _set_model(quiet=False)
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
        hyperparameters: dict[str, Any] = {},
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialise a IsotonicCalibrator class instance.

        Parameters
        ----------
        hyperparameters : dict[str, Any]
            Model hyperparameter dictionary.
        logger : logging.Logger | None, default: None
            Logger object to log prints. If None print to terminal.

        Returns
        -------
        None
            Nothing is returned.

        """
        super().__init__("isotonic", hyperparameters, logger)

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
        X = X.reshape(-1, 1) if X.ndim == 1 else X  # Reshape if 1 dimension only
        predictions = self.model.predict(X)
        return get_full_proba(expand_dims(predictions, 1))
