"""
Predictor class.

This class creates a Predictor to train and make predictions.

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, cast

from medpipe._types import C, FullProba, Labels, PredData, Predictor
from medpipe.data.utils import convert_data

from .core import create_model

SCRIPT_NAME = "models/predictors"
if TYPE_CHECKING:
    import logging

    import numpy.typing as npt


def create_predictor(
    model_type: str,
    hyperparameters: dict[str, Any] = {},
    logger: logging.Logger | None = None,
) -> Predictor:
    """
    Creates a predictor instance.

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
    predictor : Predictor
        Predictor instance.

    Raises
    ------
    ValueError
        If model_type is not valid.

    """

    match model_type:
        case "hgb-c":
            return HGBClassifier(hyperparameters, logger)
        case _:
            raise ValueError(f"model type should be hgb-c, but got {model_type}")


class BasePredictor(ABC, Generic[C]):
    """
    Class that creates an abstract BasePredictor.

    Attributes
    ----------
    model : Classifier
        Predictor model.
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
    _set_model(quiet=False)
        Set the model to default parameters.
    fit(X_train, y_train, weights=None):
        Fits the predictor based on input data.
    predict_proba(X)
        Predicts probabilities from input data.
    predict(X)
        Predicts labels from input data.
    """

    model: C

    def __init__(
        self,
        model_type: str,
        hyperparameters: dict[str, Any] = {},
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialise a BasePredictor class instance.

        Parameters
        ----------
        model_type : {"hgb-c"}
            Model type.
        hyperparameters : dict[str, Any], default={}
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
        self.model = cast(  # Cast model to a Classifier
            C,
            create_model(
                self.model_type,
                self.logger,
                quiet=quiet,
                **self.hyperparameters,
            ),
        )

    def fit(
        self, X_train: PredData, y_train: Labels, weights: npt.NDArray | None = None
    ) -> None:
        """
        Fits the predictor to the training data.

        Parameters
        ----------
        X_train : PredData
            Training data of shape (n_samples, n_features).
        y_train : Labels
            Prediction labels of shape (n_samples,).
        weights : npt.NDArray | None, default: None
            Weights to address class imbalance.

        Returns
        -------
        None
            Nothing is returned.

        """
        train_data = convert_data(X_train)
        self.model.fit(train_data, y_train.squeeze(), sample_weight=weights)

    def predict_proba(self, X: PredData) -> FullProba:
        """
        Predicts probabilities from input data.

        Parameters
        ----------
        X : PredData
            Training data of shape (n_samples, n_features).

        Returns
        -------
        probabilities : FullProba
            Predicted probabilities of shape (n_samples, 2).

        """
        return self.model.predict_proba(convert_data(X))

    @abstractmethod
    def predict(self, X: PredData) -> Labels:
        """
        Predicts labels from input data.

        Parameters
        ----------
        X : PredData
            Training data of shape (n_samples, n_features).

        Returns
        -------
        labels : Labels
            Predicted labels of shape (n_samples,).

        """
        pass


class HGBClassifier(BasePredictor):
    """
    Class that creates a HGBClassifier.

    Attributes
    ----------
    model : HistGradientBoostingClassifier
        Predictor model.
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
    _set_model(quiet=False)
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
        hyperparameters: dict[str, Any],
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialise a HGBClassifier class instance.

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
        super().__init__("hgb-c", hyperparameters, logger)

    def predict(self, X: PredData) -> Labels:
        """
        Predicts labels from input data.

        Parameters
        ----------
        X : PredData
            Training data of shape (n_samples, n_features).

        Returns
        -------
        labels : Labels
            Predicted labels of shape (n_samples,).

        """
        return self.model.predict(convert_data(X))
