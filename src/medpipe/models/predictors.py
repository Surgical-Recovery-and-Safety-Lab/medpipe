"""
Predictor class.

This class creates a Predictor to train and make predictions.

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, cast

from numpy import ndarray

from medpipe._types import C, Data, FProbas, Labels, Predictor

from .core import create_model

SCRIPT_NAME = "models/predictors"
if TYPE_CHECKING:
    import logging

    import numpy.typing as npt
    import pandas as pd


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

    def _convert_data(self, data: Data) -> Data | None:
        """
        Convert data to a ndarray if possible.

        The function checks if all columns are numeric to assess convertability.
        If the data can be converted, the pd.DataFrame is converted to ndarray.
        If the data is already ndarray the data is returned.

        Parameters
        ----------
        data : Data
            Data to check.

        Returns
        -------
        converted_data : npt.NDArray
            Converted data of shape (n_samples, n_labels)

        Raises
        ------
        TypeError
            If data is not a Data type.

        """
        convertable = True
        if isinstance(data, ndarray):
            # Data does not need to be converted
            return data
        elif isinstance(data, pd.DataFrame):
            for col in data.columns:
                if not pd.api.types.is_numeric_dtype(col):
                    # If one of the columns is not numeric return False
                    convertable = False
            if convertable:
                # Convert data to a ndarray
                return data.to_numpy()
        else:
            raise TypeError(f"data should be a Data type, but got {type(data)}")

    def fit(
        self, X_train: Data, y_train: Labels, weights: npt.NDArray | None = None
    ) -> None:
        """
        Fits the predictor to the training data.

        Parameters
        ----------
        X_train : Data
            Training data of shape (n_samples, n_labels).
        y_train : Labels
            Prediction labels of shape (n_samples,).
        weights : npt.NDArray | None, default: None
            Weights to address class imbalance.

        Returns
        -------
        None
            Nothing is returned.

        """
        train_data = self._convert_data(X_train)
        self.model.fit(train_data, y_train.squeeze(), sample_weight=weights)

    def predict_proba(self, X: Data) -> FProbas:
        """
        Predicts probabilities from input data.

        Parameters
        ----------
        X : Data
            Training data of shape (n_samples, n_labels).

        Returns
        -------
        probabilities : FProbas
            Predicted probabilities of shape (n_samples, 2).

        """
        return self.model.predict_proba(self._convert_data(X))

    @abstractmethod
    def predict(self, X: Data) -> Labels:
        """
        Predicts labels from input data.

        Parameters
        ----------
        X : Data
            Training data of shape (n_samples, n_labels).

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

    def predict(self, X: Data) -> Labels:
        """
        Predicts labels from input data.

        Parameters
        ----------
        X : Data
            Training data of shape (n_samples, n_labels).

        Returns
        -------
        labels : Labels
            Predicted labels of shape (n_samples,).

        """
        return self.model.predict(self._convert_data(X))
