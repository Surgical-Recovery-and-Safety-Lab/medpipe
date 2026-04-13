"""
Preprocessor class.

This class creates a Preprocessor to prepare data.

"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Mapping

from medpipe._types import PreprocessOp, PreprocessOpConfig
from medpipe.utils.logger import print_message

from .preprocessing import (
    bin_score,
    convert_object_to_categorical,
    downcast_dtypes,
    fit_preprocess_operations,
)

if TYPE_CHECKING:
    import logging

    import pandas as pd

SCRIPT_NAME = "data/preprocessor"


class Preprocessor:
    """
    Class that creates a Preprocessor.

    Attributes
    ----------
    preprocess : bool
        Flag to preprocess data or not.
    transform_seq : PreprocessOpConfig
        Transformation sequence for the data.
    operations : Mapping[str, PreprocessorOp | str]
        Preprocessing operations that are fitted.
    logger : logging.Logger | None, default: None
        Logger object to log prints. If None print to terminal.

    Methods
    -------
    __init__(preprocessor_config, logger)
        Init method.
    _clean_data(X)
        Cleans data in preparation for transformation.
    fit_transform(X)
        Fits the operations and transforms the input data.
    fit(X)
        Fits the operations based on input data.
    transform(X)
        Transforms input data based on fitted operations.
    """

    def __init__(
        self,
        preprocessor_config: PreprocessOpConfig,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialise a Preprocessor class instance.

        Parameters
        ----------
        preprocessor_config : PreprocessorOpConfig
            Configuration parameters for the preprocessor object.
        logger : logging.Logger | None, default: None
            Logger object to log prints. If None print to terminal.

        Returns
        -------
        None
            Nothing is returned.

        """
        self.preprocess = preprocessor_config.pop("preprocess")
        self.transform_seq = preprocessor_config
        self.operations: Mapping[str, PreprocessOp | str] = (
            dict()
        )  # Empty dict to contain operations
        self.logger = logger

    def _clean_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans data before transformation.

        Removes rows with Nan values and converts objects to
        categoricals.

        Parameters
        ----------
        X : pd.Dataframe
            Data to clean of shape (n_samples, n_features).

        Returns
        -------
        data : pd.Dataframe
             Cleaned data of shape (n_samples, n_features).

        """
        # Convert objects to categorical (not saved so needs to be here)
        data = convert_object_to_categorical(X)

        # Remove NaN values
        nb_nan_rows = data.isna().any(axis=1).sum()
        data = data.dropna()

        print_message(
            f"Dropped {nb_nan_rows} rows with NaN values", self.logger, SCRIPT_NAME
        )

        return data

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fits the operations and transforms the input data.

        Parameters
        ----------
        X : pd.Dataframe
            Data of shape (n_samples, n_features) to clean.

        Returns
        -------
        data : pd.Dataframe
             Transformed data of shape (n_samples, n_features).

        """
        data = self._clean_data(deepcopy(X))  # Clean data before transformation
        self.fit(data)  # Fit operations
        return self.transform(data)  # Transform data

    def fit(self, X: pd.DataFrame) -> None:
        """
        Fits the operations based on input data.

        Parameters
        ----------
        X : pd.Dataframe
            Data to clean of shape (n_samples, n_features).

        Returns
        -------
        None
            Nothings is returned.

        """
        data = self._clean_data(X)  # Clean data before transformation

        if self.preprocess:
            # If the preprocess flag is true
            print_message("Fitting preprocessing operations", self.logger, SCRIPT_NAME)
            self.operations = fit_preprocess_operations(data, self.transform_seq)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms input data based on fitted operations.

        Parameters
        ----------
        X : pd.Dataframe
            Data to clean of shape (n_samples, n_features).

        Returns
        -------
        data : pd.Dataframe
             Transformed data of shape (n_samples, n_features).

        """
        data = self._clean_data(deepcopy(X))  # Clean data before transformation

        if self.preprocess:
            # If the preprocess flag is true
            print_message("Preprocessing data", self.logger, SCRIPT_NAME)
            for operation in self.operations:
                features = self.transform_seq[operation]["feature_list"]

                if isinstance(self.operations[operation], str):
                    transformed_data = bin_score(data[features].to_numpy())
                else:
                    transformed_data = self.operations[operation].transform(
                        data[features]
                    )
                data[features] = transformed_data

        data = downcast_dtypes(data)  # Downcast for speed
        return data
