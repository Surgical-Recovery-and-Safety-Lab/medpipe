"""
MedpipePipeline class.

This class creates a MedpipePipeline to prepare data, fit a predictor and
a calibrator.

"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Type, overload

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, available_if

import medpipe.data.weighting as weight
from medpipe._types import Data, FullProba, Labels, PosProba, PredData
from medpipe.data.sampler import data_sampler
from medpipe.data.utils import (
    convert_data,
    extract_labels,
    get_data_from_idx,
    get_validation_idx,
)
from medpipe.metrics.core import print_metrics
from medpipe.models.calibrators import create_calibrator
from medpipe.models.core import get_positive_proba, test_model
from medpipe.models.predictors import create_predictor
from medpipe.utils.config import MedpipeConfig
from medpipe.utils.io import read_toml_configuration
from medpipe.utils.logger import print_message

SCRIPT_NAME = "pipeline/pipeline"
if TYPE_CHECKING:
    import logging

    import numpy.typing as npt


class MedpipePipeline(Pipeline):
    """
     Class that creates a Pipeline.

     Attributes
     ----------
     version : str
         Version number.
     outcomes : list[str]
         List of labels to predict.
     n_labels : int
         Number of labels to predict.
     predictor_algo : str
         Algorithm used by the predictor.
     calibrator_method : Literal["isotonic", "logistic"]
         Method used by the calibrator.
    preprocessor : Preprocessor | None
         Data preprocessor object.
     predictor : dict[str, Predictor]
         Dictionary of Predictor instances for each label.
     calibrator : dict[str, Calibrator] | None
         Dictionary of Calibrator instances for each label.
     predictor_train_outputs : dict[str, dict[int, PredProba]]
         Dictionary of predicted probabilities for each predictor
         The dictionary keys are the labels and the values are
         the predicted probabilities of the predictor for that fold.
     calibrator_train_outputs : dict[str, dict[int, PredProba]] | None
         Dictionary of predicted probabilities for each calibrator
         The dictionary keys are the labels and the values are
         the predicted probabilities of the calibrator for that fold.
     logger : logging.Logger | None, default: None
         Logger object to log prints. If None print to terminal.

     Methods
     -------
     fit_preprocessor(X)
         Fits the preprocessor operations based on input data.
     transform(X)
         Transforms input data based on preprocessor fitted operations.
     fit_model(X, y, model, **kwargs)
         Fits the predictor or calibrator model on the provided dataset.
     test_model(X, y, model, outcomes, key=None)
         Tests the predictor or calibrator model on the provided dataset.
     run(X)
         Run pipeline with input data.
     predict_proba(X)
         Predicts probabilities from predictor or calibrator based on input data.
     predict(X)
         Predicts labels from predictor or calibrator based on input data.
    """

    @overload
    def __init__(self, config: str | Path, logger: logging.Logger | None) -> None: ...

    @overload
    def __init__(
        self, config: MedpipeConfig, logger: logging.Logger | None
    ) -> None: ...

    def __init__(
        self,
        config: str | Path | MedpipeConfig,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialise a MedpipePipeline class instance.

        Parameters
        ----------
        config : str | Path | MedpipeConfig
            Path to the top-level configuration file to load or
            loaded MedpipeConfig.
        logger : logging.Logger | None, default: None
            Logger object to log prints. If None print to terminal.

        Returns
        -------
        None
            Nothing is returned.

        """
        if isinstance(config, (str, Path)):
            self.medpipe_config = read_toml_configuration(config)
        elif isinstance(config, MedpipeConfig):
            self.medpipe_config = config
        else:
            raise ValueError(
                "A configuration file or a MedpipeConfig must be specified."
            )

        self.logger = logger

        top_level = (
            self.medpipe_config.top_level.model_dump()
        )  # Get top-level parameters

        # Extract some top-level parameters
        self.version = top_level["meta"]["version"]
        self.predictor_algo = top_level["model"]["algorithm"]
        self.calibrator_method = top_level["calibration"]["method"]

        # Get outcomes from configuration
        self.outcomes = self.medpipe_config.data.outcomes
        self.n_outcomes = len(self.outcomes)

        print_message(
            f"Setting up MedpipePipeline {self.version}",
            self.logger,
            SCRIPT_NAME,
        )

        # Setup preprocessor
        preprocessing_steps = self._set_preprocessing_steps()
        self.preprocessor = (  # Set preprocessor to Pipeline or None
            Pipeline(steps=preprocessing_steps) if preprocessing_steps else None
        )

        # Cache hyperparameters for the loop
        pred_params = (
            self.medpipe_config.hyperparameters.hyperparameters.predictor.model_dump()
        )
        if self.medpipe_config.hyperparameters.hyperparameters.calibrator:
            cal_params = (
                self.medpipe_config.hyperparameters.hyperparameters.calibrator.model_dump()
            )
        else:
            cal_params = {}

        # Efficiently initialize model dictionaries
        # Using a single loop to populate all per-label attributes
        self.predictor = {}
        self.calibrator = {}
        self.predictor_train_outputs = {}  # Store training outputs
        self.calibrator_train_outputs = {}  # Store training outputs

        for outcome in self.outcomes:
            # Initialize Predictor
            self.predictor[outcome] = create_predictor(
                self.predictor_algo,
                hyperparameters=pred_params,
                logger=self.logger,
            )
            self.predictor_train_outputs[outcome] = {}

            # Initialize Calibrator (if applicable)
            if self.calibrator_method:
                self.calibrator[outcome] = create_calibrator(
                    self.calibrator_method,
                    hyperparameters=cal_params,
                    logger=self.logger,
                )
                self.calibrator_train_outputs[outcome] = {}

    def _set_preprocessing_steps(self) -> ColumnTransformer | None:
        """
        Sets data preprocessing steps if the have been specified.

        Returns
        -------
        transformers : ColumnTransformer or None
            If preprocessing operations are provided return the ColumnTransformer
            otherwise return None.

        """
        preprocessing_dict = self.medpipe_config.workflow.preprocessing
        if preprocessing_dict and preprocessing_dict.preprocess:
            # Preprocessing config passed with preprocess flag True
            transformers = []  # Empty list to be passed to the ColumnTransformer
            if preprocessing_dict.operations:
                for i, operation in enumerate(preprocessing_dict.operations):
                    op_type = self._check_operation(operation.name)
                    op_extras = (
                        {} if not operation.model_extra else operation.model_extra
                    )
                    transformers.append(
                        (
                            f"op_{i+1}",
                            op_type(**op_extras),
                            operation.columns,
                        )
                    )
            return ColumnTransformer(transformers=transformers, remainder="passthrough")

        return None

    def _check_operation(self, op: str) -> Type:
        """
        Internal function that checks if the operation is valid.

        Currently checks the sklearn.preprocessing and sklearn.impute modules.

        Parameters
        ----------
        op : str
            Operation name.

        Returns
        -------
        operation : Type
            Return the attribute if it exists.

        Raises
        ------
        ValueError
            If the operation is invalid.

        """
        if hasattr(sklearn.preprocessing, op):
            return getattr(sklearn.preprocessing, op)

        if hasattr(sklearn.impute, op):
            return getattr(sklearn.impute, op)

        raise ValueError(
            f"{op} is not found in sklearn.preprocessing or sklearn.impute, "
            "please check that the operation matches"
        )

    def _has_preprocessor(self) -> bool:
        """
        Internal function that checks for the presence of
        a preprocessing attribute.

        Returns
        -------
        has_preprocessor : bool
            True if the pipeline has a preprocessor.

        """
        if self.medpipe_config.workflow.preprocessing:
            preprocess = self.medpipe_config.workflow.preprocessing.preprocess
            return preprocess if preprocess else False
        return False

    @available_if(_has_preprocessor)
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms input data based on preprocessor fitted operations.

        Parameters
        ----------
        X : pd.Dataframe
            Data to clean of shape (n_samples, n_features).

        Returns
        -------
        data : pd.Dataframe
             Transformed data of shape (n_samples, n_features).
             Returns X if no preprocessor exists.

        Warns
        -----
        UserWarning
            If no preprocessor object exists.

        """
        if self.preprocessor:
            return self.preprocessor.transform(X)
        else:
            warn("No preprocessor object created so data not transformed")
            return X

    @available_if(_has_preprocessor)
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fits the preprocessor operations and transforms the input data.


        Parameters
        ----------
        X : pd.Dataframe
            Data to clean of shape (n_samples, n_features).

        Returns
        -------
        data : pd.Dataframe
             Transformed data of shape (n_samples, n_features).
             Returns X if no preprocessor exists.

        Warns
        -----
        UserWarning
            If no preprocessor object exists.

        """
        if self.preprocessor:
            return self.preprocessor.fit_transform(X)
        else:
            warn("No preprocessor object created so data not transformed")
            return X

    def get_test_data(
        self, X: pd.DataFrame, test_group_vals: list[Any] | None = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns train and test data based on input data.

        Parameters
        ----------
        X : pd.DataFrame
            Data of shape (n_samples, n_features) to split.
        test_group_vals : list[Any] | None, default: None
            Group values that should be in the test set.

        Returns
        -------
        X_train : pd.DataFrame
            Train set of shape (n_samples, n_features).
        X_test : pd.DataFrame
            Test set of shape (n_samples, n_features).

        """
        # Access config with defaults
        split_vars = self.preprocessor_config.get("split_variables", {})
        group_col = split_vars.get("group_name")
        test_size = split_vars.get("test_size", 0.1)

        # Determine indices
        indices = np.arange(len(X))

        if group_col and test_group_vals:
            # Group-based split using the optimized utility
            groups = X[group_col].to_numpy()
            train_idx, test_idx = get_validation_idx(
                indices, groups=groups, group_vals=test_group_vals
            )
        else:
            # Standard random split
            train_idx, test_idx = get_validation_idx(indices, val_size=test_size)

        # Extract and return
        return X.iloc[train_idx], X.iloc[test_idx]

    def fit_model(
        self,
        X: Data,
        y: Labels,
        model: str,
        label: str,
        weights: npt.NDArray = np.array([]),
    ) -> None:
        """
        Fits the predictor or calibrator model on the provided dataset.

        Parameters
        ----------
        X : Data
            Training data of shape (n_samples, n_features) or (n_samples,).
        y : Labels
            Prediction labels of shape (n_samples, self.n_labels).
        model : {"predictor", "calibrator"}
            Model to fit.
        label : str
            Label associated with the model to use.
        weights : npt.NDArray, default: np.array([])
            Weights for addressing class imbalance.

        Returns
        -------
        None
            Nothing is returned.

        Raises
        ------
        ValueError
            If model is not "predictor" or "calibrator".

        """
        match model:
            case "predictor":
                self.predictor[label].fit(X, y, weights)
            case "calibrator":
                self.calibrator[label].fit(X, y)
            case _:
                raise ValueError(
                    f"Model should be predictor or calibrator, but got {model}"
                )

    def test_model(self, X: Data, y: Labels, model: str, label: str) -> None:
        """
        Tests the predictor or calibrator model on the provided dataset.

        Parameters
        ----------
        X : Data
            Training data of shape (n_samples, n_features).
        y : Labels
            Prediction labels of shape (n_samples,).
        model : {"predictor", "calibrator"}
            Model to test.
        label : str
            Label associated with the model to use.

        Returns
        -------
        None
            Nothing is returned.

        Raises
        ------
        ValueError
            If model is not "predictor" or "calibrator".

        """
        # Map model to message
        messages = {
            "predictor": "Uncalibrated metrics",
            "calibrator": "Calibrated metrics",
        }

        if model not in messages:
            raise ValueError(
                f"Model should be predictor or calibrator, but got {model}"
            )

        # Get the positive class probabilities (PosProba)
        probabilities = self.predict_proba(X, outcomes=label, model_type=model)

        # Get positive probabilities and predictions by thresholding
        prob_arr = get_positive_proba(probabilities).squeeze()
        predictions = (prob_arr >= 0.5).astype(int)

        # Compute test metrics
        metric_dict = test_model(
            y,
            predictions,
            probabilities,
        )

        print_message(messages[model], self.logger, SCRIPT_NAME)
        print_metrics(metric_dict, [label], self.logger)

    def run(self, X: pd.DataFrame) -> None:
        """
        Run pipeline with input data.

        Parameters
        ----------
        X : pd.DataFrame
            Training data of shape (n_samples, n_features).

        Returns
        -------
        None
            Nothing is returned.

        """
        # Preprocessing Data
        data = (
            self.transform(X) if self.preprocessor.operations else self.fit_transform(X)
        )

        # Safe Config Access
        cv_cfg = self.preprocessor_config.get("cv_variables", {})
        split_cfg = self.preprocessor_config.get("split_variables", {})

        cv_group_name = cv_cfg.get("group_name")
        cv_drop = cv_cfg.get("drop", False)
        split_group_name = split_cfg.get("group_name")
        split_drop = split_cfg.get("drop", False)

        X_full, y_full = extract_labels(data, self.outcomes)
        cv_groups_full = data[cv_group_name].to_numpy() if cv_group_name else None

        # Setup Calibration Set
        X_cal_set, y_cal_set = np.array([]), np.array([])
        if self.calibrator_type:
            split_groups = (
                data[split_group_name].to_numpy() if split_group_name else np.array([])
            )
            t_idx, c_idx = get_validation_idx(
                np.arange(len(y_full)), groups=split_groups
            )

            X_cal_set = X_full.iloc[c_idx]
            y_cal_set = y_full[c_idx]

            # If calibration data needs the split group dropped
            if split_drop and split_group_name:
                X_cal_set = X_cal_set.drop(columns=[split_group_name])

            if cv_drop and cv_group_name and split_group_name != cv_group_name:
                X_cal_set = X_cal_set.drop(columns=[cv_group_name])

            # Restrict training/CV to the remaining data
            X_work, y_work = X_full.iloc[t_idx], y_full[t_idx]
            cv_groups = cv_groups_full[t_idx] if cv_groups_full is not None else None
        else:
            X_work, y_work, cv_groups = X_full, y_full, cv_groups_full

        # Drop splitting column if it's not the same as the CV column
        if split_drop and split_group_name and split_group_name != cv_group_name:
            X_work = X_work.drop(columns=[split_group_name], errors="ignore")

        # Cross-Validation Loop
        if cv_cfg["n_splits"] > 0:
            kfold_it = train_test_it(**cv_cfg)
            n_folds = kfold_it.get_n_splits(X_work, y_work[:, 0], groups=cv_groups)

            for i, (train_idx, test_idx) in enumerate(
                kfold_it.split(X_work, y_work[:, 0], groups=cv_groups)
            ):
                # Determine fold key (Group ID or index)
                fold_key = cv_groups[test_idx[0]] if cv_groups is not None else i
                fold_groups = (
                    cv_groups[train_idx] if cv_groups is not None else np.array([])
                )

                # Prepare X_fold (keeping or dropping CV group based on config)
                X_fold = (
                    X_work.drop(columns=[cv_group_name])
                    if (cv_groups is not None and cv_drop)
                    else X_work
                )
                X_fold = convert_data(X_fold)

                X_train_f = get_data_from_idx(X_fold, train_idx)
                X_test_f = get_data_from_idx(X_fold, test_idx)
                y_train_f, y_test_f = y_work[train_idx], y_work[test_idx]

                for j, label in enumerate(self.outcomes):
                    # Sample and Weight for specific label
                    X_tr_label, y_tr_label, _ = self._sample_data(
                        X_train_f, y_train_f[:, j : j + 1], fold_groups
                    )
                    weights = self._weight_data(y_tr_label)

                    print_message(
                        f"Label: {label} | {fold_key} | Fold {i+1}/{n_folds}",
                        self.logger,
                        SCRIPT_NAME,
                    )

                    # Execute training and evaluation
                    self._fit_and_evaluate(
                        X_tr_label,
                        y_tr_label,
                        X_test_f,
                        y_test_f[:, j],
                        label,
                        fold_key,
                        X_cal=X_cal_set,
                        y_cal=y_cal_set[:, j] if self.calibrator_type else np.array([]),
                        weights=weights,
                    )

        # Final Model Training
        print_message(
            "Final training on all non-calibration data", self.logger, SCRIPT_NAME
        )
        X_final = (
            X_work.drop(columns=[cv_group_name])
            if (cv_groups is not None and cv_drop)
            else X_work
        )

        for k, label in enumerate(self.outcomes):
            X_tr_final, y_tr_final, _ = self._sample_data(
                X_final, y_work[:, k : k + 1], cv_groups
            )
            weights = self._weight_data(y_tr_final)
            print_message(
                f"Train set size: {len(X_tr_final):,}", self.logger, SCRIPT_NAME
            )
            print_message(
                f"Calibration set size: {len(X_cal_set):,}", self.logger, SCRIPT_NAME
            )
            self.fit_model(X_tr_final, y_tr_final, "predictor", label, weights)

            if self.calibrator_type:
                self.fit_model(
                    self._get_calibrator_data(X_cal_set, label),
                    y_cal_set[:, k],
                    "calibrator",
                    label,
                )

    def _fit_and_evaluate(
        self,
        X_train: PredData,
        y_train: Labels,
        X_test: Data,
        y_test: Labels,
        label: str,
        fold_key: Any,
        X_cal: PosProba = np.array([]),
        y_cal: Labels = np.array([]),
        weights: npt.NDArray = np.array([]),
    ) -> None:
        """
        Helper function to handle the Train -> Test -> Store lifecycle for a single label.

        Parameters
        ----------
        X_train : PredData
            Train data of shape (n_samples, n_features) for the predictor.
        y_train : Labels
            Train labels of shape (n_samples,) for the predictor.
        X_test : Data
            Test data of shape (n_test_samples, n_features).
        y_test : Labels
            Test labels of shape (n_test_samples,).
        label: str
            Label associated with the model to train.
        fold_key : Any
            Key representing the current fold.
        X_cal : PosProba, default: np.array([])
            Calibration data of shape (n_samples,) for the calibrator.
        y_cal : Labels, default: np.array([])
            Calibration labels of shape (n_samples,) for the calibrator.
        weights : npt.NDArray, default: np.array([])
            Weights to address class imbalance.

        """
        print_message(f"Train set size: {len(X_train):,}", self.logger, SCRIPT_NAME)
        print_message(f"Calibration set size: {len(X_cal):,}", self.logger, SCRIPT_NAME)
        print_message(f"Test set size: {len(X_test):,}", self.logger, SCRIPT_NAME)
        # Fit predictor on train set
        self.fit_model(X_train, y_train, "predictor", label, weights)

        # Fit calibrator on validation set
        if self.calibrator_type:
            self.fit_model(
                self._get_calibrator_data(X_cal, label),
                y_cal,
                "calibrator",
                label,
            )

        # Evaluation
        modes = ["predictor"]
        if self.calibrator_type:
            modes.append("calibrator")

        for mode in modes:
            # Use optimized test_model (performs a single inference pass)
            self.test_model(X_test, y_test, mode, label)

            probas = get_positive_proba(
                self.predict_proba(X_test, label, model_type=mode)
            )

            # Store results in the dictionaries initialized in __init__
            storage = (
                self.calibrator_train_outputs
                if mode == "calibrator"
                else self.predictor_train_outputs
            )
            storage[label][fold_key] = probas

        # Reset model state (quietly) to clear weights/fit data for the next fold
        self.predictor[label]._set_model(quiet=True)
        if self.calibrator_type:
            self.calibrator[label]._set_model(quiet=True)

    def predict_proba(
        self,
        X: Data,
        outcomes: str | list[str] = "all",
        model_type: str | list[str] = "predictor",
    ) -> FullProba:
        """
        Predicts probabilities from predictor or calibrator based on input data.

        Parameters
        ----------
        X : Data
            Data to use of shape (n_samples, n_features) or (n_samples,).
        outcomes : str | list[str], default: "all"
            Label or list of labels associated with the model to use.
            If all, all models are used.
        model_type : str | list[str], default: "predictor"
            Model type or list of model types to use.

        Returns
        -------
        probabilities : FullProba
            Full predicted probabilities of shape (n_samples, 2).

        Raises
        ------
        ValueError
            If the label list and the model type list are not the same length.

        """
        X_clean = self._prepare_inference_data(X)
        target_labels = (
            self.outcomes
            if outcomes == "all"
            else ([outcomes] if isinstance(outcomes, str) else outcomes)
        )
        model_types = [model_type] if isinstance(model_type, str) else model_type

        # Check same length
        if len(model_types) != len(target_labels):
            raise ValueError("Label list and model types should be the same length")

        results = [
            self._inference_wrapper(
                X_clean, target_labels[i], "predict_proba", model_types[i]
            )
            for i in range(len(target_labels))
        ]

        # If single label, return just the array; if multiple, return the stacked array
        return results[0] if len(target_labels) == 1 else np.asarray(results)

    def predict(
        self,
        X: Data,
        outcomes: str | list[str] = "all",
        model_type: str | list[str] = "predictor",
    ) -> Labels:
        """
        Predicts labels from predictor or calibrator based on input data.

        Parameters
        ----------
        X : Data
            Data to use of shape (n_samples, n_features) or (n_samples,).
        outcomes : str | list[str], default: "all"
            Label or list of labels associated with the model to use.
            If all, all models are used.
        model_type : str | list[str], default: "predictor"
            Model type or list of model types to use.

        Returns
        -------
        labels : Labels
            Predicted labels of shape (n_samples,).

        Raises
        ------
        ValueError
            If the label list and the model type list are not the same length.

        """
        probabilities = self.predict_proba(X, outcomes, model_type)
        pos_proba = get_positive_proba(probabilities)

        return np.array(pos_proba > 0.5, dtype=np.int32)

    def _inference_wrapper(
        self, X: Data, label: str, prediction_type: str, model_category: str
    ) -> Labels | FullProba:
        """
        Unified wrapper for predictor and calibrator inference.

        Parameters
        ----------
        X : PredData
            Data for predictor classes on of shape (n_samples, n_features).
        label : str
            Label associated with the model to use.
        prediction_type : {"predict", "predict_proba"}
            Prediction function to use.

        Returns
        -------
        estimates : Labels | FullProba
            Labels or probabilities estimated from model based on prediction_type.

        Raises
        ------
        AttributeError
            If the model does not have a predict or predict_proba method

        """
        # Select the model dictionary (predictor or calibrator)
        model_dict = getattr(self, model_category)
        model = model_dict[label]

        # Get data for calibrators
        if model_category == "calibrator":
            X = self._get_calibrator_data(X, label)

        # Get "predict" or "predict_proba" method
        try:
            inference_method = getattr(model, prediction_type)
        except AttributeError:
            raise ValueError(
                f"Model category '{model_category}' does not support '{prediction_type}'"
            )

        return inference_method(X)

    def _prepare_inference_data(self, X: Data) -> Data:
        """
        Ensure group columns are removed before passing to models.

        Parameters
        ----------
        X : Data
            Input data of shape (n_samples, n_features + 1) or (n_samples,).

        Returns
        -------
        X_clean : Data
            Cleaned data of shape (n_samples, n_feautres) or (n_samples,).

        """
        group_col = self.preprocessor_config.get("split_variables", {}).get(
            "group_name"
        )
        cv_cfg = self.preprocessor_config.get("cv_variables", {})
        cv_drop = cv_cfg.get("drop", False)

        if isinstance(X, pd.DataFrame) and group_col in X.columns and cv_drop:
            return X.drop(columns=[group_col])
        return X

    def _sample_data(
        self, X: PredData, y: Labels, groups: npt.NDArray | None
    ) -> tuple[PredData, Labels, npt.NDArray]:
        """
        Samples the data based on configuration.

        Parameters
        ----------
        X : PredData
            Data to sample of shape (n_samples, n_features).
        y : Labels
            Labels to sample of shape (n_samples,).
        groups : npt.NDArray | None
            Groups of the examples of shape (n_samples,) or None.

        Returns
        -------
        X_sampled : pd.DataFrame
            Sampled data of shape (n_sampled_samples, n_features).
        y_sampled : Labels
            Sampled labels of shape (n_sampled_samples,).
        groups_sampled : pd.Series
            Groups of the examples of shape (n_sampled_samples,) or empty

        """
        sampler_fn = self.predictor_config["sampler"]["sampler_fn"]
        if groups is None:
            groups = np.array([])

        if sampler_fn:
            return data_sampler(X, y, groups=groups, **self.predictor_config["sampler"])

        return X, y, groups

    def _weight_data(self, y: Labels) -> npt.NDArray:
        """
        Gets the weights for the data based on configuration.

        If no weighting function is provided in the configuration, the function
        returns the sample weights equal to 1 by default.

        Parameters
        ----------
        y : Labels
            Labels of shape (n_samples, self.n_labels) needed for creation of weights.

        Returns
        -------
        weights : npt.NDArray
            Sample or class weights based on labels.
            Sample weights are of shape (n_samples,).
            Class weights are of shape (1).

        """
        weighting_fn = self.predictor_config["weighting"]["weighting_fn"]

        if weighting_fn:
            return getattr(weight, weighting_fn)(y)

        return np.ones(y.shape[0])

    def _get_calibrator_data(self, X: PredData, label: str) -> PosProba:
        """
        Get the calibrator data based on the calibrator type.

        Parameters
        ----------
        X : PredData
            Data of shape (n_samples, n_features) to transform for calibrator.
        label : str
            Label associated with the model to use.

        Returns
        -------
        calibrator_data : npt.NDArray
            Data of shape (n_samples,) or (n_samples, 2) for calibrator model
            based on its type.

        """
        calibrator_data = get_positive_proba(
            self.predict_proba(X, label, model_type="predictor")
        )

        # Flatten to avoid (n_samples, 1) vs (n_samples,) inconsistencies
        return calibrator_data.ravel()
