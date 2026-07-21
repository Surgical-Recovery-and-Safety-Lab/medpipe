"""
MedpipePipeline class.

This class creates a MedpipePipeline to prepare data, fit a predictor and
a recalibrator.

"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Type, overload
from warnings import warn

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.sparse import csr_array, csr_matrix
from sklearn.base import BaseEstimator, ClassifierMixin, is_classifier
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from medpipe._types import Labels, TransformedData
from medpipe.data.utils import convert_dtypes, extract_labels, split_data
from medpipe.metrics.core import (
    METRIC_MAPPING,
    build_scorers,
    compute_metrics,
    print_metrics,
)
from medpipe.metrics.plots import (
    plot_probability_distribution,
    plot_reliability_diagram,
    plot_ROC_curve,
    plot_strata_heatmap,
)
from medpipe.models.core import create_estimator
from medpipe.utils.config import MedpipeConfig
from medpipe.utils.io import load_data, read_toml_configuration
from medpipe.utils.logger import print_message

SCRIPT_NAME = "pipeline/pipeline"
if TYPE_CHECKING:
    import logging

    import numpy.typing as npt


class MedpipePipeline(BaseEstimator, ClassifierMixin):
    """
    Class that creates a MedpipePipeline.

    Attributes
    ----------
    version : str
        Version number.
    outcomes : list[str]
        List of labels to predict.
    n_outcomes : int
        Number of outcomes to predict.
    metrics : list[str]
        List of metrics to evaluate models with.
    predictor_algo : str
        Algorithm used by the predictor.
    recalibrator_method : Literal["isotonic", "logistic"]
        Method used by the recalibrator.
    preprocessor : ColumnTransformer | None
        Column transformer to preprocess data.
    predictor : dict[str, Type]
        Dictionary of predictor instances for each label.
    recalibrator : dict[str, Type] | None
        Dictionary of recalibrator instances for each label.
    predictor_train_outputs : dict[str, dict[int, PredProba]]
        Dictionary of predicted probabilities for each predictor
        The dictionary keys are the labels and the values are
        the predicted probabilities of the predictor for that fold.
    recalibrator_train_outputs : dict[str, dict[int, PredProba]] | None
        Dictionary of predicted probabilities for each recalibrator
        The dictionary keys are the labels and the values are
        the predicted probabilities of the recalibrator for that fold.
    folds : dict[str | int, int]
        Dictionary containing the fold names and the fold index.
    logger : logging.Logger | None, default: None
        Logger object to log prints. If None print to terminal.

    Methods
    -------
    fit_transfrom(X)
        Fits the preprocessor operations and transforms the input data.
    transform(X)
        Transforms input data based on preprocessor fitted operations.
    fit(X, y, X_recal, y_recal)
        Fit the MedpipePipeline estimator and recalibrator.
    get_data_sets(data)
        Splits data into train, test, and recalibration sets.
    run(data)
        Run pipeline with input data.
    save()
        Saves the pipeline as {project_name}_{version}.joblib.
    test_models(X, y)
        Tests the predictor or recalibrator model on the provided dataset.
    predict_proba(X)
        Predicts probabilities from predictor or recalibrator based on input data.
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
        self.config = config

        if isinstance(self.config, (str, Path)):
            self.medpipe_config = read_toml_configuration(self.config)
        elif isinstance(self.config, MedpipeConfig):
            self.medpipe_config = self.config
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
        if top_level["recalibration"]:
            self.recalibrator_method = top_level["recalibration"]["method"]
        else:
            self.recalibrator_method = None

        # Get outcomes and metrics from configuration
        self.outcomes = self.medpipe_config.data.outcomes
        self.n_outcomes = len(self.outcomes)
        self.metrics = self.medpipe_config.workflow.evaluation.metrics.metrics

        print_message(
            f"Setting up MedpipePipeline {self.version}",
            self.logger,
            SCRIPT_NAME,
        )

        # Setup preprocessor
        self.preprocessor = (
            self._set_preprocessing_steps() if self._has_preprocessor() else None
        )

        # Create empty dictionary
        self.predictor = {}
        self.recalibrator = {}
        self.predictor_train_outputs = {}  # Store training outputs
        self.folds = {}  # Store fold name and fold index

        for outcome in self.outcomes:
            # Initialise predictors
            self.predictor[outcome] = create_estimator(
                self.predictor_algo,
                **self.medpipe_config.hyperparameters.hyperparameters.predictor.model_dump(),
            )
            self.predictor_train_outputs[outcome] = {}

            # Initialise recalibrators (if applicable)
            if (
                self.recalibrator_method
                and self.medpipe_config.hyperparameters.hyperparameters.recalibrator
            ):
                self.recalibrator[outcome] = create_estimator(
                    self.recalibrator_method,
                    **self.medpipe_config.hyperparameters.hyperparameters.recalibrator.model_dump(),
                )

            # Initialise folds
            self.folds[outcome] = {}

    def _set_preprocessing_steps(self) -> Pipeline | None:
        """
        Sets data preprocessing steps if the have been specified.

        Returns
        -------
        pipe : Pipeline or None
            If preprocessing operations are provided return the Pipeline
            of ColumnTransformers, otherwise return None.

        """
        preprocessing_dict = self.medpipe_config.workflow.preprocessing
        if preprocessing_dict and preprocessing_dict.preprocess:
            # Preprocessing config passed with preprocess flag True
            steps = []  # Empty list to be passed to the ColumnTransformer
            if preprocessing_dict.operations:
                ct_columns_dict = {
                    pred: pred for pred in self.medpipe_config.data.predictors
                }
                for i, operation in enumerate(preprocessing_dict.operations):
                    op_type = self._check_operation(operation.name)
                    op_extras = (
                        {} if not operation.model_extra else operation.model_extra
                    )

                    ct_columns = [  # Columns for the ColumnTransformer
                        ct_columns_dict[column] for column in operation.columns
                    ]
                    ct = ColumnTransformer(  # ColumnTransformer for operation
                        [(f"op_{i+1}", op_type(**op_extras), ct_columns)],
                        remainder="passthrough",
                    )
                    ct.set_output(transform="pandas")  # Return pd.DataFrame

                    if i == len(preprocessing_dict.operations) - 1:
                        ct.set_output(transform="default")  # Last ct return a np.array

                    steps.append((f"transformer_{i+1}", ct))  # Steps for the Pipeline

                    # Update the ct_columns_dict
                    ct_columns_dict = {
                        pred: (
                            f"remainder__{column}"
                            if column not in ct_columns
                            else f"op_{i+1}__{column}"
                        )
                        for (pred, column) in ct_columns_dict.items()
                    }

            return Pipeline(steps=steps)

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
        the preprocessing attributes.

        Returns
        -------
        has_preprocessor : bool
            True if the pipeline has a preprocessor.

        """
        if self.medpipe_config.workflow.preprocessing:
            preprocess = self.medpipe_config.workflow.preprocessing.preprocess
            return preprocess if preprocess else False
        return False

    def _drop_group_columns(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame | None = None,
        X_recal: pd.DataFrame | None = None,
    ) -> tuple[
        pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None, npt.NDArray | None
    ]:
        """
        Drops the different group columns if needed.

        Parameters
        ----------
        X_train : pd.DataFrame
            Train data.
        X_recal, X_test : pd.DataFrame | None, default: None
            Recalibration and test data if needed, None otherwise.

        Returns
        -------
        X_train_dropped : pd.DataFrame
            Train data without the group columns.
        X_recal_dropped, X_test_dropped : pd.DataFrame | None
            Recalibration and test data without the group columns or None.
        groups : npt.NDArray | None
            Train set group column for cross-validation, None if not
            specified.

        """

        @overload
        def _drop_column(X: pd.DataFrame, col: str | None) -> pd.DataFrame: ...

        @overload
        def _drop_column(X: None, col: str | None) -> None: ...

        def _drop_column(
            X: pd.DataFrame | None, col: str | None
        ) -> pd.DataFrame | None:
            """
            Drop the column in data if they exist.

            Parameters
            ----------
            X : pd.DataFrame | None
                Data to drop the column in or None.
            col : str | None
                Name of the column to drop or None

            Returns
            -------
            X_dropped : pd.DataFrame | None
                Data with dropped column if it was present.

            """
            X_dropped = X.drop(col, axis=1, errors="ignore") if X is not None else X
            return X_dropped

        # Extract validation configurations
        validation_config = self.medpipe_config.workflow.validation
        test_split_config = validation_config.test_split
        cross_val_config = validation_config.cross_validation

        test_split_grp_col = test_split_config.group_column
        groups = None

        cross_val_grp_col = cross_val_config.group_column if cross_val_config else None
        cross_val_strat = cross_val_config.strategy if cross_val_config else None

        if cross_val_strat == "group" and cross_val_grp_col:
            # If a cross-validation group column is specified drop it
            if cross_val_grp_col in X_train.columns:
                groups = X_train[cross_val_grp_col].to_numpy()
            X_train = _drop_column(X_train, cross_val_grp_col)
            X_test = _drop_column(X_test, cross_val_grp_col)
            X_recal = _drop_column(X_recal, cross_val_grp_col)

        if (
            cross_val_strat == "group"
            and test_split_config.strategy == "group"
            and cross_val_grp_col != test_split_grp_col
        ):
            # If cross-validation and test group columns are different
            X_train = _drop_column(X_train, test_split_grp_col)
            X_test = _drop_column(X_test, test_split_grp_col)
            X_recal = _drop_column(X_recal, test_split_grp_col)

            # No need to check test_split_grp_col so exit early
            return X_train, X_test, X_recal, groups

        if test_split_config.strategy == "group" and test_split_grp_col:
            # If a test split group column is specified drop it
            X_train = _drop_column(X_train, test_split_grp_col)
            X_test = _drop_column(X_test, test_split_grp_col)
            X_recal = _drop_column(X_recal, test_split_grp_col)
        return X_train, X_test, X_recal, groups

    def _prepare_features(
        self, X_train: pd.DataFrame, X_recal: pd.DataFrame | None
    ) -> tuple[TransformedData, TransformedData | None]:
        """
        Prepares features of the train and recalibration data sets
        by transforming them with the preprocessor.

        The preprocessor is fitted on X_train.

        Parameters
        ----------
        X_train : pd.DataFrame
            Train data.
        X_recal : pd.DataFrame | None
            Recalibration data if needed in the pipeline, None otherwise.

        Returns
        -------
        X_train : TransformedData
            Processed train data.
        X_recal : TransformedData | None
            Processed recalibration data or None.

        Raises
        ------
        ValueError
            If the preprocessor does not exist.

        """
        try:
            assert self.preprocessor
        except AssertionError:
            raise ValueError("No preprocessor was found")

        # Fit preprocessor on X_train than transform datasets
        X_train = self.preprocessor.fit_transform(X_train)

        if X_recal is not None:
            X_recal = self.preprocessor.transform(X_recal)

        try:  # Try to convert to float
            X_train = X_train.astype(np.float32)
            if X_recal is not None:
                X_recal = X_recal.astype(np.float32)
        except ValueError:
            pass

        return (X_train, X_recal)

    def _get_cv_generator(self) -> StratifiedKFold | GroupKFold:
        """
        Creates a cross-validation generator from the configuration.

        Returns
        -------
        cv_generator : StratifiedKFold | GroupKFold
            Cross-validation generator.

        """
        cv_config = self.medpipe_config.workflow.validation.cross_validation
        assert cv_config
        kwargs = cv_config.model_dump()  # Keyword args for cv generators

        # Get strategy and remove group_column for keyword args
        strategy = kwargs.pop("strategy")
        kwargs.pop("group_column")

        if strategy == "random":
            return StratifiedKFold(**kwargs)
        else:
            return GroupKFold(**kwargs)

    def _cv_fit(
        self,
        X_train: TransformedData,
        y_train: Labels,
        outcome: str,
        cv_generator: StratifiedKFold | GroupKFold,
        groups: npt.NDArray | None = None,
        X_recal: TransformedData | None = None,
        y_recal: Labels | None = None,
    ) -> dict[str, Any]:
        """
        Performs cross-validation and fitting of predictor and recalibrator.

        Parameters
        ----------
        X_train : TransformedData
            Training data.
        y_train : Labels
            Training labels.
        outcome : str
            Outcome to predict.
        cv_generator : StratifiedKFold | GroupKFold
            Cross-validation generator.
        groups : npt.NDArray | None, default: None
            Groups for the GroupKFold cross-validation.
        X_recal : TransformedData | None, default: None
            Recalibration data, or None if no recalibrator.
        y_recal : Labels | None, default: None
            Recalibration labels, or None if no recalibrator.

        Returns
        -------
        cv_results : dict[str, Any]
            Cross-validation results for predictor and recalibrator.

        Raises
        ------
        ValueError
            If no recalibration data is specified with a recalibrator.

        """
        if self.recalibrator and (X_recal is None or y_recal is None):
            raise ValueError(
                "Recalibration dataset is needed when recalibrator is specified"
            )

        cv_results = self._cross_validate_and_fit(
            outcome, X_train, y_train.ravel(), cv_generator, groups
        )

        # Save the fold outputs for the predictor
        self._save_fold_outputs(outcome, cv_results, X_train, groups)

        if self.recalibrator and self.recalibrator_method:
            # Final fit for the recalibrator
            raw_outputs = self.predictor[outcome].predict_proba(X_recal)
            self.recalibrator[outcome].fit(raw_outputs[:, 1], y_recal)

        return cv_results

    def _cross_validate_and_fit(
        self,
        outcome: str,
        X_train: TransformedData,
        y_train: Labels,
        cv_generator: StratifiedKFold | GroupKFold,
        groups: npt.NDArray | None,
    ) -> dict[str, Any]:
        """
        Computes the cross-validation step for the predictor and fits the model.

        Parameters
        ----------
        X_train : TransformedData
            Training data.
        y_train : Labels
            Training labels.
        outcome : str
            Outcome to predict.
        cv_generator : StratifiedKFold | GroupKFold
            Cross-validation generator.
        groups : npt.NDArray | None, default: None
            Groups for the GroupKFold cross-validation.

        Returns
        -------
        cv_results : dict[str, Any]
            Cross-validation results for predictor.

        """
        scorers = build_scorers(self.metrics)

        cv_results = cross_validate(  # Generate cross-validation results
            self.predictor[outcome],
            X=X_train,
            y=y_train,
            cv=cv_generator,
            groups=groups,
            return_estimator=True,
            return_indices=True,
            scoring=scorers,
        )
        self.predictor[outcome].fit(X_train, y_train)  # Fit predictor

        return cv_results

    def _save_fold_outputs(
        self,
        outcome: str,
        cv_results: dict[str, Any],
        X_train: TransformedData,
        groups: npt.NDArray | None,
    ) -> None:
        """
        Save fold outputs from predictor.

        Parameters
        ----------
        outcome : str
            Outcome to predict.
        cv_results : dict[str, Any]
            Cross-validation results for predictor and recalibrator.
        X_train : TransformedData
            Training data.
        groups : npt.NDArray | None, default: None
            Groups for the GroupKFold cross-validation.

        Returns
        -------
        None
            Nothing is returned.

        """
        for fold_idx, (estimator, test_idx) in enumerate(
            zip(cv_results["estimator"], cv_results["indices"]["test"])
        ):
            # Iterate to get the training predictions for the predictor
            fold_test_data = (
                X_train[test_idx]
                if isinstance(X_train, (np.ndarray, csr_matrix, csr_array))
                else X_train.iloc[test_idx]
            )
            raw_outputs = estimator.predict_proba(fold_test_data)
            fold_name = fold_idx

            if groups is not None:
                fold_name = groups[test_idx][0]  # Get the first group name

            # Update dictionaries
            self.folds[outcome].update({fold_name: fold_idx})
            self.predictor_train_outputs[outcome].update({fold_name: raw_outputs})

    def _print_fold_metrics(self, cv_results: dict[str, Any], outcome: str) -> None:
        """
        Prints fold metrics to the terminal.

        Parameters
        ----------
        cv_results : dict[str, Any]
            Cross-validation results for predictor and recalibrator.
        outcome : str
            Outcome to predict.

        Returns
        -------
        None
            Nothing is returned.

        """
        print(f"Outcome: {outcome}")

        test_fold_results = self._extract_fold_results(cv_results)

        for fold_name, fold_idx in self.folds[outcome].items():
            print(f"  Fold: {fold_name}")
            for i in range(len(self.metrics)):
                print_metrics(
                    np.array([test_fold_results[i, fold_idx]]), [self.metrics[i]]
                )

    def _extract_fold_results(self, cv_results: dict[str, Any]) -> npt.NDArray:
        """
        Extract fold results from the cross-validation results.

        Parameters
        ----------
        cv_results : dict[str, Any]
            Cross-validation results for predictor and recalibrator.

        Returns
        -------
        results : npt.NDArray
            Results using the metrics list indexing.

        """
        prefix = "test_"
        assert self.medpipe_config.workflow.validation.cross_validation
        n_splits = self.medpipe_config.workflow.validation.cross_validation.n_splits

        results = np.zeros((len(self.metrics), n_splits))  # Store results

        for i, metric in enumerate(self.metrics):
            results[i, :] = cv_results[prefix + metric]

        return results

    def _prepare_data(
        self,
        X: pd.DataFrame,
        X_recal: pd.DataFrame | None = None,
        fit: bool = False,
    ) -> tuple[TransformedData, TransformedData | None]:
        """
        Prepares data for the functions outside the cross-validate loop.

        If a preprocessor object exists, the data is preprocessed
        otherwise, data types are converted to categoricals and
        unused columns are dropped. If the fit flag is True the
        preprocessor object is fitted on X.

        Parameters
        ----------
        X : pd.DataFrame
            Main data to use of shape (n_samples, n_features).
        X_recal : pd.DataFrame | None, default: None
            Data for the recalibrator of shape
            (n_samples, n_features) or None.
        fit : bool
            If True the preprocessor object is fitted with X.

        Returns
        -------
        X_processed : TransformedData | npt.NDArray
            Main processed data of shape (n_samples, n_features).
        X_recal_processed : TransformedData | None
            Processed data for the recalibrator of shape
            (n_samples, n_features) or None.

        Raises
        ------
        TypeError
            If X is not a pd.DataFrame.
            If X_recal is not a pd.DataFrame or None.

        """
        if not isinstance(X, pd.DataFrame):
            expr = f"Input X should be pd.DataFrame, but got {type(X)}"
            raise TypeError(expr)
        if X_recal is not None and not isinstance(X_recal, pd.DataFrame):
            expr = f"Input X_recal should be pd.DataFrame, but got {type(X_recal)}"
            raise TypeError(expr)

        # Drop group columns if they are present
        X_processed, _, X_recal_processed, _ = self._drop_group_columns(
            X, None, X_recal
        )

        # Preprocess based on presence of preprocessor and fit flag
        if self.preprocessor and fit:
            X_processed, X_recal_processed = self._prepare_features(
                X_processed, X_recal_processed
            )

        elif self.preprocessor:
            check_is_fitted(self.preprocessor)
            X_processed = self.transform(X_processed)
            if X_recal_processed is not None:
                X_recal_processed = self.transform(X_recal_processed)

        # Convert data types if no preprocessor
        elif isinstance(X_processed, pd.DataFrame):
            X_processed = convert_dtypes(X_processed)

            if X_recal_processed is not None:
                X_recal_processed = convert_dtypes(X_recal_processed)

        return X_processed, X_recal_processed

    def _get_strata_idx(
        self, data: pd.DataFrame, idx_data: pd.DataFrame
    ) -> tuple[list[str], list[npt.NDArray]]:
        """
        Extracts the indices of the different strata from input data.

        Requires the presence of the fairness configuration parameters.

        Parameters
        ----------
        data : pd.DataFrame
            Original data containing the strata columns.
        idx_data : pd.DataFrame
            Subset of data for which to find the indices for.

        Returns
        -------
        strata : list[str]
            List of strata names.
        strata_idx : list[npt.NDArray]
            List of the indices for each strata.

        Raises
        ------
        AssertionError
            If the fairness parameters are not provided in medpipe_config.
        ValueError
            If a strata column is not in data.

        """
        # Check that fairness data is provided
        fairness_config = self.medpipe_config.workflow.evaluation.fairness
        assert fairness_config

        columns = fairness_config.strata
        groups = fairness_config.groups
        assert columns
        assert groups

        strata = []  # Empty list to hold all strata
        strata_idx = []  # Empty list to hold strata indices

        for col in columns:
            if col not in data.columns:
                raise ValueError(f"Strata column '{col}' not found in data.")

            # Case 1: Custom range limits provided (e.g., age groups)
            if col in groups and groups[col]:
                for limits in groups[col]:
                    lower, upper = limits
                    # Find matching rows within the inclusive range
                    mask = (idx_data[col] >= lower) & (idx_data[col] <= upper)
                    indices = np.where(mask)[0]

                    strata.append(f"{lower} — {upper}")
                    strata_idx.append(indices)

            # Case 2: Standard categorical column (e.g., gender)
            else:
                # Get unique values present in the original data to keep consistent mapping
                unique_values = data[col].dropna().unique()

                for value in unique_values:
                    mask = idx_data[col] == value
                    indices = np.where(mask)[0]

                    strata.append(value)
                    strata_idx.append(indices)

        return strata, strata_idx

    def _classifier_plots(
        self,
        X: pd.DataFrame,
        y: Labels,
        strata: list[str],
        strata_idx: list[npt.NDArray],
    ) -> None:
        """Plot key figures for a classifier.

        Parameters
        ----------
        X : pd.DataFrame
            Data to use for plotting.
        y : Labels
            Labels associated with the data.
        strata : list[str]
            List of strata names.
        strata_idx : list[npt.NDArray]
            List of the indices for each strata.

        Returns
        _______
        None
            Nothing is returned.

        """
        save_path = self.medpipe_config.top_level.paths.figure_dir  # Save path
        version = self.medpipe_config.top_level.meta.version

        calibration_config = self.medpipe_config.workflow.evaluation.calibration
        calibration_kwargs = (
            calibration_config.model_dump() if calibration_config is not None else {}
        )

        scores = np.zeros((self.n_outcomes, len(self.metrics)))
        strata_scores = np.zeros((len(strata), self.n_outcomes, len(self.metrics)))

        for i, outcome in enumerate(self.outcomes):
            # Plots for each outcome
            raw_predictions = self.predict_proba(X, outcome, "predictor")[0]
            plot_probability_distribution(
                raw_predictions,
                label="Distribution",
                show_fig=False,
                dpi=300,
                set_title=f"{outcome} predicted distribution",
                save_path=save_path + f"{outcome}_distribution_{version}",
            )
            plot_ROC_curve(
                y[:, i],
                raw_predictions[:, i],
                label="ROC",
                show_fig=False,
                dpi=300,
                set_title=f"{outcome} ROC curve",
                save_path=save_path + f"{outcome}_ROC_curve_{version}",
            )
            plot_reliability_diagram(
                y[:, i],
                raw_predictions,
                **calibration_kwargs,
                show_fig=False,
                dpi=300,
                save_path=save_path + f"{outcome}_calibration_{version}",
            )
            scores[i, :] = compute_metrics(self.metrics, y[:, i], raw_predictions)

            for j, idx in enumerate(strata_idx):
                # Get scores for different strata
                strata_predictions = self.predict_proba(
                    X.iloc[idx], outcome, "predictor"
                )[0]
                strata_scores[j, i, :] = compute_metrics(
                    self.metrics, y[idx, i], strata_predictions
                )

        for i, metric in enumerate(self.metrics):
            plot_strata_heatmap(
                self.outcomes,
                metric,
                strata,
                scores[:, i],
                strata_scores[:, :, i],
                show_fig=False,
                dpi=300,
                set_title=f"{METRIC_MAPPING[metric][-1]} fairness heatmap",
                save_path=save_path + f"{metric}_fairness_{version}",
            )

    def _print_test_metrics(
        self,
        results: npt.NDArray,
        outcome: str,
        recal_results: npt.NDArray | None = None,
    ) -> None:
        """
        Prints final test metrics to the terminal.

        Parameters
        ----------
        results : npt.NDArray
            Test results for predictor.
        outcome : str
            Outcome to predict.
        recal_results : npt.NDArray | None, default: None
            Recalibration results or None.

        Returns
        -------
        None
            Nothing is returned.

        """
        print(f"Outcome: {outcome}")

        for i in range(len(self.metrics)):
            print_metrics(np.array([results[i]]), [self.metrics[i]])

            # Run loop a second time to get recalibration afterwards
            if recal_results is not None:
                print(f"  Recalibrated:")
                for i in range(len(self.metrics)):
                    print_metrics(np.array([recal_results[i]]), [self.metrics[i]])

    def _validate_predict_inputs(
        self,
        outcomes: str | list[str],
        estimator_type: Literal["predictor", "recalibrator"] | list[str],
    ) -> tuple[list[str], list[str]]:
        """
        Validate the inputs to the predict and predict_proba functions.

        Parameters
        ----------
        outcomes : str | list[str]
            Label or list of labels associated with the model to use.
            If all, all models are used.
        estimator_type : Literal["predictor", "recalibrator"] | list[str],
            Estimator type or list of estimator types to use.

        Returns
        -------
        valid_outcomes : list[str]
            Validated outcomes.
        estimators : list[str]
            Validated estimator types.

        Raises
        ------
        TypeError
            If outcomes is not a string, list of strings, or 'all'.
            If estimator_types is not 'predictor', 'recalibrator',
            or a list of strings.
        ValueError
            If outcomes and estimators are not the same length.

        """
        # Safety checks
        if outcomes == "all":  # Get all outcomes
            outcomes = self.outcomes
        if isinstance(outcomes, str):
            outcomes = [outcomes]  # Convert to list
        if not isinstance(outcomes, list):
            expr = (
                "Input outcomes should be a string, 'all', or a list of strings, "
                f"but got {type(outcomes)}"
            )
            raise TypeError(expr)

        if estimator_type == "predictor" or estimator_type == "recalibrator":
            # Create a list with exact same estimator type
            estimators = [estimator_type] * len(outcomes)  # type: ignore
        elif not isinstance(estimator_type, list):
            expr = (
                "Input estimator_type should be a 'predictor', 'recalibrator' "
                f"or list of strings, but got {type(estimator_type)}"
            )
            raise TypeError(expr)
        else:
            # If already a list
            estimators: list[str] = estimator_type

        if len(outcomes) != len(estimators):
            expr = (
                "Inputs outcomes and estimator_type should be the same length, "
                f"but got {len(outcomes)} and {len(estimator_type)}"
            )
            raise ValueError(expr)

        return (outcomes, estimators)

    def transform(self, X: pd.DataFrame | npt.NDArray) -> TransformedData:
        """
        Transforms input data based on preprocessor fitted operations.

        Parameters
        ----------
        X : pd.Dataframe | npt.NDArray
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
            check_is_fitted(self.preprocessor)
            return self.preprocessor.transform(X)

        warn("No preprocessor object created so data not transformed")
        return X

    def fit_transform(self, X: pd.DataFrame | npt.NDArray) -> TransformedData:
        """
        Fits the preprocessor operations and transforms the input data.


        Parameters
        ----------
        X : pd.DataFrame | npt.NDArray
            Data to clean of shape (n_samples, n_features).

        Returns
        -------
        data : pd.DataFrame | npt.NDArray | csr_matrix | csr_array:
             Transformed data of shape (n_samples, n_features).
             Returns X if no preprocessor exists.

        Warns
        -----
        UserWarning
            If no preprocessor object exists.

        """
        if self.preprocessor:
            return self.preprocessor.fit_transform(X)

        warn("No preprocessor object created so data not transformed")
        return X

    def fit(
        self,
        X: pd.DataFrame,
        y: Labels,
        X_recal: pd.DataFrame | None = None,
        y_recal: Labels | None = None,
    ) -> None:
        """
        Fit the MedpipePipeline estimators.

        X and X_recal are transformed if there is a
        preprocessor object.
        X_recal is first passed through the predictor object.

        Parameters
        ----------
        X : pd.DataFrame
            Data to fit on of shape (n_samples, n_features).
        y : Labels
            Ground truth labels of shape (n_samples,).
        X_recal : pd.DataFrame | None
            Data to train the recalibrator
            of shape (n_samples, n_features) or None.
        y_recal : Labels | None
            Ground truth labels for the recalibration data
            of shape (n_samples,).

        Returns
        -------
        None
            Nothing is returned.

        Raises
        ------
        ValueError
            If no recalibration data is specified with a recalibrator.

        """
        X_train, X_recal_fit = self._prepare_data(X, X_recal, fit=True)

        for i, outcome in enumerate(self.outcomes):
            if len(y.shape) == 2 and y.shape[1] > 1:
                labels = y[:, i]
            else:
                labels = y.squeeze()
            predictor = self.predictor[outcome]
            predictor.fit(X_train, labels)

            if self.recalibrator:
                # Fit the recalibrators if specified
                if X_recal is None or y_recal is None:
                    raise ValueError(
                        f"Cannot fit {outcome} recalibrator without recalibration data"
                    )

                if hasattr(predictor, "predict_proba"):
                    raw_outputs = predictor.predict_proba(X_recal_fit)
                    raw_outputs = raw_outputs[:, 1]  # Only take positive class
                else:
                    raw_outputs = predictor.predict(X_recal_fit)
                self.recalibrator[outcome].fit(raw_outputs, y_recal[:, i])

    def get_data_sets(self, data: pd.DataFrame) -> tuple[
        pd.DataFrame,
        Labels,
        pd.DataFrame,
        Labels,
        pd.DataFrame | None,
        Labels | None,
        npt.NDArray | None,
    ]:
        """
        Splits data into train, test, and recalibration sets.

        Only the columns containing the predictors and the features
        are extracted and split.
        Returns in order X_train, y_train, X_test, y_test, X_recal,
        y_recal, group_column.
        If no recalibration is required, X_recal and y_recal are None.
        The group_column is the data column of the train set based on
        cross-validation parameters.

        Parameters
        ----------
        data : pd.DataFrame
            Data to split.

        Returns
        -------
        X_train, X_test : pd.DataFrame
            Train and test data.
        y_train, y_test : Labels
            Train and test labels.
        X_recal : pd.DataFrame | None
            Recalibration data if needed in the pipeline, None otherwise.
        y_recal : Labels | None
            Recalibration labels if needed in the pipeline, None otherwise.
        groups : npt.NDArray | None
            Train set group column for cross-validation, None if not
            specified.

        Raises
        ------
        TypeError
            If data is not a pd.DataFrame.
        ValueError
            If some predictors or outcomes are not in the data.

        """
        # Extract labels from the data
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"data should be a pd.DataFrame, but got {type(data)}")

        # Extract validation configurations
        validation_config = self.medpipe_config.workflow.validation
        test_split_config = validation_config.test_split
        cross_val_config = validation_config.cross_validation

        # Create list of columns to extract from the data
        column_list = self.outcomes + self.medpipe_config.data.predictors
        if test_split_config.strategy == "group" and test_split_config.group_column:
            column_list.append(test_split_config.group_column)
        if cross_val_config:
            if cross_val_config.strategy == "group" and cross_val_config.group_column:
                column_list.append(cross_val_config.group_column)

        try:
            pipeline_data = pd.DataFrame(data[column_list])
        except KeyError:
            raise ValueError("Some outcomes or predictors are not in the data")

        features, labels = extract_labels(pipeline_data, self.outcomes)

        # Set to default value None
        X_recal = None
        y_recal = None
        groups = None

        # Split test set
        X_train, y_train, X_test, y_test = split_data(
            features, labels, **test_split_config.model_dump()
        )

        if validation_config.recalibration_split:
            recal_split_config = validation_config.recalibration_split

            _, _, X_recal, y_recal = split_data(
                features, labels, **recal_split_config.model_dump()
            )
            y_recal = y_recal.astype(int)  # Convert to ints

        # Drop group columns
        X_train, X_test, X_recal, groups = self._drop_group_columns(
            X_train, X_test, X_recal
        )

        assert X_test is not None  # Add check just in case

        X_train = convert_dtypes(X_train)
        X_test = convert_dtypes(X_test)
        if X_recal is not None:
            X_recal = convert_dtypes(X_recal)

        return (
            X_train,
            y_train.astype(int),
            X_test,
            y_test.astype(int),
            X_recal,
            y_recal,
            groups,
        )

    def run(self, data: pd.DataFrame | None = None) -> None:
        """
        Run the entire pipeline from preprocessing to plotting.

        If data is None, the data path from the configuration is used.

        Parameters
        ----------
        data : pd.DataFrame | None, default: None
            Data to fit on of shape (n_samples, n_features) or None.

        Returns
        -------
        None
            Nothing is returned.

        Raises
        ------
        TypeError
            If data is not a pd.DataFrame.

        """
        if data is None:
            data = load_data(self.medpipe_config.data.path)

        if not isinstance(data, pd.DataFrame):
            expr = f"Input data should be a pd.DataFrame, but got {type(data)}"
            raise TypeError(expr)

        # Get different split data sets
        X_train, y_train, X_test, y_test, X_recal, y_recal, groups = self.get_data_sets(
            data
        )

        run_mode = self.medpipe_config.top_level.meta.run_mode
        if run_mode != "fast":
            if self.preprocessor:
                X_train, X_recal = self._prepare_features(X_train, X_recal)
            # Create cross-validation generator
            cv_generator = self._get_cv_generator()

            for i, outcome in enumerate(self.outcomes):
                # Perform cross-validation for each outcome and fit model
                recal_labels = y_recal[:, i] if y_recal is not None else None
                cv_results = self._cv_fit(
                    X_train,
                    y_train[:, i],
                    outcome,
                    cv_generator,
                    groups=groups,
                    X_recal=X_recal,
                    y_recal=recal_labels,
                )

                self._print_fold_metrics(cv_results, outcome)
        else:
            self.fit(X_train, y_train, X_recal, y_recal)

        print("Final test results")
        self.test_models(X_test, y_test)

        if run_mode == "audit" and is_classifier(self.predictor[self.outcomes[0]]):
            # Only run if the predictors are classifiers and run mode is audit
            strata_idx = self._get_strata_idx(data, X_test)
            self._classifier_plots(X_test, y_test, *strata_idx)

    def save(self) -> None:
        """
        Saves the pipeline as {project_name}_{version}.joblib.

        Returns
        -------
        None
            Nothing is returned.

        """
        top_level = self.medpipe_config.top_level

        save_name = f"{top_level.meta.project_name}_{self.version}.joblib"
        save_dir = Path(self.medpipe_config.top_level.paths.model_dir)
        save_dir = save_dir.expanduser().resolve()

        joblib.dump(self, save_dir / save_name)

    def test_models(self, X: pd.DataFrame, y: Labels) -> None:
        """
        Tests the predictor or recalibrator models on the provided dataset.

        X is transformed if there is a preprocessor object.

        Parameters
        ----------
        X : pd.DataFrame
            Test data of shape (n_samples, n_features).
        y : Labels
            Prediction labels of shape (n_samples,).

        Returns
        -------
        None
            Nothing is returned.

        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Input X should be a pd.DataFrame, but got {type(X)}")
        X_processed, _ = self._prepare_data(X, None)

        for i, outcome in enumerate(self.outcomes):
            recal_results = None

            if hasattr(self.predictor[outcome], "predict_proba"):
                raw_outputs = self.predictor[outcome].predict_proba(X_processed)
                raw_outputs = raw_outputs[:, 1]
                results = compute_metrics(self.metrics, y[:, i].ravel(), raw_outputs)
            else:
                raw_outputs = self.predictor[outcome].predict(X_processed)
                results = compute_metrics(self.metrics, y[:, i].ravel(), raw_outputs)

            if self.recalibrator:
                recal_results = compute_metrics(
                    self.metrics,
                    y.ravel(),
                    self.recalibrator[outcome].predict(raw_outputs),
                )

            self._print_test_metrics(results, outcome, recal_results)

    def predict_proba(
        self,
        X: pd.DataFrame,
        outcomes: str | list[str] = "all",
        estimator_type: Literal["predictor", "recalibrator"] | list[str] = "predictor",
    ) -> list[npt.NDArray]:
        """
        Predicts probabilities from predictor or recalibrator based on input data.

        The function checks if a predict_proba method exist for each model.
        The function preprocesses X if a preprocessor exists.

        Parameters
        ----------
        X : pd.DataFrame
            Data to use of shape (n_samples, n_features).
        outcomes : str | list[str], default: "all"
            Label or list of labels associated with the model to use.
            If all, all models are used.
        estimator_type : Literal["predictor", "recalibrator"] | list[str],
            default: "predictor"
            Model type or list of model types to use.

        Returns
        -------
        probabilities : list[npt.NDArray]
            List of probabilities of shape (n_samples, 2).

        Raises
        ------
        NotImplementedError
            If no predict_proba method exists in predictor.
        ValueError
            If no recalibrator is present with a recalibrator estimator.

        """
        valid_outcomes, estimators = self._validate_predict_inputs(
            outcomes, estimator_type
        )
        # Process data if needed
        X_processed, _ = self._prepare_data(X, None)

        probabilities = []
        for i, outcome in enumerate(valid_outcomes):
            predictor = self.predictor[outcome]
            check_is_fitted(predictor)
            if not hasattr(predictor, "predict_proba"):
                expr = (
                    f"Predictor of type {type(predictor).__name__} does not implement "
                    "the predict_proba method"
                )
                raise NotImplementedError(expr)
            outputs = predictor.predict_proba(X_processed)
            probabilities.append(outputs)

            if estimators[i] == "recalibrator":
                if self.recalibrator:
                    recalibrator = self.recalibrator[outcome]
                    check_is_fitted(recalibrator)
                    pos_proba = recalibrator.predict(outputs[:, 1])
                    probabilities[i] = np.array([1 - pos_proba, pos_proba]).T

                else:
                    raise ValueError("No recalibrator present in pipeline")

        return probabilities

    def predict(
        self,
        X: pd.DataFrame,
        outcomes: str | list[str] = "all",
        estimator_type: Literal["predictor", "recalibrator"] | list[str] = "predictor",
    ) -> list[npt.NDArray]:
        """
        Predicts outputs from predictor or recalibrator based on input data.

        The function preprocesses X if a preprocessor exists.

        Parameters
        ----------
        X : pd.DataFrame
            Data to use of shape (n_samples, n_features).
        outcomes : str | list[str], default: "all"
            Label or list of labels associated with the model to use.
            If all, all models are used.
        estimator_type : Literal["predictor", "recalibrator"] | list[str],
            default: "predictor"
            Model type or list of model types to use.

        Returns
        -------
        outputs : list[npt.NDArray]
            List of outputs of shape (n_samples,).

        Raises
        ------
        ValueError
            If no recalibrator is present with a recalibrator estimator.

        """
        valid_outcomes, estimators = self._validate_predict_inputs(
            outcomes, estimator_type
        )
        X_processed, _ = self._prepare_data(X, None)

        outputs = []
        for i, outcome in enumerate(valid_outcomes):
            predictor = self.predictor[outcome]
            check_is_fitted(predictor)
            raw_outputs = predictor.predict(X_processed)
            outputs.append(raw_outputs)

            if estimators[i] == "recalibrator":
                if self.recalibrator:
                    recalibrator = self.recalibrator[outcome]
                    check_is_fitted(recalibrator)
                    outputs[i] = recalibrator.predict(raw_outputs)

                else:
                    raise ValueError("No recalibrator present in pipeline")

        return outputs
