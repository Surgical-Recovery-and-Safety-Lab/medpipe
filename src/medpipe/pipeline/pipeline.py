"""
Main entry point module for the Medpipe machine learning package.

Provides a unified, high-level interface (`Medpipe`) orchestrating data preparation,
model fitting, inference, and TRIPOD+AI compliant evaluation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import numpy.typing as npt
import pandas as pd

from medpipe.pipeline.evaluator import MedpipeEvaluator
from medpipe.pipeline.orchestrator import MedpipeOrchestrator
from medpipe.pipeline.runner import MedpipeRunner
from medpipe.utils.config import MedpipeConfig
from medpipe.utils.logger import get_console_logger


class Medpipe:
    """
    User entry point and high-level pipeline runner for Medpipe.

    Coordinates the complete machine learning lifecycle, delegating data ingress
    and preprocessing setup to `MedpipeOrchestrator`, model cross-validation
    and fitting to `MedpipeRunner`, and prediction and TRIPOD+AI evaluation to `MedpipeEvaluator`.

    Parameters
    ----------
    config : str, Path, or MedpipeConfig
        Path to the TOML configuration file or an instantiated MedpipeConfig object.
    base_artifact_dir : str or Path, default="artifacts"
        Root directory where versioned execution run artifacts, logs, and models are stored.

    Attributes
    ----------
    orchestrator : MedpipeOrchestrator
        Pipeline orchestrator instance driving data preparation and reproducibility artifacts.
    runner : MedpipeRunner
        Pipeline execution engine responsible for model training and cross-validation loops.
    evaluator : MedpipeEvaluator
        Pipeline evaluation engine computing point estimates and bootstrap confidence intervals.
    logger : logging.Logger
        Centralized logger instance configured under `"medpipe"`.

    Methods
    -------
    fit(X_train, y_train, X_recal=None, y_recal=None, groups_train=None)
        Fit machine learning models across configured target outcomes via MedpipeRunner.
    predict(X, model=None, outcome=None)
        Predict class labels for input samples.
    predict_proba(X, model=None, outcome=None)
        Predict class probabilities for input samples.
    decision_function(X, model=None, outcome=None)
        Compute decision function confidence scores for input samples.
    evaluate(X, y, outcome=None, model=None, metrics=None, subgroup_specs=None, save_artifacts=True)
        Evaluate model performance with confidence intervals on full datasets and subgroups.
    run(subgroup_specs=None, groups_train=None)
        Execute full end-to-end pipeline (data preparation, model fitting, and test evaluation).

    """

    def __init__(
        self,
        config: Union[str, Path, MedpipeConfig],
        base_artifact_dir: Union[str, Path] = "artifacts",
    ) -> None:
        self.logger = get_console_logger("medpipe")
        self.logger.info("Initialising Medpipe end-to-end pipeline.")

        self.orchestrator = MedpipeOrchestrator(config, base_artifact_dir)
        self.mp_config = self.orchestrator.config
        self.runner = MedpipeRunner(orchestrator=self.orchestrator)
        self.evaluator = MedpipeEvaluator(
            orchestrator=self.orchestrator,
            runner=self.runner,
        )

        self.logger.info("Medpipe initialisation complete.")

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_recal: Optional[pd.DataFrame] = None,
        y_recal: Optional[pd.DataFrame] = None,
        groups_train: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Fit machine learning models across configured target outcomes via MedpipeRunner.

        Parameters
        ----------
        X_train : pandas.DataFrame
            Training feature dataset.
        y_train : pandas.DataFrame
            Training outcome target labels (can contain multiple outcome columns).
        X_recal : pandas.DataFrame, optional
            Recalibration feature dataset for post-hoc model calibration, default=None.
        y_recal : pandas.DataFrame, optional
            Recalibration outcome target labels, default=None.
        groups_train : numpy.ndarray, optional
            Group identifier array for group-based cross-validation splits, default=None.

        Returns
        -------
        fitted_models : dict of str to object
            Dictionary mapping outcome target names to their finalized fitted models.

        """
        self.logger.info("Starting model fitting across configured target outcomes.")
        fitted_models = self.runner.run(
            X_train=X_train,
            y_train_df=y_train,
            X_recal=X_recal,
            y_recal_df=y_recal,
            groups_train=groups_train,
        )
        self.logger.info("Completed model fitting for all target outcomes.")
        return fitted_models

    def predict(
        self,
        X: Union[pd.DataFrame, npt.NDArray],
        model: Optional[Any] = None,
        outcome: Optional[str] = None,
    ) -> npt.NDArray:
        """
        Predict class labels for input samples.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Features dataset of shape (n_samples, n_features).
        model : object, optional
            Fitted model instance. If None, resolved via `outcome` or `runner.fitted_models`.
        outcome : str, optional
            Outcome name corresponding to an entry in `runner.fitted_models`.

        Returns
        -------
        y_pred : numpy.ndarray
            Predicted class labels of shape (n_samples,).

        """
        return self.evaluator.predict(X=X, model=model, outcome=outcome)

    def predict_proba(
        self,
        X: Union[pd.DataFrame, npt.NDArray],
        model: Optional[Any] = None,
        outcome: Optional[str] = None,
    ) -> npt.NDArray:
        """
        Predict class probabilities for input samples.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Features dataset of shape (n_samples, n_features).
        model : object, optional
            Fitted model instance. If None, resolved via `outcome` or `runner.fitted_models`.
        outcome : str, optional
            Outcome name corresponding to an entry in `runner.fitted_models`.

        Returns
        -------
        y_proba : numpy.ndarray
            Predicted class probabilities of shape (n_samples, n_classes) or (n_samples,).

        """
        return self.evaluator.predict_proba(X=X, model=model, outcome=outcome)

    def decision_function(
        self,
        X: Union[pd.DataFrame, npt.NDArray],
        model: Optional[Any] = None,
        outcome: Optional[str] = None,
    ) -> npt.NDArray:
        """
        Compute decision function confidence scores for input samples.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Features dataset of shape (n_samples, n_features).
        model : object, optional
            Fitted model instance. If None, resolved via `outcome` or `runner.fitted_models`.
        outcome : str, optional
            Outcome name corresponding to an entry in `runner.fitted_models`.

        Returns
        -------
        scores : numpy.ndarray
            Confidence scores or decision function values of shape (n_samples,).

        """
        return self.evaluator.decision_function(X=X, model=model, outcome=outcome)

    def evaluate(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, npt.NDArray],
        outcome: Optional[str] = None,
        model: Optional[Any] = None,
        metrics: Optional[List[str]] = None,
        subgroup_specs: Optional[
            Dict[str, Union[str, Callable[[pd.DataFrame], pd.Series]]]
        ] = None,
        save_artifacts: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate model performance with confidence intervals on full datasets and subgroups.

        Parameters
        ----------
        X : pandas.DataFrame
            Feature dataset of shape (n_samples, n_features).
        y : pandas.Series or numpy.ndarray
            Ground truth target values of shape (n_samples,).
        outcome : str, optional
            Outcome key name. Used for logging, model resolution, and artifact saving.
        model : object, optional
            Explicit fitted model instance. If None, resolved from `runner.fitted_models`.
        metrics : list of str, optional
            List of metrics to evaluate. If None, defaults to `self.evaluator.metrics`.
        subgroup_specs : dict of str to (str or callable), optional
            Specifications for extracting demographic or clinical subgroups.
        save_artifacts : bool, default=True
            Whether to write evaluation summary results to disk via `ArtifactManager`.

        Returns
        -------
        results : dict of str to Any
            Nested dictionary containing outcome identifier, overall slice metrics, and subgroup performance.

        """
        # Resolve single target Series/ndarray if y is supplied as a DataFrame
        if isinstance(y, pd.DataFrame):
            if outcome and outcome in y.columns:
                y_eval = y[outcome]
            elif y.shape[1] == 1:
                y_eval = y.iloc[:, 0]
            else:
                y_eval = y
        else:
            y_eval = y

        return self.evaluator.evaluate(
            X=X,
            y=y_eval,
            outcome=outcome,
            model=model,
            metrics=metrics,
            subgroup_specs=subgroup_specs,
            save_artifacts=save_artifacts,
        )

    def run(
        self,
        groups_train: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Execute full end-to-end pipeline (data preparation, model fitting, and test evaluation).

        Automates data ingestion, split creation, model cross-validation and fitting,
        and test evaluation with TRIPOD+AI reporting across all target outcomes.

        Parameters
        ----------
        groups_train : numpy.ndarray, optional
            Group labels for training samples if group-based cross-validation is configured.

        Returns
        -------
        pipeline_results : dict of str to Any
            Dictionary containing:
            - `"fitted_models"`: Dictionary mapping outcome target names to fitted model instances.
            - `"evaluations"`: Dictionary mapping outcome target names to evaluation results dictionaries.

        """
        self.logger.info("Executing full Medpipe pipeline end-to-end.")

        # 1. Prepare data splits via orchestrator
        self.logger.info("Step 1/3: Ingesting and splitting dataset.")
        X_train, y_train, X_recal, y_recal, X_test, y_test, groups_train = (
            self.orchestrator.prepare_data()
        )

        # 2. Fit models via runner
        self.logger.info("Step 2/3: Fitting outcome models.")
        fitted_models = self.fit(
            X_train=X_train,
            y_train=y_train,
            X_recal=X_recal,
            y_recal=y_recal,
            groups_train=groups_train,
        )

        # 3. Evaluate models on test set via evaluator
        self.logger.info("Step 3/3: Evaluating models on holdout test set.")
        evaluations: Dict[str, Any] = {}
        outcomes = self.orchestrator.config.data.outcomes
        subgroup_specs = self.orchestrator.get_subgroup_specs()

        for outcome in outcomes:
            y_test_outcome = y_test[outcome]
            evaluations[outcome] = self.evaluate(
                X=X_test,
                y=y_test_outcome.to_numpy(),
                outcome=outcome,
                subgroup_specs=subgroup_specs,
                save_artifacts=True,
            )

        self.logger.info("Full Medpipe pipeline execution finished successfully.")

        return {
            "fitted_models": fitted_models,
            "evaluations": evaluations,
        }
