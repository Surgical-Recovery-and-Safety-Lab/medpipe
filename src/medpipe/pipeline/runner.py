import json
from typing import Any, Dict, Optional, Union

import joblib
import ngboost
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, is_regressor
from sklearn.calibration import CalibratedClassifierCV, FrozenEstimator
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import (
    BaseCrossValidator,
    GridSearchCV,
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline

from medpipe.data.transformers import BoundedLogitTransformer
from medpipe.metrics.core import build_scorers
from medpipe.models.registry import ModelRegistry
from medpipe.pipeline.orchestrator import MedpipeOrchestrator
from medpipe.utils.logger import get_console_logger


class MedpipeRunner:
    """
    Executes the training, hyperparameter tuning, and optional recalibration loops.

    The runner acts purely as an execution engine. It requests instantiated
    preprocessing pipelines from the orchestrator, initializes models via the
    ModelRegistry, handles GridSearchCV or standard cross-validation, applies
    post-hoc calibration, and persists the finalized models to the run directory.

    Parameters
    ----------
    orchestrator : MedpipeOrchestrator
        The configured orchestrator instance, providing resolved configurations,
        preprocessing pipelines, and the execution run directory.

    Attributes
    ----------
    orchestrator : MedpipeOrchestrator
        Orchestrator instance driving the environment state.
    logger : logging.Logger
        Logger instance for the runner.
    fitted_models : Dict[str, Union[Pipeline, CalibratedClassifierCV]]
        Dictionary storing the finalized models, keyed by outcome name.

    Methods
    -------
    fit_outcome(outcome, X_train, y_train, X_recal=None, y_recal=None, groups_train=None)
        Trains, evaluates, and optionally calibrates a model for a single outcome.
    run(X_train, y_train_df, X_recal=None, y_recal_df=None, groups_train=None)
        Executes the pipelines for all configured target outcomes.

    """

    def __init__(self, orchestrator: MedpipeOrchestrator) -> None:
        self.orchestrator = orchestrator
        self.logger = get_console_logger("medpipe.runner")
        self.fitted_models: Dict[str, Union[Pipeline, CalibratedClassifierCV]] = {}

    def _instantiate_estimator(
        self, algo_name: str, params: Dict[str, Any]
    ) -> BaseEstimator:
        """
        Instantiates an estimator using the ModelRegistry.

        Automatically wraps regressors or ngboost models in a
        TransformedTargetRegressor to handle bounded logit outputs.

        Parameters
        ----------
        algo_name : str
            The lookup name for the algorithm in the ModelRegistry.
        params : Dict[str, Any]
            The hyperparameters to initialize the estimator with.

        Returns
        -------
        BaseEstimator
            The un-fitted scikit-learn estimator.

        """
        estimator_class = ModelRegistry.get(algo_name)

        # Filter out hyperparameters that are lists/tuples if we are doing GridSearch
        # Initial instantiation should only use scalar/default values if available
        init_params = {
            k: (v[0] if isinstance(v, (list, tuple)) else v) for k, v in params.items()
        }

        estimator = estimator_class(**init_params)

        if is_regressor(estimator) or isinstance(estimator, ngboost.NGBRegressor):
            self.logger.info("Wrapping regressor in TransformedTargetRegressor")
            return TransformedTargetRegressor(
                regressor=estimator,
                transformer=BoundedLogitTransformer(),
                check_inverse=False,
            )

        return estimator

    def _create_cv_splitter(
        self, strategy: str, n_splits: int, random_state: int | None
    ) -> Union[StratifiedKFold, StratifiedGroupKFold]:
        """
        Instantiates the appropriate cross-validation splitter.

        Parameters
        ----------
        strategy : {'random', 'group'}
            The CV strategy to employ.
        n_splits : int
            The number of cross-validation folds.
        random_state : int | None
            The random seed for reproducibility.

        Returns
        -------
        Union[StratifiedKFold, StratifiedGroupKFold]
            The instantiated cross-validation splitter object.

        Raises
        ------
        ValueError
            If an unsupported strategy is provided.

        """
        if strategy == "random":
            return StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=random_state
            )
        elif strategy == "group":
            return StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True, random_state=random_state
            )
        else:
            raise ValueError(f"Strategy must be 'random' or 'group', got {strategy}")

    def _save_model(
        self, model: Union[Pipeline, CalibratedClassifierCV], outcome: str
    ) -> None:
        """
        Saves the fitted model to the orchestrator's run directory.

        Parameters
        ----------
        model : Union[Pipeline, CalibratedClassifierCV]
            The finalized, fully fitted model.
        outcome : str
            The name of the outcome, used to name the saved file.

        """
        save_dir = self.orchestrator.run_dir / "models"
        save_dir.mkdir(exist_ok=True, parents=True)

        file_path = save_dir / f"{outcome}_model.joblib"
        joblib.dump(model, file_path, compress=3)
        self.logger.info(f"[{outcome}] Model saved to {file_path}")

    def _save_final_models(self) -> None:
        """
        Saves the complete dictionary of fitted outcome models into a single
        file in the models directory for downstream evaluation.

        """
        save_dir = self.orchestrator.run_dir / "models"
        save_dir.mkdir(exist_ok=True, parents=True)

        project_name = self.orchestrator.config.meta.project_name
        models_path = save_dir / f"{project_name}_fitted.joblib"
        joblib.dump(self.fitted_models, models_path, compress=3)
        self.logger.info(f"Fitted final models saved to {models_path}")

    def _save_cv_results(self, outcome: str, cv_results_df: pd.DataFrame) -> None:
        """
        Save cross-validation results to disk in the artifacts directory.

        Exports a detailed fold-level CSV file for spreadsheet auditing alongside
        a JSON file containing summary statistics (mean and standard deviation)
        for each test evaluation metric.

        Parameters
        ----------
        outcome : str
            The name of the outcome being evaluated.
        cv_results_df : pd.DataFrame
            DataFrame containing fold-by-fold cross-validation metrics and timing
            information derived from scikit-learn's `cross_validate` or `GridSearchCV`.

        Returns
        -------
        None

        """
        artifacts_dir = self.orchestrator.run_dir / "CV"
        artifacts_dir.mkdir(exist_ok=True, parents=True)

        # Save detailed fold-level CSV
        csv_path = artifacts_dir / f"{outcome}_cv_results.csv"
        cv_results_df.to_csv(csv_path, index=False)
        self.logger.info(f"[{outcome}] Detailed CV results saved to {csv_path}")

        # Extract and save summary report (mean ± std dev for test scores)
        summary = {}
        for col in cv_results_df.columns:
            if col.startswith("test_"):
                metric_name = col.replace("test_", "")
                summary[metric_name] = {
                    "mean": float(cv_results_df[col].mean()),
                    "std": float(cv_results_df[col].std()),
                }

        json_path = self.orchestrator.artifact_manager.save_json(
            summary, artifacts_dir, f"{outcome}_cv_summary.json"
        )
        self.logger.info(f"[{outcome}] CV summary saved to {json_path}")

    def _train_model_cv(
        self,
        outcome: str,
        pipeline: Pipeline,
        hyperparams: dict,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        cv_splitter: BaseCrossValidator,
        groups_train: Optional[np.ndarray],
    ) -> Pipeline:
        """
        Handles hyperparameter tuning via GridSearchCV or standard cross-validation.

        Parameters
        ----------
        outcome : str
            The name of the outcome being modeled.
        pipeline : sklearn.pipeline.Pipeline
            The base pipeline containing the preprocessor and estimator.
        hyperparams : dict
            A dictionary of hyperparameters to apply or search over.
        X_train : pd.DataFrame
            The training feature set.
        y_train : np.ndarray
            The 1D training target labels.
        cv_splitter : BaseCrossValidator
            The cross-validation splitting strategy object.
        groups_train : Optional[np.ndarray]
            Group labels for the training set, used for group-based
            cross-validation splits.

        Returns
        -------
        sklearn.pipeline.Pipeline
            The best fitted pipeline after cross-validation or grid search.

        Raises
        ------
        ValueError
            If the groups_train is None when strategy is group.

        """
        # Retrieve the list of metrics from the configuration safely
        try:
            configured_metrics = (
                self.orchestrator.config.workflow.evaluation.metrics.metrics
            )
        except AttributeError:
            configured_metrics = ["roc_auc"]

        cv_cfg = self.orchestrator.config.workflow.validation.cross_validation
        assert cv_cfg

        if cv_cfg.strategy == "group" and groups_train is None:
            raise ValueError("The 'groups' parameter should not be None")

        scorers_dict = build_scorers(configured_metrics)

        if cv_cfg.grid_search:
            self.logger.info(f"[{outcome}] Running GridSearchCV tuning.")

            pipeline_params = {
                f"classifier__{k}": (v if isinstance(v, (list, tuple)) else [v])
                for k, v in hyperparams.items()
            }

            search = GridSearchCV(
                estimator=pipeline,
                param_grid=pipeline_params,
                cv=cv_splitter,
                scoring=scorers_dict,
                refit=configured_metrics[0],  # type: ignore
                n_jobs=-1,
            )
            search.fit(X_train, y_train, groups=groups_train)

            # Convert search results to DataFrame & save artifacts
            cv_results_df = pd.DataFrame(search.cv_results_)
            self._save_cv_results(outcome, cv_results_df)

            self.logger.info(f"[{outcome}] Best parameters: {search.best_params_}")
            return search.best_estimator_

        else:
            self.logger.info(f"[{outcome}] Running standard cross-validation.")

            results = cross_validate(
                estimator=pipeline,
                X=X_train,
                y=y_train,
                groups=groups_train,
                cv=cv_splitter,
                scoring=scorers_dict,
                n_jobs=-1,
            )

            # Convert cross_validate dict to DataFrame & save artifacts
            cv_results_df = pd.DataFrame(results)
            self._save_cv_results(outcome, cv_results_df)

            self.logger.info(
                f"[{outcome}] Fitting final base pipeline on full training data."
            )
            return pipeline.fit(X_train, y_train)

    def _calibrate_model(
        self,
        outcome: str,
        best_pipeline: Pipeline,
        model_config: dict,
        X_recal: Optional[pd.DataFrame],
        y_recal: Optional[np.ndarray],
    ) -> Union[Pipeline, CalibratedClassifierCV]:
        """
        Wraps the model in a FrozenEstimator and fits a calibrator if holdout data exists.

        Parameters
        ----------
        outcome : str
            The name of the outcome being modeled.
        best_pipeline : sklearn.pipeline.Pipeline
            The fitted base pipeline to be calibrated.
        model_config : dict
            The configuration dictionary for the specific outcome, containing
            recalibration settings.
        X_recal : pd.DataFrame or None
            The recalibration feature set. Can be None if recalibration
            is not performed.
        y_recal : np.ndarray or None
            The 1D recalibration target labels. Can be None if recalibration
            is not performed.

        Returns
        -------
        Union[sklearn.pipeline.Pipeline, sklearn.calibration.CalibratedClassifierCV]
            The calibrated model if recalibration was performed, otherwise
            the original fitted pipeline.

        """
        recal_config = model_config.get("recalibration")

        if X_recal is not None and not X_recal.empty and recal_config:
            self.logger.info(f"[{outcome}] Fitting recalibrator on holdout dataset.")
            calibrator_method = recal_config.get("method")

            calibrator = CalibratedClassifierCV(
                estimator=FrozenEstimator(best_pipeline),
                cv=2,
                method=calibrator_method,
            )
            return calibrator.fit(X_recal, y_recal)

        else:
            self.logger.info(
                f"[{outcome}] Skipping recalibration (missing dataset or configuration)."
            )
            return best_pipeline

    def fit_outcome(
        self,
        outcome: str,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_recal: Optional[pd.DataFrame] = None,
        y_recal: Optional[np.ndarray] = None,
        groups_train: Optional[np.ndarray] = None,
    ) -> Union[Pipeline, CalibratedClassifierCV]:
        """
        Trains, evaluates, and optionally calibrates a model for a single outcome.

        Parameters
        ----------
        outcome : str
            The name of the outcome being modeled.
        X_train : pd.DataFrame
            The training feature set.
        y_train : np.ndarray
            The 1D training target labels.
        X_recal : Optional[pd.DataFrame], default=None
            The recalibration feature set.
        y_recal : Optional[np.ndarray], default=None
            The 1D recalibration target labels.
        groups_train : Optional[np.ndarray], default=None
            Group labels for the training set, required if strategy is 'group'.

        Returns
        -------
        Union[Pipeline, CalibratedClassifierCV]
            The fully trained model ready for evaluation.

        Raises
        ------
        ValueError
            If no algorithm is specified for an outcome

        """
        self.logger.info(f"--- Starting execution for outcome: {outcome} ---")

        model_config = self.orchestrator.config.resolved_models[outcome]
        algo_name = model_config.algorithm

        if not algo_name:
            raise ValueError(f"No algorithm specified for outcome: {outcome}")

        hyperparams = model_config.hyperparameters

        # 1. Build Base Pipeline
        preprocessor = self.orchestrator.build_preprocessor()
        estimator = self._instantiate_estimator(algo_name, hyperparams)

        steps = []
        if preprocessor is not None:
            steps.append(("preprocessor", preprocessor))
        steps.append(("classifier", estimator))
        pipeline = Pipeline(steps)

        # 2. Configure Cross-Validation
        cv_config = self.orchestrator.config.workflow.validation.cross_validation

        # 3. Training Loop
        if self.orchestrator.config.meta.run_mode in ["audit", "cv"]:
            assert cv_config
            assert cv_config.n_splits

            cv_splitter = self._create_cv_splitter(
                strategy=cv_config.strategy,
                n_splits=cv_config.n_splits,
                random_state=self.orchestrator.config.workflow.random_state,
            )

            best_pipeline = self._train_model_cv(
                outcome=outcome,
                pipeline=pipeline,
                hyperparams=hyperparams,
                X_train=X_train,
                y_train=y_train,
                groups_train=groups_train,
                cv_splitter=cv_splitter,
            )
        else:
            best_pipeline = pipeline.fit(X_train, y_train)

        # 4. Optional Post-Hoc Recalibration
        final_model = self._calibrate_model(
            outcome=outcome,
            best_pipeline=best_pipeline,
            model_config=model_config.model_dump(),
            X_recal=X_recal,
            y_recal=y_recal,
        )

        # 5. Save and Return
        self._save_model(final_model, outcome)
        return final_model

    def run(
        self,
        X_train: pd.DataFrame,
        y_train_df: pd.DataFrame,
        X_recal: Optional[pd.DataFrame] = None,
        y_recal_df: Optional[pd.DataFrame] = None,
        groups_train: Optional[np.ndarray] = None,
    ) -> Dict[str, Union[Pipeline, CalibratedClassifierCV]]:
        """
        Executes the pipelines for all configured target outcomes.

        Parameters
        ----------
        X_train : pd.DataFrame
            The training feature set.
        y_train_df : pd.DataFrame
            The training targets DataFrame (can contain multiple outcome columns).
        X_recal : Optional[pd.DataFrame], default=None
            The recalibration feature set.
        y_recal_df : Optional[pd.DataFrame], default=None
            The recalibration targets DataFrame.
        groups_train : Optional[np.ndarray], default=None
            Group labels for the training set.

        Returns
        -------
        Dict[str, Union[Pipeline, CalibratedClassifierCV]]
            A dictionary mapping outcome names to their finalized, fitted models.

        """
        outcomes = self.orchestrator.config.data.outcomes

        for outcome in outcomes:
            # We assume y_train_df columns correspond to the requested outcomes
            y_train = y_train_df[outcome].to_numpy().ravel()

            y_recal = None
            if y_recal_df is not None and outcome in y_recal_df.columns:
                y_recal = y_recal_df[outcome].to_numpy().ravel()

            final_model = self.fit_outcome(
                outcome=outcome,
                X_train=X_train,
                y_train=y_train,
                X_recal=X_recal,
                y_recal=y_recal,
                groups_train=groups_train,
            )

            self.fitted_models[outcome] = final_model

        # Save final models dictionary
        self._save_final_models()

        return self.fitted_models
