"""
Main entry point module for the medpipe machine learning package.

Provides unified, high-level interfaces (`MedpipeClassifier` for binary
classification outcomes, `MedpipeRegressor` for regression outcomes)
orchestrating data preparation, model fitting, inference, and TRIPOD+AI
compliant evaluation.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from medpipe.pipeline.evaluator import (
    MedpipeClassifierEvaluator,
    MedpipeRegressorEvaluator,
)
from medpipe.pipeline.orchestrator import (
    DataSplits,
    FairnessSplits,
    MedpipeOrchestrator,
)
from medpipe.pipeline.runner import MedpipeClassifierRunner, MedpipeRegressorRunner
from medpipe.utils.config import MedpipeClassifierConfig, MedpipeRegressorConfig
from medpipe.utils.logger import get_console_logger
from medpipe.visualisation.displayer import (
    MedpipeClassifierDisplayer,
    MedpipeRegressorDisplayer,
)

if TYPE_CHECKING:

    import numpy.typing as npt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure, SubFigure
    from ordboost.distributions import ContinuousPredictiveDistribution
    from ordboost.mappers import BaseBinMapper
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.pipeline import Pipeline

    from medpipe.pipeline.estimator import DistributionalPipeline


class MedpipeClassifier:
    """
    User entry point and high-level pipeline runner for MedpipeClassifier.

    Coordinates the complete machine learning lifecycle, delegating data ingress
    and preprocessing setup to `MedpipeOrchestrator`, model cross-validation
    and fitting to `MedpipeClassifierRunner`, and prediction and TRIPOD+AI evaluation to
    `MedpipeClassifierEvaluator`.

    Parameters
    ----------
    config : str, Path, or MedpipeClassifierConfig
        Path to the TOML configuration file or an instantiated
        MedpipeClassifierConfig object.
    base_artifact_dir : str or Path, default="artifacts"
        Root directory where versioned execution run artifacts, logs, and
        models are stored.
    verbose : Union[bool, int, str, None], default=None
        Console verbosity setting configuration override.

    Attributes
    ----------
    _orchestrator : MedpipeOrchestrator
        Pipeline orchestrator instance driving data preparation and
        reproducibility artifacts.
    _runner : MedpipeClassifierRunner
        Pipeline execution engine responsible for model training and
        cross-validation loops.
    _evaluator : MedpipeClassifierEvaluator
        Pipeline evaluation engine computing point estimates and
        bootstrap confidence intervals.
    _displayer : MedpipeClassifierDisplayer
        Visualisation engine rendering and persisting evaluation figures.
    _logger : logging.Logger
        Centralized logger instance configured under `"medpipe"`.

    Methods
    -------
    fit(X_train, y_train, X_recal=None, y_recal=None, groups_train=None)
        Fit machine learning models across configured target outcomes via
        MedpipeClassifierRunner.
    predict(X, model=None, outcome=None)
        Predict class labels for input samples.
    predict_proba(X, model=None, outcome=None)
        Predict class probabilities for input samples.
    decision_function(X, model=None, outcome=None)
        Compute decision function confidence scores for input samples.
    evaluate(X, y, outcome=None, model=None, metrics=None, subgroup_specs=None,
    fairness_data=None, save_artifacts=True)
        Evaluate model performance with confidence intervals on full datasets
        and subgroups.
    plot_all(y_true, probas, outcome="default", n_bootstraps=None, save=None,
    show=None, **style_kwargs)
        Generate and persist all standard evaluation figures for
        a specific outcome.
    run(subgroup_specs=None, groups_train=None)
        Execute full end-to-end pipeline (data preparation, model fitting, and
        test evaluation).

    """

    def __init__(
        self,
        config: str | Path | MedpipeClassifierConfig,
        base_artifact_dir: str | Path = "artifacts",
        verbose_override: bool | int | str | None = None,
    ) -> None:
        self._logger = get_console_logger("medpipe")

        self._logger.info("Initialising MedpipeClassifier end-to-end pipeline.")

        self._orchestrator = MedpipeOrchestrator(
            config, base_artifact_dir, verbose_override
        )
        self.mp_config = self._orchestrator.config
        self._runner = MedpipeClassifierRunner(orchestrator=self._orchestrator)
        self._evaluator = MedpipeClassifierEvaluator(
            orchestrator=self._orchestrator,
            runner=self._runner,
        )
        self._displayer = MedpipeClassifierDisplayer(orchestrator=self._orchestrator)

        self._logger.info("MedpipeClassifier initialisation complete.")

    @property
    def models(self) -> dict[str, Pipeline | CalibratedClassifierCV]:
        """Return the fitted models.

        Returns
        -------
        fitted_models : Dict[str, Union[Pipeline, CalibratedClassifierCV]]
            Fitted models from the MedpipeClassifierRunner object.

        """
        return self._runner.fitted_models

    @property
    def is_fitted(self) -> bool:
        """Checks if the MedpipeClassifier is fitted by looking at
        MedpipeClassifierRunner.

        Returns
        -------
        fitted : bool
            True if the models have been fitted, False otherwise.

        """
        return self._runner.fitted_models != {}

    @property
    def run_dir(self) -> Path:
        """Returns the run_dir from the MedpipeOrchestrator object.

        Returns
        -------
        run_dir : Path
            Path to the run_dir.

        """
        return self._orchestrator.run_dir

    @property
    def data_split(self) -> DataSplits:
        """Access any of the data splits.

        Return
        ------
        DataSplits
            Any of the DataSplits attributes.

        """
        return self._orchestrator.splits

    @property
    def fairness_split(self) -> FairnessSplits | None:
        """Access fairness stratification columns aligned with each data
        split.

        Lets fairness analyses be run directly against the relevant
        columns (e.g. `pipeline.fairness_split.test["HOSPITAL"]") without
        re-extracting them from the raw dataset, and independently of
        whether those columns are also model predictors.

        Return
        ------
        FairnessSplits or None
            Any of the FairnessSplits attributes, or None if no
            `workflow.evaluation.fairness` configuration is set.

        """
        return self._orchestrator.fairness_splits

    @property
    def metrics(self) -> list[str]:
        """Returns metrics used during evaluation.

        Returns
        -------
        metrics : list[str]
            List of metrics used during evaluation.

        """
        return self._evaluator.metrics

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_recal: pd.DataFrame | None = None,
        y_recal: pd.DataFrame | None = None,
        groups_train: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """
        Fit machine learning models across configured target outcomes via
        MedpipeClassifierRunner.

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
            Group identifier array for group-based cross-validation splits,
            default=None.

        Returns
        -------
        fitted_models : dict of str to object
            Dictionary mapping outcome target names to their finalized fitted models.

        """
        self._logger.info("Starting model fitting across configured target outcomes.")
        fitted_models = self._runner.run(
            X_train=X_train,
            y_train_df=y_train,
            X_recal=X_recal,
            y_recal_df=y_recal,
            groups_train=groups_train,
        )
        self._logger.info("Completed model fitting for all target outcomes.")
        return fitted_models

    def predict(
        self,
        X: pd.DataFrame | npt.NDArray,
        model: Any | None = None,
        outcome: str | None = None,
    ) -> npt.NDArray:
        """
        Predict class labels for input samples.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Features dataset of shape (n_samples, n_features).
        model : object, optional
            Fitted model instance. If None, resolved via `outcome` or
            `runner.fitted_models`.
        outcome : str, optional
            Outcome name corresponding to an entry in `runner.fitted_models`.

        Returns
        -------
        y_pred : numpy.ndarray
            Predicted class labels of shape (n_samples,).

        """
        return self._evaluator.predict(X=X, model=model, outcome=outcome)

    def predict_proba(
        self,
        X: pd.DataFrame | npt.NDArray,
        model: Any | None = None,
        outcome: str | None = None,
    ) -> npt.NDArray:
        """
        Predict class probabilities for input samples.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Features dataset of shape (n_samples, n_features).
        model : object, optional
            Fitted model instance. If None, resolved via `outcome` or
            `runner.fitted_models`.
        outcome : str, optional
            Outcome name corresponding to an entry in `runner.fitted_models`.

        Returns
        -------
        y_proba : numpy.ndarray
            Predicted class probabilities of shape (n_samples, n_classes) or
            (n_samples,).

        """
        return self._evaluator.predict_proba(X=X, model=model, outcome=outcome)

    def decision_function(
        self,
        X: pd.DataFrame | npt.NDArray,
        model: Any | None = None,
        outcome: str | None = None,
    ) -> npt.NDArray:
        """
        Compute decision function confidence scores for input samples.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Features dataset of shape (n_samples, n_features).
        model : object, optional
            Fitted model instance. If None, resolved via `outcome` or
            `runner.fitted_models`.
        outcome : str, optional
            Outcome name corresponding to an entry in `runner.fitted_models`.

        Returns
        -------
        scores : numpy.ndarray
            Confidence scores or decision function values of shape (n_samples,).

        """
        return self._evaluator.decision_function(X=X, model=model, outcome=outcome)

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series | npt.NDArray,
        outcome: str | None = None,
        model: Any | None = None,
        metrics: list[str] | None = None,
        subgroup_specs: dict[str, str | Callable[[pd.DataFrame], pd.Series]]
        | None = None,
        fairness_data: pd.DataFrame | None = None,
        save_artifacts: bool = True,
    ) -> dict[str, Any]:
        """
        Evaluate model performance with confidence intervals on full datasets
        and subgroups.

        Parameters
        ----------
        X : pandas.DataFrame
            Feature dataset of shape (n_samples, n_features).
        y : pandas.Series or numpy.ndarray
            Ground truth target values of shape (n_samples,).
        outcome : str, optional
            Outcome key name. Used for logging, model resolution, and artifact
            saving.
        model : object, optional
            Explicit fitted model instance. If None, resolved from
            `runner.fitted_models`.
        metrics : list of str, optional
            List of metrics to evaluate. If None, defaults to
            `self._evaluator.metrics`.
        subgroup_specs : dict of str to (str or callable), optional
            Specifications for extracting demographic or clinical subgroups.
        fairness_data : pandas.DataFrame, optional
            Fairness stratification columns aligned with `X` (see
            `fairness_split`), used to resolve `subgroup_specs` for strata
            that are not themselves model predictors (e.g. "HOSPITAL").
            Defaults to `X` when not provided.
        save_artifacts : bool, default=True
            Whether to write evaluation summary results to disk via
            `ArtifactManager`.

        Returns
        -------
        results : dict of str to Any
            Nested dictionary containing outcome identifier, overall slice
            metrics, and subgroup performance.

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

        return self._evaluator.evaluate(
            X=X,
            y=y_eval,
            outcome=outcome,
            model=model,
            metrics=metrics,
            subgroup_specs=subgroup_specs,
            fairness_data=fairness_data,
            save_artifacts=save_artifacts,
        )

    def run(
        self,
        groups_train: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """
        Execute full end-to-end pipeline (data preparation, model fitting, and
        test evaluation).

        Automates data ingestion, split creation, model cross-validation and fitting,
        and test evaluation with TRIPOD+AI reporting across all target outcomes.
        The model fitting duration and total run duration are recorded at
        the debug log level.

        Parameters
        ----------
        groups_train : numpy.ndarray, optional
            Group labels for training samples if group-based cross-validation
            is configured.

        Returns
        -------
        pipeline_results : dict of str to Any
            Dictionary containing:
            - `"fitted_models"`: Dictionary mapping outcome target names
            to fitted model instances.
            - `"evaluations"`: Dictionary mapping outcome target names
            to evaluation results dictionaries.
            - `"plots"`: (Audit mode only) Nested dictionary mapping outcome
            target names to figure objects.

        """
        self._logger.info("Executing full MedpipeClassifier pipeline end-to-end.")
        run_start_time = time.perf_counter()

        run_mode = self.mp_config.meta.run_mode

        self._logger.debug(
            f"Executing full MedpipeClassifier pipeline in {run_mode} mode."
        )

        n_steps = 3
        if run_mode == "audit" or run_mode == "eval":
            n_steps = 4

        # 1. Prepare data splits via orchestrator
        data_kwargs = self.mp_config.data.kwargs  # Get extra data arguments
        self._logger.info(f"Step 1/{n_steps}: Ingesting and splitting dataset.")
        X_train, y_train, X_recal, y_recal, X_test, y_test, groups_train = (
            self._orchestrator.prepare_data(**data_kwargs)
        )

        # 2. Fit models via runner
        self._logger.info(f"Step 2/{n_steps}: Fitting outcome models.")
        fit_start_time = time.perf_counter()
        fitted_models = self.fit(
            X_train=X_train,
            y_train=y_train,
            X_recal=X_recal,
            y_recal=y_recal,
            groups_train=groups_train,
        )
        fit_duration = time.perf_counter() - fit_start_time
        self._logger.debug(f"Model fitting completed in {fit_duration:.2f} seconds.")

        # 3. Evaluate models on test set via evaluator
        self._logger.info(
            f"Step 3/{n_steps}: Evaluating models on holdout test set.",
        )
        evaluations: dict[str, Any] = {}
        plots: dict[str, dict[str, tuple[Figure | SubFigure, Axes]]] = {}
        outcomes = self._orchestrator.config.data.outcomes
        subgroup_specs = self._orchestrator.get_subgroup_specs()
        fairness_splits = self._orchestrator.fairness_splits
        fairness_data = fairness_splits.test if fairness_splits is not None else None

        for outcome in outcomes:
            y_test_outcome = y_test[outcome]
            evaluations[outcome] = self.evaluate(
                X=X_test,
                y=y_test_outcome.to_numpy(),
                outcome=outcome,
                subgroup_specs=subgroup_specs,
                fairness_data=fairness_data,
                save_artifacts=True,
            )

        if run_mode == "audit" or run_mode == "eval":
            self._logger.info(f"Step 4/{n_steps}: Plotting graphs.")
            for outcome in outcomes:
                y_true_outcome = y_test[outcome].to_numpy()
                probas_outcome = self.predict_proba(X=X_test, outcome=outcome)

                plots[outcome] = self.plot_all(
                    y_true=y_true_outcome,
                    probas=probas_outcome,
                    outcome=outcome,
                )
            # Generate cross-outcome strata heatmaps per metric
            self._logger.info("Generating subgroup strata heatmaps across outcomes.")
            strata_heatmaps = self._displayer.plot_all_heatmaps(
                evaluations=evaluations,
            )
            plots["strata_heatmaps"] = strata_heatmaps

        run_duration = time.perf_counter() - run_start_time
        self._logger.debug(
            f"Full pipeline run completed in {run_duration:.2f} seconds."
        )
        self._logger.info(
            "Full MedpipeClassifier pipeline execution finished successfully."
        )

        results: dict[str, Any] = {
            "fitted_models": fitted_models,
            "evaluations": evaluations,
        }
        if plots:
            results["plots"] = plots

        return results

    def plot_all(
        self,
        y_true: np.ndarray,
        probas: np.ndarray,
        outcome: str = "default",
        n_bootstraps: int | None = None,
        save: bool | None = None,
        show: bool | None = None,
        **style_kwargs: Any,
    ) -> dict[str, tuple[Figure | SubFigure, Axes]]:
        """Execute all core evaluation visualization routines for a given outcome.

        Generates and optionally persists the ROC curve, Precision-Recall curve,
        Probability Distribution histogram, Reliability Diagram, and Decision
        Curve Analysis (DCA).

        Parameters
        ----------
        y_true : numpy.ndarray
            Ground truth binary target labels of shape (n_samples,).
        probas : numpy.ndarray
            Predicted probabilities of shape (n_samples, 2) or (n_samples,).
        outcome : str, default="default"
            Outcome identifier used for figure titles and directory structuring.
        n_bootstraps : int, optional
            Number of bootstrap iterations for confidence interval estimation on ROC,
            PR, and reliability curves. If None, resolved from the display
            configuration.
        save : bool, optional
            Automatically save all generated plot artifacts to the run directory.
            If None, resolved from the display configuration.
        show : bool, optional
            Whether to display figures interactively before closing. If None,
            resolved from the display configuration.
        **style_kwargs : Any
            Additional style parameters forwarded to underlying drawing primitives.

        Returns
        -------
        plots : dict of str to tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes)
            Dictionary mapping plot keys
            ('roc', 'pr', 'distribution', 'reliability', 'dca')
            to their rendered (Figure, Axes) Matplotlib objects.

        """
        return self._displayer.plot_all(
            y_true=y_true,
            probas=probas,
            outcome=outcome,
            n_bootstraps=n_bootstraps,
            save=save,
            show=show,
            **style_kwargs,
        )

    @classmethod
    def load(cls, run_dir: str | Path) -> MedpipeClassifier:
        """Reconstruct a MedpipeClassifier instance from a run artifact directory.

        Parses the saved JSON configuration and restores serialized outcome
        model artifacts into the runner and evaluator engines.

        Parameters
        ----------
        run_dir : str or Path
            Directory path of a previously executed MedpipeClassifier run artifact.

        Returns
        -------
        pipe : MedpipeClassifier
            Re-instantiated MedpipeClassifier object ready for inference,
            evaluation, or visualization.

        Raises
        ------
        FileNotFoundError
            If the resolved configuration JSON file is missing from the run directory.

        """
        run_path = Path(run_dir)
        config_path = run_path / "env/resolved_config.json"
        models_dir = run_path / "models"

        if not config_path.exists():
            raise FileNotFoundError(
                "Cannot load MedpipeClassifier instance: Configuration JSON missing "
                f"in '{run_path}'"
            )

        # 1. Load JSON dict and instantiate MedpipeClassifierConfig
        import json

        import joblib

        with open(config_path, encoding="utf-8") as f:
            config_dict = json.load(f)

        mp_config = MedpipeClassifierConfig.model_validate(config_dict)

        # 2. Instantiate MedpipeClassifier with reconstructed MedpipeClassifierConfig
        new_run_path = run_path / "eval"
        pipe = cls(config=mp_config, base_artifact_dir=new_run_path)
        pipe._orchestrator.run_dir = new_run_path
        pipe._displayer.run_dir = new_run_path

        # 3. Restore serialized model binaries into runner and evaluator engines
        if models_dir.exists():
            project_name = pipe.mp_config.meta.project_name
            pipe._runner.fitted_models = joblib.load(
                models_dir / f"{project_name}_fitted.joblib"
            )
            pipe._evaluator.fitted_models = pipe._runner.fitted_models
        pipe._logger.info(f"Succesfully loaded MedpipeClassifier from {run_dir}")

        return pipe


class MedpipeRegressor:
    """
    User entry point and high-level pipeline runner for MedpipeRegressor.

    Coordinates the complete machine learning lifecycle for regression
    outcomes, delegating data ingress and preprocessing setup to
    `MedpipeOrchestrator`, model cross-validation and fitting to
    `MedpipeRegressorRunner`, and prediction and TRIPOD+AI evaluation to
    `MedpipeRegressorEvaluator`.

    Unlike `MedpipeClassifier`, there is no post-hoc recalibration step (not
    a meaningful concept for regression outcomes). Distributional diagnostic
    figures (coverage, sharpness, Winkler score, marginal calibration, PIT
    histogram) are available via `plot_all`, but only for models producing a
    predictive CDF through `predict_dist` (currently OrdBoost only); `run()`
    does not call it automatically yet.

    Parameters
    ----------
    config : str, Path, or MedpipeRegressorConfig
        Path to a TOML configuration file (parsed by `MedpipeOrchestrator`
        via `read_regressor_toml_configuration`) or an instantiated
        `MedpipeRegressorConfig` object.
    base_artifact_dir : str or Path, default="artifacts"
        Root directory where versioned execution run artifacts, logs, and
        models are stored.
    verbose : Union[bool, int, str, None], default=None
        Console verbosity setting configuration override.

    Attributes
    ----------
    _orchestrator : MedpipeOrchestrator
        Pipeline orchestrator instance driving data preparation and
        reproducibility artifacts.
    _runner : MedpipeRegressorRunner
        Pipeline execution engine responsible for model training and
        cross-validation loops.
    _evaluator : MedpipeRegressorEvaluator
        Pipeline evaluation engine computing point estimates and
        bootstrap confidence intervals.
    _displayer : MedpipeRegressorDisplayer
        Visualisation engine rendering and persisting distributional
        diagnostic figures.
    _logger : logging.Logger
        Centralized logger instance configured under `"medpipe"`.

    Methods
    -------
    fit(X_train, y_train, X_recal=None, y_recal=None, groups_train=None)
        Fit machine learning models across configured target outcomes via
        MedpipeRegressorRunner.
    predict(X, model=None, outcome=None)
        Predict continuous values for input samples.
    predict_dist(X, model=None, outcome=None)
        Predict the full predictive distribution for input samples.
    evaluate(X, y, outcome=None, model=None, metrics=None, subgroup_specs=None,
    fairness_data=None, save_artifacts=True)
        Evaluate model performance with confidence intervals on full datasets
        and subgroups.
    plot_all(y_true, dist, mapper=None, outcome="default", coverage_levels=None,
    n_bootstraps=None, save=None, show=None, **style_kwargs)
        Execute all core distributional diagnostic visualization routines
        for a given outcome (coverage, sharpness, Winkler, marginal
        calibration, PIT histogram).
    run(subgroup_specs=None, groups_train=None)
        Execute full end-to-end pipeline (data preparation, model fitting, and
        test evaluation).

    """

    def __init__(
        self,
        config: str | Path | MedpipeRegressorConfig,
        base_artifact_dir: str | Path = "artifacts",
        verbose_override: bool | int | str | None = None,
    ) -> None:
        self._logger = get_console_logger("medpipe")

        self._logger.info("Initialising MedpipeRegressor end-to-end pipeline.")

        self._orchestrator = MedpipeOrchestrator(
            config, base_artifact_dir, verbose_override, is_classifier=False
        )
        self.mp_config = self._orchestrator.config
        self._runner = MedpipeRegressorRunner(orchestrator=self._orchestrator)
        self._evaluator = MedpipeRegressorEvaluator(
            orchestrator=self._orchestrator,
            runner=self._runner,
        )
        self._displayer = MedpipeRegressorDisplayer(orchestrator=self._orchestrator)

        self._logger.info("MedpipeRegressor initialisation complete.")

    @property
    def models(self) -> dict[str, DistributionalPipeline]:
        """Return the fitted models.

        Returns
        -------
        fitted_models : Dict[str, DistributionalPipeline]
            Fitted models from the MedpipeRegressorRunner object.

        """
        return self._runner.fitted_models

    @property
    def is_fitted(self) -> bool:
        """Checks if the MedpipeRegressor is fitted by looking at
        MedpipeRegressorRunner.

        Returns
        -------
        fitted : bool
            True if the models have been fitted, False otherwise.

        """
        return self._runner.fitted_models != {}

    @property
    def run_dir(self) -> Path:
        """Returns the run_dir from the MedpipeOrchestrator object.

        Returns
        -------
        run_dir : Path
            Path to the run_dir.

        """
        return self._orchestrator.run_dir

    @property
    def data_split(self) -> DataSplits:
        """Access any of the data splits.

        Return
        ------
        DataSplits
            Any of the DataSplits attributes.

        """
        return self._orchestrator.splits

    @property
    def fairness_split(self) -> FairnessSplits | None:
        """Access fairness stratification columns aligned with each data
        split.

        Lets fairness analyses be run directly against the relevant
        columns (e.g. `pipeline.fairness_split.test["HOSPITAL"]") without
        re-extracting them from the raw dataset, and independently of
        whether those columns are also model predictors.

        Return
        ------
        FairnessSplits or None
            Any of the FairnessSplits attributes, or None if no
            `workflow.evaluation.fairness` configuration is set.

        """
        return self._orchestrator.fairness_splits

    @property
    def metrics(self) -> list[str]:
        """Returns metrics used during evaluation.

        Returns
        -------
        metrics : list[str]
            List of metrics used during evaluation.

        """
        return self._evaluator.metrics

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_recal: pd.DataFrame | None = None,
        y_recal: pd.DataFrame | None = None,
        groups_train: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """
        Fit machine learning models across configured target outcomes via
        MedpipeRegressorRunner.

        Parameters
        ----------
        X_train : pandas.DataFrame
            Training feature dataset.
        y_train : pandas.DataFrame
            Training outcome target labels (can contain multiple outcome columns).
        X_recal : pandas.DataFrame, optional
            Unused (no recalibration step for regression); accepted for
            interface symmetry with `MedpipeClassifier.fit`.
        y_recal : pandas.DataFrame, optional
            Unused (no recalibration step for regression); accepted for
            interface symmetry with `MedpipeClassifier.fit`.
        groups_train : numpy.ndarray, optional
            Group identifier array for group-based cross-validation splits,
            default=None.

        Returns
        -------
        fitted_models : dict of str to object
            Dictionary mapping outcome target names to their finalized fitted models.

        """
        self._logger.info("Starting model fitting across configured target outcomes.")
        fitted_models = self._runner.run(
            X_train=X_train,
            y_train_df=y_train,
            X_recal=X_recal,
            y_recal_df=y_recal,
            groups_train=groups_train,
        )
        self._logger.info("Completed model fitting for all target outcomes.")
        return fitted_models

    def predict(
        self,
        X: pd.DataFrame | npt.NDArray,
        model: Any | None = None,
        outcome: str | None = None,
    ) -> npt.NDArray:
        """
        Predict continuous values for input samples.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Features dataset of shape (n_samples, n_features).
        model : object, optional
            Fitted model instance. If None, resolved via `outcome` or
            `runner.fitted_models`.
        outcome : str, optional
            Outcome name corresponding to an entry in `runner.fitted_models`.

        Returns
        -------
        y_pred : numpy.ndarray
            Predicted continuous values of shape (n_samples,).

        """
        return self._evaluator.predict(X=X, model=model, outcome=outcome)

    def predict_dist(
        self,
        X: pd.DataFrame | npt.NDArray,
        model: Any | None = None,
        outcome: str | None = None,
    ) -> Any:
        """
        Predict the full predictive distribution for input samples.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Features dataset of shape (n_samples, n_features).
        model : object, optional
            Fitted model instance. If None, resolved via `outcome` or
            `runner.fitted_models`.
        outcome : str, optional
            Outcome name corresponding to an entry in `runner.fitted_models`.

        Returns
        -------
        Any
            The distribution object returned by the resolved model's
            `predict_dist` method.

        """
        return self._evaluator.predict_dist(X=X, model=model, outcome=outcome)

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series | npt.NDArray,
        outcome: str | None = None,
        model: Any | None = None,
        metrics: list[str] | None = None,
        subgroup_specs: dict[str, str | Callable[[pd.DataFrame], pd.Series]]
        | None = None,
        fairness_data: pd.DataFrame | None = None,
        save_artifacts: bool = True,
    ) -> dict[str, Any]:
        """
        Evaluate model performance with confidence intervals on full datasets
        and subgroups.

        Parameters
        ----------
        X : pandas.DataFrame
            Feature dataset of shape (n_samples, n_features).
        y : pandas.Series or numpy.ndarray
            Ground truth target values of shape (n_samples,).
        outcome : str, optional
            Outcome key name. Used for logging, model resolution, and artifact
            saving.
        model : object, optional
            Explicit fitted model instance. If None, resolved from
            `runner.fitted_models`.
        metrics : list of str, optional
            List of metrics to evaluate. If None, defaults to
            `self._evaluator.metrics`.
        subgroup_specs : dict of str to (str or callable), optional
            Specifications for extracting demographic or clinical subgroups.
        fairness_data : pandas.DataFrame, optional
            Fairness stratification columns aligned with `X` (see
            `fairness_split`), used to resolve `subgroup_specs` for strata
            that are not themselves model predictors (e.g. "HOSPITAL").
            Defaults to `X` when not provided.
        save_artifacts : bool, default=True
            Whether to write evaluation summary results to disk via
            `ArtifactManager`.

        Returns
        -------
        results : dict of str to Any
            Nested dictionary containing outcome identifier, overall slice
            metrics, and subgroup performance.

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

        return self._evaluator.evaluate(
            X=X,
            y=y_eval,
            outcome=outcome,
            model=model,
            metrics=metrics,
            subgroup_specs=subgroup_specs,
            fairness_data=fairness_data,
            save_artifacts=save_artifacts,
        )

    def run(
        self,
        groups_train: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """
        Execute full end-to-end pipeline (data preparation, model fitting, and
        test evaluation).

        Automates data ingestion, split creation, model cross-validation and fitting,
        and test evaluation with TRIPOD+AI reporting across all target outcomes.
        The model fitting duration and total run duration are recorded at
        the debug log level.

        Note: unlike `MedpipeClassifier.run`, this does not generate any
        plots regardless of `run_mode` - not every regression algorithm
        supports `predict_dist`, so distributional diagnostics are not
        auto-generated here yet. Call `plot_all` manually once fitted with a
        distribution-capable model (e.g. `OrdBoostRegressor`).

        Parameters
        ----------
        groups_train : numpy.ndarray, optional
            Group labels for training samples if group-based cross-validation
            is configured.

        Returns
        -------
        pipeline_results : dict of str to Any
            Dictionary containing:
            - `"fitted_models"`: Dictionary mapping outcome target names
            to fitted model instances.
            - `"evaluations"`: Dictionary mapping outcome target names
            to evaluation results dictionaries.

        """
        self._logger.info("Executing full MedpipeRegressor pipeline end-to-end.")
        run_start_time = time.perf_counter()

        run_mode = self.mp_config.meta.run_mode
        self._logger.debug(
            f"Executing full MedpipeRegressor pipeline in {run_mode} mode."
        )

        n_steps = 3

        # 1. Prepare data splits via orchestrator
        data_kwargs = self.mp_config.data.kwargs  # Get extra data arguments
        self._logger.info(f"Step 1/{n_steps}: Ingesting and splitting dataset.")
        X_train, y_train, X_recal, y_recal, X_test, y_test, groups_train = (
            self._orchestrator.prepare_data(**data_kwargs)
        )

        # 2. Fit models via runner
        self._logger.info(f"Step 2/{n_steps}: Fitting outcome models.")
        fit_start_time = time.perf_counter()
        fitted_models = self.fit(
            X_train=X_train,
            y_train=y_train,
            X_recal=X_recal,
            y_recal=y_recal,
            groups_train=groups_train,
        )
        fit_duration = time.perf_counter() - fit_start_time
        self._logger.debug(f"Model fitting completed in {fit_duration:.2f} seconds.")

        # 3. Evaluate models on test set via evaluator
        self._logger.info(
            f"Step 3/{n_steps}: Evaluating models on holdout test set.",
        )
        evaluations: dict[str, Any] = {}
        outcomes = self._orchestrator.config.data.outcomes
        subgroup_specs = self._orchestrator.get_subgroup_specs()
        fairness_splits = self._orchestrator.fairness_splits
        fairness_data = fairness_splits.test if fairness_splits is not None else None

        for outcome in outcomes:
            y_test_outcome = y_test[outcome]
            evaluations[outcome] = self.evaluate(
                X=X_test,
                y=y_test_outcome.to_numpy(),
                outcome=outcome,
                subgroup_specs=subgroup_specs,
                fairness_data=fairness_data,
                save_artifacts=True,
            )

        run_duration = time.perf_counter() - run_start_time
        self._logger.debug(
            f"Full pipeline run completed in {run_duration:.2f} seconds."
        )
        self._logger.info(
            "Full MedpipeRegressor pipeline execution finished successfully."
        )

        return {
            "fitted_models": fitted_models,
            "evaluations": evaluations,
        }

    def plot_all(
        self,
        y_true: npt.NDArray,
        dist: ContinuousPredictiveDistribution,
        mapper: BaseBinMapper | None = None,
        outcome: str = "default",
        coverage_levels: npt.NDArray | None = None,
        n_bootstraps: int | None = None,
        save: bool | None = None,
        show: bool | None = None,
        **style_kwargs: Any,
    ) -> dict[str, tuple[Figure | SubFigure, Axes]]:
        """
        Execute all core distributional diagnostic visualization routines
        for a given outcome.

        Generates and optionally persists the coverage reliability curve,
        sharpness curve, Winkler score curve, marginal calibration curve,
        and PIT histogram. Only supported for models that produce a
        predictive CDF via `predict_dist` (currently OrdBoost only); a
        `mapper` (e.g. the fitted model's `mapper_` attribute) is required
        for the PIT histogram.

        Not called automatically by `run()` yet, since not every regression
        algorithm supports `predict_dist` — call this manually once you know
        the fitted model is distribution-capable (e.g. `OrdBoostRegressor`).

        Parameters
        ----------
        y_true : numpy.ndarray
            Ground truth continuous target values of shape (n_samples,).
        dist : ContinuousPredictiveDistribution
            Predictive distribution for the same samples, e.g. from
            `self.predict_dist(X, outcome=outcome)`.
        mapper : BaseBinMapper, optional
            The fitted OrdBoost model's bin mapper (e.g.
            `self.models[outcome].named_steps["regressor"].mapper_`),
            required for the PIT histogram.
        outcome : str, default="default"
            Outcome identifier used for figure titles and output folder structuring.
        coverage_levels : numpy.ndarray, optional
            Nominal central-interval coverage levels in percent, used for the
            coverage, sharpness, and Winkler score curves. Defaults to
            `np.arange(10, 100, 10)`.
        n_bootstraps : int, optional
            Number of bootstrap iterations for coverage, sharpness, and
            Winkler score curves. If None, resolved from the display
            configuration.
        save : bool, optional
            Automatically save all generated plot artifacts to the run directory.
            If None, resolved from the display configuration.
        show : bool, optional
            Whether to display figures interactively before closing. If None,
            resolved from the display configuration.
        **style_kwargs : Any
            Additional style parameters forwarded to underlying drawing primitives.

        Returns
        -------
        plots : dict of str to tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes)
            Dictionary mapping plot keys
            ('coverage', 'sharpness', 'winkler', 'marginal_calibration',
            'pit_histogram') to their rendered (Figure, Axes) Matplotlib objects.

        """
        return self._displayer.plot_all(
            y_true=y_true,
            dist=dist,
            mapper=mapper,
            outcome=outcome,
            coverage_levels=coverage_levels,
            n_bootstraps=n_bootstraps,
            save=save,
            show=show,
            **style_kwargs,
        )

    @classmethod
    def load(cls, run_dir: str | Path) -> MedpipeRegressor:
        """Reconstruct a MedpipeRegressor instance from a run artifact directory.

        Parses the saved JSON configuration and restores serialized outcome
        model artifacts into the runner and evaluator engines.

        Parameters
        ----------
        run_dir : str or Path
            Directory path of a previously executed MedpipeRegressor run artifact.

        Returns
        -------
        pipe : MedpipeRegressor
            Re-instantiated MedpipeRegressor object ready for inference,
            evaluation, or visualization.

        Raises
        ------
        FileNotFoundError
            If the resolved configuration JSON file is missing from the run directory.

        """
        run_path = Path(run_dir)
        config_path = run_path / "env/resolved_config.json"
        models_dir = run_path / "models"

        if not config_path.exists():
            raise FileNotFoundError(
                "Cannot load MedpipeRegressor instance: Configuration JSON missing "
                f"in '{run_path}'"
            )

        # 1. Load JSON dict and instantiate MedpipeRegressorConfig
        import json

        import joblib

        with open(config_path, encoding="utf-8") as f:
            config_dict = json.load(f)

        mp_config = MedpipeRegressorConfig.model_validate(config_dict)

        # 2. Instantiate MedpipeRegressor with reconstructed MedpipeRegressorConfig
        new_run_path = run_path / "eval"
        pipe = cls(config=mp_config, base_artifact_dir=new_run_path)
        pipe._orchestrator.run_dir = new_run_path

        # 3. Restore serialized model binaries into runner and evaluator engines
        if models_dir.exists():
            project_name = pipe.mp_config.meta.project_name
            pipe._runner.fitted_models = joblib.load(
                models_dir / f"{project_name}_fitted.joblib"
            )
            pipe._evaluator.fitted_models = pipe._runner.fitted_models
        pipe._logger.info(f"Succesfully loaded MedpipeRegressor from {run_dir}")

        return pipe
