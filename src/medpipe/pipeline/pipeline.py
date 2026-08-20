"""
Main entry point module for the Medpipe machine learning package.

Provides a unified, high-level interface (`Medpipe`) orchestrating data preparation,
model fitting, inference, and TRIPOD+AI compliant evaluation.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from medpipe.pipeline.evaluator import MedpipeEvaluator
from medpipe.pipeline.orchestrator import DataSplits, MedpipeOrchestrator
from medpipe.pipeline.runner import MedpipeRunner
from medpipe.utils.config import MedpipeConfig
from medpipe.utils.logger import get_console_logger
from medpipe.visualisation.displayer import MedpipeDisplayer

if TYPE_CHECKING:

    import numpy.typing as npt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure, SubFigure
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.pipeline import Pipeline


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
    verbose : Union[bool, int, str, None], default=None
        Console verbosity setting configuration override.

    Attributes
    ----------
    orchestrator : MedpipeOrchestrator
        Pipeline orchestrator instance driving data preparation and
        reproducibility artifacts.
    runner : MedpipeRunner
        Pipeline execution engine responsible for model training and
        cross-validation loops.
    evaluator : MedpipeEvaluator
        Pipeline evaluation engine computing point estimates and
        bootstrap confidence intervals.
    displayer : MedpipeDisplayer
        Visualisation engine rendering and persisting evaluation figures.
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
    evaluate(X, y, outcome=None, model=None, metrics=None, subgroup_specs=None,
    save_artifacts=True)
        Evaluate model performance with confidence intervals on full datasets
        and subgroups.
    plot_all(y_true, probas, outcome="default", n_bootstraps=1000, save=True,
    show=False, **style_kwargs)
        Generate and persist all standard evaluation figures for
        a specific outcome.
    run(subgroup_specs=None, groups_train=None)
        Execute full end-to-end pipeline (data preparation, model fitting, and test evaluation).

    """

    def __init__(
        self,
        config: Union[str, Path, MedpipeConfig],
        base_artifact_dir: Union[str, Path] = "artifacts",
        verbose_override: Union[bool, int, str, None] = None,
    ) -> None:
        self.logger = get_console_logger("medpipe")

        self.logger.info("Initialising Medpipe end-to-end pipeline.")

        self.orchestrator = MedpipeOrchestrator(
            config, base_artifact_dir, verbose_override
        )
        self.mp_config = self.orchestrator.config
        self.runner = MedpipeRunner(orchestrator=self.orchestrator)
        self.evaluator = MedpipeEvaluator(
            orchestrator=self.orchestrator,
            runner=self.runner,
        )
        self.displayer = MedpipeDisplayer(orchestrator=self.orchestrator)

        self.logger.info("Medpipe initialisation complete.")

    @property
    def models(self) -> Dict[str, Union[Pipeline, CalibratedClassifierCV]]:
        """Return the fitted models.

        Returns
        -------
        fitted_models : Dict[str, Union[Pipeline, CalibratedClassifierCV]]
            Fitted models from the MedpipeRunner object.

        """
        return self.runner.fitted_models

    @property
    def is_fitted(self) -> bool:
        """Checks if the Medpipe is fitted by looking at MedpipeRunner.

        Returns
        -------
        fitted : bool
            True if the models have been fitted, False otherwise.

        """
        return self.runner.fitted_models != {}

    @property
    def run_dir(self) -> Path:
        """Returns the run_dir from the MedpipeOrchestrator object.

        Returns
        -------
        run_dir : Path
            Path to the run_dir.

        """
        return self.orchestrator.run_dir

    @property
    def data_split(self) -> DataSplits:
        """Access any of the data splits.

        Return
        ------
        DataSplits
            Any of the DataSplits attributes.

        """
        return self.orchestrator.splits

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
            - `"fitted_models"`: Dictionary mapping outcome target names
            to fitted model instances.
            - `"evaluations"`: Dictionary mapping outcome target names
            to evaluation results dictionaries.
            - `"plots"`: (Audit mode only) Nested dictionary mapping outcome
            target names to figure objects.

        """
        self.logger.info("Executing full Medpipe pipeline end-to-end.")

        run_mode = self.mp_config.meta.run_mode

        self.logger.debug(f"Executing full Medpipe pipeline in {run_mode} mode.")

        n_steps = 3
        if run_mode == "audit" or run_mode == "eval":
            n_steps = 4

        # 1. Prepare data splits via orchestrator
        data_kwargs = self.mp_config.data.kwargs  # Get extra data arguments
        self.logger.info(f"Step 1/{n_steps}: Ingesting and splitting dataset.")
        X_train, y_train, X_recal, y_recal, X_test, y_test, groups_train = (
            self.orchestrator.prepare_data(**data_kwargs)
        )

        # 2. Fit models via runner
        self.logger.info(f"Step 2/{n_steps}: Fitting outcome models.")
        fitted_models = self.fit(
            X_train=X_train,
            y_train=y_train,
            X_recal=X_recal,
            y_recal=y_recal,
            groups_train=groups_train,
        )

        # 3. Evaluate models on test set via evaluator
        self.logger.info(
            f"Step 3/{n_steps}: Evaluating models on holdout test set.",
        )
        evaluations: Dict[str, Any] = {}
        plots: Dict[str, Dict[str, Tuple[Figure | SubFigure, Axes]]] = {}
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

        if run_mode == "audit" or run_mode == "eval":
            self.logger.info(f"Step 4/{n_steps}: Plotting graphs.")
            for outcome in outcomes:
                y_true_outcome = y_test[outcome].to_numpy()
                probas_outcome = self.predict_proba(X=X_test, outcome=outcome)

                plots[outcome] = self.plot_all(
                    y_true=y_true_outcome,
                    probas=probas_outcome,
                    outcome=outcome,
                )
            # Generate cross-outcome strata heatmaps per metric
            self.logger.info("Generating subgroup strata heatmaps across outcomes.")
            strata_heatmaps = self.displayer.plot_all_heatmaps(
                evaluations=evaluations,
            )
            plots["strata_heatmaps"] = strata_heatmaps

        self.logger.info("Full Medpipe pipeline execution finished successfully.")

        results: Dict[str, Any] = {
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
        n_bootstraps: int = 1000,
        save: bool = True,
        show: bool = False,
        **style_kwargs: Any,
    ) -> dict[str, Tuple[Figure | SubFigure, Axes]]:
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
        n_bootstraps : int, default=1000
            Number of bootstrap iterations for confidence interval estimation on ROC,
            PR, and reliability curves.
        save : bool, default=True
            Automatically save all generated plot artifacts to the run directory.
        show : bool, default=False
            Whether to display figures interactively before closing.
        **style_kwargs : Any
            Additional style parameters forwarded to underlying drawing primitives.

        Returns
        -------
        plots : dict of str to tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes)
            Dictionary mapping plot keys ('roc', 'pr', 'distribution', 'reliability', 'dca')
            to their rendered (Figure, Axes) Matplotlib objects.

        """
        return self.displayer.plot_all(
            y_true=y_true,
            probas=probas,
            outcome=outcome,
            n_bootstraps=n_bootstraps,
            save=save,
            show=show,
            **style_kwargs,
        )

    @classmethod
    def load(cls, run_dir: Union[str, Path]) -> Medpipe:
        """Reconstruct a Medpipe instance from a run artifact directory.

        Parses the saved JSON configuration and restores serialized outcome model artifacts
        into the runner engine.

        Parameters
        ----------
        run_dir : str or Path
            Directory path of a previously executed Medpipe run artifact.

        Returns
        -------
        pipe : Medpipe
            Re-instantiated Medpipe object ready for inference,
            evaluation, or visualization.

        Raises
        ------
        FileNotFoundError
            If the resolved configuration JSON file is missing from the run directory.

        """
        run_path = Path(run_dir)
        config_path = run_path / "resolved_config.json"
        models_dir = run_path / "models"

        if not config_path.exists():
            # Fallback check if named resolved_config.json
            config_path = run_path / "resolved_config.json"
            if not config_path.exists():
                raise FileNotFoundError(
                    "Cannot load Medpipe instance: Configuration JSON missing "
                    f"in '{run_path}'"
                )

        # 1. Load JSON dict and instantiate MedpipeConfig
        import json

        import joblib

        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)

        mp_config = MedpipeConfig.model_validate(config_dict)

        # 2. Instantiate Medpipe with reconstructed MedpipeConfig
        pipe = cls(config=mp_config, base_artifact_dir=run_path.parent)
        pipe.orchestrator.run_dir = run_path
        pipe.displayer.run_dir = run_path

        # 3. Restore serialized model binaries into runner engine
        if models_dir.exists():
            project_name = pipe.mp_config.meta.project_name
            pipe.runner.fitted_models = joblib.load(
                models_dir / f"{project_name}_fitted.joblib"
            )
        pipe.logger.info(f"Succesfully loaded Medpipe from {run_dir}")

        return pipe
