"""
Configuration utilities module.

This module provides configuration schemas.

"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# ==============================================================================
# CUSTOM VERBOSITY TYPES
# ==============================================================================
VerbosityMode = Literal[
    "quiet",
    "compact",
    "progress",
    "info",
    "detailed",
    "debug",
    "warning",
]
VerbosityInt = Literal[0, 1, 2, 3]

VerboseType = VerbosityMode | bool | VerbosityInt


# ==============================================================================
# SHARED VALIDATION HELPERS
# ==============================================================================
def _validate_metrics_registered(metrics: list[str]) -> None:
    """Validate that all given metric names are registered in MetricRegistry.

    Parameters
    ----------
    metrics : list of str
        Candidate metric identifiers.

    Raises
    ------
    ValueError
        If any entry in `metrics` is not a registered metric.

    """
    from medpipe.metrics.registry import MetricRegistry

    # Query the live registry (not the `METRICS` snapshot taken at import
    # time) so metrics registered by users afterwards are accepted.
    available = MetricRegistry.list_registered()
    for metric in metrics:
        if metric not in available:
            expr = (
                f"{metric} was not found in available metric "
                f"list. Available metrics are {available}"
            )
            raise ValueError(expr)


def _validate_metrics_for_task(
    metrics: list[str], task: Literal["classification", "regression"]
) -> None:
    """Validate that the metrics can be computed for the type of outcome.

    A metric that needs class probabilities, decision scores, or class labels
    (a `predict_proba`, `decision_function`, or thresholded `predict` response)
    cannot be computed on a continuous outcome, and a metric that needs a
    predictive distribution (`predict_dist`) cannot be computed on a class
    outcome. The other metrics, such as `rmse` and `mae`, are accepted for both.

    Parameters
    ----------
    metrics : list of str
        Metric identifiers, already known to be registered.
    task : {"classification", "regression"}
        Type of outcome the pipeline models.

    Raises
    ------
    ValueError
        If a metric cannot be computed for the type of outcome.

    """
    from medpipe.metrics.registry import MetricRegistry

    for name in metrics:
        spec = MetricRegistry.get(name)
        methods = (
            (spec.response_method,)
            if isinstance(spec.response_method, str)
            else tuple(spec.response_method)
        )
        if task == "classification" and "predict_dist" in methods:
            raise ValueError(
                f"The metric '{name}' needs a predictive distribution and cannot "
                "be computed for classification outcomes."
            )
        if task == "regression" and (
            spec.needs_threshold
            or "predict_proba" in methods
            or "decision_function" in methods
        ):
            raise ValueError(
                f"The metric '{name}' is a classification metric and cannot be "
                "computed for regression outcomes."
            )


def _metric_response_methods(name: str) -> tuple[str, ...]:
    """Return the response methods a registered metric needs.

    Parameters
    ----------
    name : str
        Identifier of a registered metric.

    Returns
    -------
    tuple of str
        The response methods of the metric.

    """
    from medpipe.metrics.registry import MetricRegistry

    method = MetricRegistry.get(name).response_method
    return (method,) if isinstance(method, str) else tuple(method)


def _validate_metrics_need_distribution(resolved_algorithms: dict[str, str]) -> None:
    """Validate that models can provide the distribution a metric needs.

    Parameters
    ----------
    resolved_algorithms : dict of str to str
        Mapping of outcome name to the algorithm name configured for it.

    Raises
    ------
    ValueError
        If the algorithm of an outcome has neither a `predict_dist` nor a
        `pred_dist` method.

    """
    from medpipe.models.registry import ModelRegistry

    for outcome, algorithm in resolved_algorithms.items():
        model_class = ModelRegistry.get(algorithm)
        if not (
            hasattr(model_class, "predict_dist") or hasattr(model_class, "pred_dist")
        ):
            raise ValueError(
                f"A configured metric needs a predictive distribution, but the "
                f"algorithm '{algorithm}' of outcome '{outcome}' cannot predict "
                "one."
            )


def _validate_algorithms_registered(algorithms: dict[str, str]) -> None:
    """Validate that all given algorithm names can be resolved by ModelRegistry.

    Parameters
    ----------
    algorithms : dict of str to str
        Mapping of outcome name to the algorithm name configured for it.

    Raises
    ------
    ValueError
        If an algorithm is neither registered nor found in the fallback
        modules of `ModelRegistry`.

    """
    from medpipe.models.registry import ModelRegistry

    for outcome, algorithm in algorithms.items():
        try:
            ModelRegistry.get(algorithm)
        except ValueError as err:
            raise ValueError(
                f"Algorithm for outcome '{outcome}' could not be resolved: {err}"
            ) from err


def _cascade_model_setup(
    base_setup: dict[str, Any], override: dict[str, Any]
) -> dict[str, Any]:
    """Apply an outcome override on top of the default model setup.

    Parameters
    ----------
    base_setup : dict of str to Any
        Dump of the default model setup. It is modified in place.
    override : dict of str to Any
        Dump of the outcome override, containing only the fields that were
        explicitly set.

    Returns
    -------
    dict of str to Any
        The resolved model setup.

    Raises
    ------
    ValueError
        If the override introduces a recalibration for a model without one
        and does not set `recalibrate`.

    """
    # 1. Algorithm & hyperparameters
    algorithm = override.get("algorithm")
    if algorithm is not None and algorithm != base_setup["algorithm"]:
        base_setup["algorithm"] = algorithm
        # Algorithm changed: replace hyperparameters entirely to prevent
        # collisions
        base_setup["hyperparameters"] = override.get("hyperparameters") or {}
    elif override.get("hyperparameters"):
        # Same (or omitted) algorithm: merge hyperparameters
        base_setup["hyperparameters"].update(override["hyperparameters"])

    # 2. Recalibration
    if "recalibration" not in override:
        return base_setup

    override_recal = override["recalibration"]
    base_recal = base_setup.get("recalibration")

    if override_recal is None:
        # Explicitly disable recalibration for this outcome
        base_setup["recalibration"] = None
    elif base_recal is None:
        if "recalibrate" not in override_recal:
            raise ValueError(
                "'recalibrate' must be set in an outcome override that adds "
                "a recalibration to a default model without one"
            )
        base_setup["recalibration"] = {
            "recalibrate": override_recal["recalibrate"],
            "method": override_recal.get("method", "isotonic"),
            "hyperparameters": override_recal.get("hyperparameters") or {},
        }
    else:
        method = override_recal.get("method")
        if method is not None and method != base_recal["method"]:
            # Method changed: replace hyperparameters entirely
            base_recal["method"] = method
            base_recal["hyperparameters"] = override_recal.get("hyperparameters") or {}
        elif override_recal.get("hyperparameters"):
            base_recal["hyperparameters"].update(override_recal["hyperparameters"])
        if "recalibrate" in override_recal:
            base_recal["recalibrate"] = override_recal["recalibrate"]

    return base_setup


def _validate_preprocessing_columns(
    preprocessing: PreprocessingConfig | None, predictors: list[str]
) -> None:
    """Validate that preprocessing operations only act on predictors.

    Parameters
    ----------
    preprocessing : PreprocessingConfig or None
        Preprocessing configuration of the workflow.
    predictors : list of str
        Predictor columns declared in the data configuration.

    Raises
    ------
    ValueError
        If an operation refers to a column that is not a predictor.

    """
    if not preprocessing or not preprocessing.preprocess:
        return

    for operation in preprocessing.operations or []:
        unknown = [col for col in operation.columns if col not in predictors]
        if unknown:
            raise ValueError(
                f"Preprocessing operation '{operation.name}' refers to columns "
                f"that are not in data.predictors: {unknown}"
            )


def _validate_theme_keys(theme: dict[str, Any] | None) -> None:
    """Validate that theme parameters are attributes of `MedpipeTheme`.

    Parameters
    ----------
    theme : dict of str to Any or None
        Candidate theme parameters.

    Raises
    ------
    ValueError
        If a key is not an attribute of `MedpipeTheme`.

    """
    if not theme:
        return

    from dataclasses import fields

    from medpipe.visualisation.themes import MedpipeTheme

    valid = {f.name for f in fields(MedpipeTheme)}
    unknown = sorted(set(theme) - valid)
    if unknown:
        raise ValueError(
            f"Unknown theme parameters {unknown}. Valid parameters are {sorted(valid)}"
        )


def _validate_plot_type_keys(
    v: dict[str, Any], valid_plots: set[str]
) -> dict[str, Any]:
    """Ensure plot override identifiers correspond to a set of valid plot
    types.

    Parameters
    ----------
    v : dict of str to Any
        Candidate `overrides` or `outcome_overrides` mapping.
    valid_plots : set of str
        Set of recognized plot type identifiers (case-insensitive).

    Returns
    -------
    dict of str to Any
        The validated mapping.

    Raises
    ------
    ValueError
        If any plot type key is not a recognized plot type.

    """
    for key, val in v.items():
        if isinstance(val, dict) and any(
            isinstance(sub_v, dict) for sub_v in val.values()
        ):
            # Outcome overrides dictionary: outcome_name -> {plot_type: params}
            for plot_key in val:
                if plot_key.lower() not in valid_plots:
                    raise ValueError(
                        f"Unknown plot override type '{plot_key}'. "
                        f"Valid plot types are: {sorted(valid_plots)}"
                    )
        else:
            # Plot type overrides dictionary: plot_type -> params
            if key.lower() not in valid_plots:
                raise ValueError(
                    f"Unknown plot override type '{key}'. "
                    f"Valid plot types are: {sorted(valid_plots)}"
                )
    return v


def _validate_recalibration_split_strategy(
    test_split: SplitTestConfig,
    recalibration_split: SplitRecalibrationConfig | None,
) -> None:
    """Validate that recalibration and test split have the same strategy.

    Parameters
    ----------
    test_split : SplitTestConfig
        Configuration for the train/test split.
    recalibration_split : SplitRecalibrationConfig or None
        Configuration for the recalibration split, if any.

    Raises
    ------
    ValueError
        If `recalibration_split` is set and its `strategy` differs from
        `test_split.strategy`.

    """
    if recalibration_split and recalibration_split.strategy != test_split.strategy:
        raise ValueError("Recalibration and test strategies should match")


def _validate_recalibration_split_group_column(
    test_split: SplitTestConfig,
    recalibration_split: SplitRecalibrationConfig | None,
) -> None:
    """Validate that recalibration and test split have the same group column.

    Parameters
    ----------
    test_split : SplitTestConfig
        Configuration for the train/test split.
    recalibration_split : SplitRecalibrationConfig or None
        Configuration for the recalibration split, if any.

    Raises
    ------
    ValueError
        If both splits use the "group" strategy but their group columns
        differ.

    """
    if (
        recalibration_split
        and recalibration_split.strategy == "group"
        and test_split.strategy == "group"
        and recalibration_split.group_column != test_split.group_column
    ):
        raise ValueError("Recalibration and test group columns should match")


def _validate_recalibration_split_values(
    test_split: SplitTestConfig,
    recalibration_split: SplitRecalibrationConfig | None,
) -> None:
    """Validate that recalibration and test split have different values.

    Parameters
    ----------
    test_split : SplitTestConfig
        Configuration for the train/test split.
    recalibration_split : SplitRecalibrationConfig or None
        Configuration for the recalibration split, if any.

    Raises
    ------
    ValueError
        If both splits use the "group" strategy and share any group value.

    """
    if (
        recalibration_split
        and recalibration_split.values is not None
        and recalibration_split.strategy == "group"
        and test_split.strategy == "group"
    ):
        for value in recalibration_split.values:
            if value in test_split.values:  # type: ignore
                raise ValueError("Recalibration and test values should be different")


# ==============================================================================
# CONFIGURATION SCHEMA (pydantic)
# ==============================================================================
# --- TOP-LEVEL MASTER SCHEMAS ---
class MetaConfig(BaseModel):
    """The master schema for the meta section of the configuration file.

    Attributes
    ----------
    project_name : str
        Name of the project, used for labeling artifacts and logs.
    run_mode : {"fast", "eval", "cv", "audit"}, default="audit"
        Execution mode controlling which pipeline stages run.
    verbose : VerboseType, default="compact"
        Console logging verbosity level.

    """

    project_name: str
    run_mode: Literal["fast", "eval", "cv", "audit"] = "audit"
    verbose: VerboseType = Field(
        default="compact",
        description=(
            "Console logging verbosity: 'quiet' (0), 'compact' (1), "
            "'info' (2), 'debug' (3)."
        ),
    )
    model_config = {"extra": "forbid"}

    @field_validator("project_name")
    @classmethod
    def validate_project_name(cls, name: str) -> str:
        """Validate that project name is not empty.

        Parameters
        ----------
        name : str
            Candidate project name.

        Returns
        -------
        str
            The validated project name.

        Raises
        ------
        ValueError
            If `name` is an empty string.

        """
        if not name:
            raise ValueError("Project name should not be an empty string.")

        return name


# --- DATA SCHEMAS ---
class DataConfig(BaseModel):
    """The master schema for the data section of the configuration file.

    Attributes
    ----------
    path : str
        Path to the dataset file to load.
    predictors : list of str
        Column names used as model predictors.
    outcomes : list of str
        Column names used as target outcomes.
    kwargs : dict of str to Any, default={}
        Additional keyword arguments forwarded to the data loader.

    """

    path: str
    predictors: list[str]
    outcomes: list[str]
    kwargs: dict[str, Any] = Field(default_factory=dict)
    model_config = {"extra": "forbid"}

    @field_validator("path")
    @classmethod
    def validate_path(cls, file: str) -> str:
        """Validate that path points to a file with a suffix.

        Parameters
        ----------
        file : str
            Candidate file path.

        Returns
        -------
        str
            The validated file path.

        Raises
        ------
        ValueError
            If `file` has no file extension.

        """
        data_path = Path(file)
        suffix = data_path.suffix
        if suffix == "":
            raise ValueError("path should be a file, but got no suffix")
        return file

    @model_validator(mode="after")
    def check_for_target_leakage(self) -> DataConfig:
        """Validate that no column appears in both predictors and outcomes.

        Returns
        -------
        DataConfig
            The validated configuration instance.

        Raises
        ------
        ValueError
            If any column is listed in both `predictors` and `outcomes`.

        """
        # Check if any outcome intersects with the predictor list
        overlap = set(self.outcomes).intersection(set(self.predictors))
        if overlap:
            raise ValueError(
                "Overlap between predictors and outcomes which will break "
                f"model validity: {list(overlap)}"
            )
        return self


# --- WORKFLOW SCHEMAS ---
class PreprocessOperationConfig(BaseModel):
    """Configuration for a single preprocessing operation.

    Attributes
    ----------
    name : str
        Name of the registered preprocessing operation class to apply.
    columns : list of str
        Columns the operation is applied to.

    """

    name: str  # Matches the exact class name
    columns: list[str]  # The specific columns this transformer applies to
    model_config = {"extra": "allow"}

    @field_validator("columns")
    @classmethod
    def validate_columns(cls, columns: list[str]) -> list[str]:
        """Validate that columns are not empty.

        Parameters
        ----------
        columns : list of str
            Candidate columns the operation applies to.

        Returns
        -------
        list of str
            The validated columns.

        Raises
        ------
        ValueError
            If `columns` is an empty list.

        """
        if not columns:
            raise ValueError("Columns cannot be an empty list")
        return columns


class PreprocessingConfig(BaseModel):
    """Configuration controlling whether and how preprocessing is applied.

    Attributes
    ----------
    preprocess : bool or None, default=None
        Whether preprocessing should be applied.
    operations : list of PreprocessOperationConfig or None, default=None
        Ordered preprocessing operations to run when `preprocess` is True.

    """

    preprocess: bool | None = None
    operations: list[PreprocessOperationConfig] | None = None
    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def validate_operations(self) -> PreprocessingConfig:
        """Validate that operations are specified when preprocessing is
        enabled.

        Returns
        -------
        PreprocessingConfig
            The validated configuration instance.

        Raises
        ------
        ValueError
            If `preprocess` is True but `operations` is empty or None.

        """
        if self.preprocess and not self.operations:
            raise ValueError("Operations must be specified if preprocess is True")
        return self


class SplitTestConfig(BaseModel):
    """Configuration for the train/test split strategy.

    Attributes
    ----------
    strategy : {"random", "group"}, default="random"
        Strategy used to split the dataset into train and test sets.
    group_column : str or None, default=None
        Column identifying groups when `strategy` is "group".
    values : list of (str or int), or None, default=None
        Group values assigned to the test split when `strategy` is "group".
    test_size : float or None, default=None
        Fraction of the dataset reserved for testing when `strategy` is
        "random".

    """

    strategy: Literal["random", "group"] = "random"
    group_column: str | None = None
    values: list[str | int] | None = None
    test_size: float | None = Field(default=None, gt=0.0, lt=1.0)
    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def validate_strategy(self) -> SplitTestConfig:
        """Validate that the chosen split strategy has its required fields
        set.

        Returns
        -------
        SplitTestConfig
            The validated configuration instance.

        Raises
        ------
        ValueError
            If `strategy` is "random" and `test_size` is not set, or if
            `strategy` is "group" and `group_column`/`values` are not set.

        """
        if self.strategy == "random" and not self.test_size:
            raise ValueError("The random strategy requires a test size")

        if self.strategy == "group":
            msg = "The group strategy requires "
            if not self.group_column:
                raise ValueError(msg + "a group column to be specified")
            elif not self.values:
                raise ValueError(msg + "values to be specified")

        return self


class SplitRecalibrationConfig(BaseModel):
    """Configuration for the recalibration split strategy.

    Attributes
    ----------
    strategy : {"random", "group"}
        Strategy used to split the recalibration set. It must match the
        strategy of the test split.
    group_column : str or None, default=None
        Column identifying groups when `strategy` is "group".
    values : list of (str or int), or None, default=None
        Group values assigned to the recalibration split when `strategy`
        is "group".
    recalibration_size : float or None, default=None
        Fraction of the dataset reserved for recalibration when
        `strategy` is "random".

    """

    strategy: Literal["random", "group"]
    group_column: str | None = None
    values: list[str | int] | None = None
    recalibration_size: float | None = Field(default=None, gt=0.0, lt=1.0)
    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def validate_strategy(self) -> SplitRecalibrationConfig:
        """Validate that the chosen split strategy has its required fields
        set.

        Returns
        -------
        SplitRecalibrationConfig
            The validated configuration instance.

        Raises
        ------
        ValueError
            If `strategy` is "random" and `recalibration_size` is not
            set, or if `strategy` is "group" and `group_column`/`values`
            are not set.

        """
        if self.strategy == "random" and not self.recalibration_size:
            raise ValueError("The random strategy requires a test size")

        if self.strategy == "group":
            msg = "The group strategy requires "
            if not self.group_column:
                raise ValueError(msg + "a group column to be specified")
            elif not self.values:
                raise ValueError(msg + "values to be specified")

        return self


class CrossValConfig(BaseModel):
    """Configuration for cross-validation during model fitting.

    Attributes
    ----------
    strategy : {"random", "group", "loso"}
        Cross-validation splitting strategy. "loso" is leave-one-group-out:
        each fold holds out one group.
    grid_search : bool or None, default=None
        Whether to perform a grid search over hyperparameters.
    group_column : str or None, default=None
        Column identifying groups when `strategy` is "group" or "loso".
    n_splits : int or None, default=None
        Number of cross-validation folds, required when `strategy` is
        "random" or "group". Not used when `strategy` is "loso", which has
        one fold per group.
    shuffle : bool or None, default=None
        Whether to shuffle samples before splitting.

    """

    strategy: Literal["random", "group", "loso"]
    grid_search: bool | None = None
    group_column: str | None = None
    n_splits: int | None = Field(default=None, ge=2)
    shuffle: bool | None = None
    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def validate_strategy(self) -> CrossValConfig:
        """Validate that the group-based strategies have a group column set.

        Returns
        -------
        CrossValConfig
            The validated configuration instance.

        Raises
        ------
        ValueError
            If `strategy` is "group" or "loso" and `group_column` is not set,
            or if `strategy` is "random" or "group" and `n_splits` is not set.

        """
        if self.strategy in ("group", "loso") and not self.group_column:
            raise ValueError(
                f"The {self.strategy} strategy requires a group column to be specified"
            )
        if self.strategy in ("random", "group") and self.n_splits is None:
            raise ValueError(
                f"The {self.strategy} strategy requires the number of splits "
                "(n_splits) to be specified"
            )
        return self


class ValidationSubConfig(BaseModel):
    """Configuration grouping the test, cross-validation, and
    recalibration split settings.

    Attributes
    ----------
    test_split : SplitTestConfig
        Configuration for the train/test split.
    cross_validation : CrossValConfig or None, default=None
        Configuration for cross-validation, required when `run_mode`
        is "cv" or "audit".
    recalibration_split : SplitRecalibrationConfig or None, default=None
        Configuration for the recalibration split, required when a
        recalibration method is used.

    """

    test_split: SplitTestConfig
    cross_validation: CrossValConfig | None = None
    recalibration_split: SplitRecalibrationConfig | None = None

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def validate_group_strategies(self) -> ValidationSubConfig:
        """Validate that recalibration and test split have same strategy.

        Returns
        -------
        ValidationSubConfig
            The validated configuration instance.

        Raises
        ------
        ValueError
            If `recalibration_split` is set and its `strategy` differs
            from `test_split.strategy`.

        """
        _validate_recalibration_split_strategy(
            self.test_split, self.recalibration_split
        )
        return self

    @model_validator(mode="after")
    def validate_group_columns(self) -> ValidationSubConfig:
        """Validate that recalibration and test split have same group
        columns.

        Returns
        -------
        ValidationSubConfig
            The validated configuration instance.

        Raises
        ------
        ValueError
            If both splits use the "group" strategy but their group
            columns differ.

        """
        _validate_recalibration_split_group_column(
            self.test_split, self.recalibration_split
        )
        return self

    @model_validator(mode="after")
    def validate_group_values(self) -> ValidationSubConfig:
        """Validate that recalibration and test split have different
        values.

        Returns
        -------
        ValidationSubConfig
            The validated configuration instance.

        Raises
        ------
        ValueError
            If both splits use the "group" strategy and share any group
            value.

        """
        _validate_recalibration_split_values(self.test_split, self.recalibration_split)
        return self


class MetricsConfig(BaseModel):
    """Configuration for evaluation metrics and bootstrap confidence
    intervals.

    Attributes
    ----------
    metrics : list of str, default=["roc_auc", "ici"]
        Metric identifiers to compute, must be registered in
        `MetricRegistry`.
    n_bootstraps : int, default=200
        Number of bootstrap resamples used to compute confidence
        intervals.
    ci_level : float, default=0.95
        Confidence level for the computed interval bounds.
    cv_splines : int, default=3
        Number of internal cross-validation folds `SplineCalib` uses to
        select its regularization strength when computing the `ici`
        metric. Lower values fit faster (this runs once per bootstrap
        resample) at the cost of a coarser regularization search;
        `SplineCalib`'s own default is 5.

    """

    metrics: list[str] = Field(default=["roc_auc", "ici"])
    n_bootstraps: int = Field(default=200, ge=0)
    ci_level: float = Field(default=0.95, ge=0.0, le=1.0)
    cv_splines: int = Field(default=3, ge=2)
    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def validate_metrics(self) -> MetricsConfig:
        """Validate that all requested metrics are registered.

        Returns
        -------
        MetricsConfig
            The validated configuration instance.

        Raises
        ------
        ValueError
            If any entry in `metrics` is not a registered metric.

        """
        _validate_metrics_registered(self.metrics)
        return self


class FairnessConfig(BaseModel):
    """Configuration for subgroup fairness evaluation.

    Attributes
    ----------
    strata : list of str
        Column names used to stratify the evaluation.
    groups : dict of str to list of list of (int, float, or str), optional
        Mapping of stratum column names to the group value combinations
        to evaluate. Defaults to None.

    """

    strata: list[str]
    groups: dict[str, list[list[int | float | str]]] | None = None
    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def validate_group_keys(self) -> FairnessConfig:
        """Validate group keys are in strata.

        Returns
        -------
        FairnessConfig
            The validated configuration instance.

        Raises
        ------
        ValueError
            If a key in `groups` is not present in `strata`, or if its
            value list is empty.

        """
        if self.groups:
            for key in self.groups:
                if key not in self.strata:
                    raise ValueError(f"{key} should be in the strata list")
                if not self.groups[key]:
                    raise ValueError(f"{key} should not have an empty list")
        return self


class EvaluationSubConfig(BaseModel):
    """Configuration grouping the metrics and fairness evaluation
    settings.

    Attributes
    ----------
    metrics : MetricsConfig
        Configuration for evaluation metrics and bootstrap confidence
        intervals.
    fairness : FairnessConfig or None, default=None
        Configuration for subgroup fairness evaluation. If not set, no
        subgroup analysis is performed and the subgroup strata heatmaps are
        not plotted.

    """

    metrics: MetricsConfig
    fairness: FairnessConfig | None = None
    model_config = {"extra": "forbid"}


class WorkflowConfig(BaseModel):
    """The master schema for the workflow subconfiguration file.

    Attributes
    ----------
    random_state : int or None, default=42
        Seed controlling reproducibility of stochastic operations.
    n_jobs : int or None, default=1
        Number of parallel jobs to use during model fitting.
    preprocessing : PreprocessingConfig or None, default=None
        Configuration controlling whether and how preprocessing is
        applied.
    validation : ValidationSubConfig
        Configuration grouping the test, cross-validation, and
        recalibration split settings.
    evaluation : EvaluationSubConfig
        Configuration grouping the metrics and fairness evaluation
        settings.

    """

    random_state: int | None = Field(default=42, ge=0)
    n_jobs: int | None = Field(default=1, ge=-1)

    preprocessing: PreprocessingConfig | None = None
    validation: ValidationSubConfig
    evaluation: EvaluationSubConfig

    model_config = {"extra": "forbid"}


# --- MODEL SCHEMAS ---
class RecalibrationConfig(BaseModel):
    """Configuration for post-hoc model recalibration.

    Attributes
    ----------
    recalibrate : bool
        Whether to recalibrate the fitted model.
    method : {"isotonic", "sigmoid", "temperature"}, default="isotonic"
        Recalibration method to apply.
    hyperparameters : dict of str to Any, default={}
        Reserved for hyperparameters of the recalibration method. Currently
        not used: the values are accepted but not passed to the recalibrator.

    """

    recalibrate: bool
    method: Literal["isotonic", "sigmoid", "temperature"] = Field(default="isotonic")
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    model_config = {"extra": "forbid"}


class RecalibrationOverrideConfig(BaseModel):
    """Outcome-specific changes to the default recalibration.

    Only the fields that are set are applied on top of the default
    recalibration of the model.

    Attributes
    ----------
    recalibrate : bool or None, default=None
        Whether to recalibrate the fitted model. Required when the default
        model has no recalibration.
    method : {"isotonic", "sigmoid", "temperature"} or None, default=None
        Recalibration method. A method different from the default replaces
        the default hyperparameters.
    hyperparameters : dict of str to Any or None, default=None
        Hyperparameters forwarded to the recalibration method, merged into
        the default ones when the method is unchanged.

    """

    recalibrate: bool | None = None
    method: Literal["isotonic", "sigmoid", "temperature"] | None = None
    hyperparameters: dict[str, Any] | None = None
    model_config = {"extra": "forbid"}


class ModelSetup(BaseModel):
    """Configuration for a model, recalibrator, and their hyperparameters.

    Attributes
    ----------
    algorithm : str
        Name of the registered model class to instantiate.
    hyperparameters : dict of str to Any, default={}
        Hyperparameters forwarded to the model constructor.
    recalibration : RecalibrationConfig or None, default=None
        Configuration for post-hoc recalibration of the fitted model.

    """

    algorithm: str
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    recalibration: RecalibrationConfig | None = None
    model_config = {"extra": "forbid"}


class ModelOverride(BaseModel):
    """Outcome-specific changes to the default model.

    Only the fields that are set are applied on top of the default model.
    Setting `recalibration` to None disables the recalibration of the
    outcome.

    Attributes
    ----------
    algorithm : str or None, default=None
        Name of the registered model class. Defaults to the algorithm of the
        default model.
    hyperparameters : dict of str to Any or None, default=None
        Hyperparameters forwarded to the model constructor. They are merged
        into the default ones when the algorithm is unchanged, and replace
        them otherwise.
    recalibration : RecalibrationOverrideConfig or None, default=None
        Changes to the recalibration of the default model.

    """

    algorithm: str | None = None
    hyperparameters: dict[str, Any] | None = None
    recalibration: RecalibrationOverrideConfig | None = None
    model_config = {"extra": "forbid"}


# --- DISPLAY SCHEMAS ---
class DisplayDefaultsConfig(BaseModel):
    """Default visualization parameters across all plot types.

    Attributes
    ----------
    n_bootstraps : int, default=1000
        Number of bootstrap resamples used to compute confidence
        intervals.
    save : bool, default=True
        Whether to persist generated plots to disk.
    show : bool, default=False
        Whether to display generated plots interactively.
    n_bins : int, default=10
        Number of bins used for the reliability diagram.
    dist_n_bins : int, default=10
        Number of bins used for the data distribution histogram.
    dist_yscale : {"linear", "log", "symlog", "logit"}, default="linear"
        Scale of the y-axis of the data distribution histogram.
    strategy : {"uniform", "quantile", "spline"}, default="uniform"
        Strategy used to compute calibration curves.
    n_jobs : int or None, default=1
        Number of parallel jobs used to compute bootstrap resamples (e.g.
        the spline-calibration bootstrap loop behind the reliability
        diagram). Independent of `workflow.n_jobs`, since plots may be
        rendered standalone without a full workflow run in scope.
    cv_splines : int, default=3
        Number of internal cross-validation folds `SplineCalib` uses to
        select its regularization strength when `strategy='spline'`.
        Lower values fit faster (this runs once per bootstrap resample)
        at the cost of a coarser regularization search; `SplineCalib`'s
        own default is 5.

    """

    n_bootstraps: int = Field(default=1000, ge=0)
    save: bool = True
    show: bool = False
    n_bins: int = Field(default=10, ge=1)
    dist_n_bins: int = Field(default=10, ge=1)
    dist_yscale: Literal["linear", "log", "symlog", "logit"] = "linear"
    strategy: Literal["uniform", "quantile", "spline"] = "uniform"
    n_jobs: int | None = Field(default=1, ge=-1)
    cv_splines: int = Field(default=3, ge=2)
    model_config = {"extra": "allow"}


class DisplayConfig(BaseModel):
    """Configuration settings for pipeline evaluation graphics and themes.

    Attributes
    ----------
    defaults : DisplayDefaultsConfig
        Default visualization parameters applied across all plot types.
    overrides : dict of str to dict of str to Any, default={}
        Plot-type-specific parameter overrides, keyed by plot type.
    outcome_overrides : dict, default={}
        Outcome-specific plot parameter overrides, keyed by outcome name
        then plot type, as ``{outcome: {plot_type: {param: value}}}``.
    theme : dict of str to Any, or None, default=None
        Parameters of `MedpipeTheme` (e.g. `dpi`, `primary_color`, `font_family`).

    """

    defaults: DisplayDefaultsConfig = Field(default_factory=DisplayDefaultsConfig)
    overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)
    outcome_overrides: dict[str, dict[str, dict[str, Any]]] = Field(
        default_factory=dict
    )
    theme: dict[str, Any] | None = Field(default=None)

    model_config = {"extra": "forbid"}

    @model_validator(mode="before")
    @classmethod
    def handle_flat_and_legacy_keys(cls, data: Any) -> Any:
        """Process legacy flat keys (e.g., calibration_strategy) into
        defaults.

        Parameters
        ----------
        data : Any
            Raw input data passed to the model validator.

        Returns
        -------
        Any
            The input data with legacy top-level keys folded into
            `defaults`.

        """
        if isinstance(data, dict):
            data = data.copy()
            defaults = data.get("defaults", {})
            if not isinstance(defaults, dict):
                defaults = {}

            # Backward compatibility for flat calibration_strategy key
            if "calibration_strategy" in data:
                defaults.setdefault("strategy", data.pop("calibration_strategy"))

            # Lift any top-level parameter overrides into defaults dict
            for legacy_key in ("n_bootstraps", "save", "show", "n_bins", "strategy"):
                if legacy_key in data:
                    defaults.setdefault(legacy_key, data.pop(legacy_key))

            if defaults:
                data["defaults"] = defaults
        return data

    @field_validator("theme")
    @classmethod
    def validate_theme_keys(cls, v: dict[str, Any] | None) -> dict[str, Any] | None:
        """Ensure theme parameters are attributes of `MedpipeTheme`.

        Parameters
        ----------
        v : dict of str to Any or None
            Candidate `theme` mapping.

        Returns
        -------
        dict of str to Any or None
            The validated mapping.

        Raises
        ------
        ValueError
            If a key is not a theme parameter.

        """
        _validate_theme_keys(v)
        return v

    @field_validator("overrides", "outcome_overrides")
    @classmethod
    def validate_plot_override_keys(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Ensure plot override identifiers correspond to valid plot
        types.

        Parameters
        ----------
        v : dict of str to Any
            Candidate `overrides` or `outcome_overrides` mapping.

        Returns
        -------
        dict of str to Any
            The validated mapping.

        Raises
        ------
        ValueError
            If any plot type key is not a recognized plot type.

        """
        valid_plots = {
            "calibration",
            "reliability",
            "reliability_diagram",
            "precision_recall",
            "pr",
            "pr_curve",
            "roc",
            "roc_curve",
            "distribution",
            "data_distribution",
            "dist",
            "dca",
            "dca_curve",
            "strata_heatmap",
            "heatmap",
        }
        return _validate_plot_type_keys(v, valid_plots)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DisplayConfig:
        """Instantiate DisplayConfig from parsed TOML dictionary.

        Parameters
        ----------
        data : dict of str to Any
            Parsed TOML data for the display configuration section.

        Returns
        -------
        DisplayConfig
            The constructed configuration instance.

        """
        return cls.model_validate(data)


# --- GLOBAL MEDPIPE CONFIGURATION SCHEMA ---
class MedpipeClassifierConfig(BaseModel):
    """The master schema for a single-file configuration.

    Attributes
    ----------
    meta : MetaConfig
        Project metadata and run-mode configuration.
    data : DataConfig
        Dataset location and predictor/outcome column configuration.
    workflow : WorkflowConfig
        Preprocessing, validation, and evaluation configuration.
    display : DisplayConfig or None, default=None
        Visualization configuration, required when `run_mode` is "audit"
        or "eval".
    default_model : ModelSetup
        Default model, hyperparameter, and recalibration setup applied
        to all outcomes.
    outcome_overrides : dict of str to ModelOverride, default={}
        Per-outcome overrides cascaded onto `default_model`.
    resolved_models : dict of str to ModelSetup, default={}
        Fully resolved per-outcome model configurations, generated
        during validation.

    """

    meta: MetaConfig
    data: DataConfig
    workflow: WorkflowConfig
    display: DisplayConfig | None = None

    # The default setup applied to all outcomes
    default_model: ModelSetup

    # Optional overrides keyed by outcome name
    outcome_overrides: dict[str, ModelOverride] = Field(default_factory=dict)

    # Dynamically generated during validation: fully resolved configurations per outcome
    resolved_models: dict[str, ModelSetup] = Field(default_factory=dict, init_var=False)

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def resolve_cascading_models(self) -> MedpipeClassifierConfig:
        """Cascade default_model settings into outcome_overrides.

        Returns
        -------
        MedpipeClassifierConfig
            The validated configuration instance, with `resolved_models`
            populated.

        Raises
        ------
        ValueError
            If an override adds a recalibration to a default model that has
            none without setting `recalibrate`.

        """
        resolved = {}
        for outcome in self.data.outcomes:
            base_setup = self.default_model.model_dump()
            if outcome in self.outcome_overrides:
                override_setup = self.outcome_overrides[outcome].model_dump(
                    exclude_unset=True
                )
                try:
                    base_setup = _cascade_model_setup(base_setup, override_setup)
                except ValueError as err:
                    raise ValueError(f"Outcome override '{outcome}': {err}") from err
            resolved[outcome] = ModelSetup(**base_setup)

        self.resolved_models = resolved
        return self

    @model_validator(mode="after")
    def validate_metrics_for_task(self) -> MedpipeClassifierConfig:
        """Validate that the metrics can be computed for the type of outcome.

        Returns
        -------
        MedpipeClassifierConfig
            The validated configuration instance.

        Raises
        ------
        ValueError
            If a metric cannot be computed for the type of outcome.

        """
        metrics = self.workflow.evaluation.metrics.metrics
        _validate_metrics_for_task(metrics, "classification")
        return self

    @model_validator(mode="after")
    def validate_preprocessing_columns(self) -> MedpipeClassifierConfig:
        """Validate that preprocessing operations only act on predictors.

        Returns
        -------
        MedpipeClassifierConfig
            The validated configuration instance.

        Raises
        ------
        ValueError
            If an operation refers to a column that is not in
            `data.predictors`.

        """
        _validate_preprocessing_columns(
            self.workflow.preprocessing, self.data.predictors
        )
        return self

    @model_validator(mode="after")
    def validate_algorithms_registered(self) -> MedpipeClassifierConfig:
        """Validate that every resolved algorithm is known to ModelRegistry.

        Returns
        -------
        MedpipeClassifierConfig
            The validated configuration instance.

        Raises
        ------
        ValueError
            If an algorithm cannot be resolved.

        """
        _validate_algorithms_registered(
            {name: model.algorithm for name, model in self.resolved_models.items()}
        )
        return self

    @model_validator(mode="after")
    def validate_recalibration(self) -> MedpipeClassifierConfig:
        """Check recalibration split is specified with recalibration
        method.

        Returns
        -------
        MedpipeClassifierConfig
            The validated configuration instance.

        Raises
        ------
        ValueError
            If `default_model.recalibration` is set, or if an outcome
            override enables recalibration, but
            `workflow.validation.recalibration_split` is not set.

        """
        needs_split = self.default_model.recalibration is not None or any(
            model.recalibration is not None and model.recalibration.recalibrate
            for model in self.resolved_models.values()
        )
        if needs_split and not self.workflow.validation.recalibration_split:
            expr = (
                "Recalibration validation split must be "
                "specified when a recalibration method is used"
            )
            raise ValueError(expr)
        return self

    @model_validator(mode="after")
    def validate_cross_validation(self) -> MedpipeClassifierConfig:
        """Check that a cross-validation config is passed with correct
        run modes.

        Returns
        -------
        MedpipeClassifierConfig
            The validated configuration instance.

        Raises
        ------
        ValueError
            If `meta.run_mode` is "cv" or "audit" and
            `workflow.validation.cross_validation` is not set.

        """
        if (
            self.meta.run_mode in ("cv", "audit")
            and self.workflow.validation.cross_validation is None
        ):
            expr = (
                "Cross-validation parameters must be specified "
                "when run_mode is 'cv' or 'audit'"
            )
            raise ValueError(expr)
        return self

    @model_validator(mode="after")
    def validate_audit_and_eval_run_mode(self) -> MedpipeClassifierConfig:
        """Check that audit and eval run modes have a display configuration.

        Returns
        -------
        MedpipeClassifierConfig
            The validated configuration instance.

        Raises
        ------
        ValueError
            If `run_mode` is "audit" or "eval" and `display` is not set.

        """
        run_mode = self.meta.run_mode
        if run_mode in ("audit", "eval") and self.display is None:
            raise ValueError(
                "Display parameters must be specified "
                "when run_mode is 'audit' or 'eval'"
            )
        return self

    @model_validator(mode="after")
    def validate_outcome_overrides_exist_in_outcomes(self) -> MedpipeClassifierConfig:
        """Ensures all outcome names in outcome_overrides are defined in
        data.outcomes.

        Returns
        -------
        MedpipeClassifierConfig
            The validated configuration instance.

        Raises
        ------
        ValueError
            If any key in `outcome_overrides` or
            `display.outcome_overrides` is not present in
            `data.outcomes`.

        """
        valid_outcomes = set(self.data.outcomes)

        if self.outcome_overrides and self.data and self.data.outcomes:
            for override_outcome in self.outcome_overrides:
                if override_outcome not in valid_outcomes:
                    raise ValueError(
                        f"Outcome override '{override_outcome}' is not present "
                        f"in data.outcomes: {self.data.outcomes}"
                    )

        if self.display and self.display.outcome_overrides:
            for override_outcome in self.display.outcome_overrides:
                if override_outcome not in valid_outcomes:
                    raise ValueError(
                        f"Display outcome override '{override_outcome}' is not present "
                        f"in data.outcomes: {self.data.outcomes}"
                    )
        return self


# ==============================================================================
# REGRESSOR CONFIGURATION SCHEMA (pydantic)
# ==============================================================================
# --- REGRESSOR MODEL SCHEMAS ---
class RegressorModelSetup(BaseModel):
    """Configuration for a regression model and its hyperparameters.

    Unlike `ModelSetup`, this has no `recalibration` field: post-hoc
    recalibration is a classification-only concept and is not supported on
    the regression track.

    Attributes
    ----------
    algorithm : str
        Name of the registered model class to instantiate.
    hyperparameters : dict of str to Any, default={}
        Hyperparameters forwarded to the model constructor.

    """

    algorithm: str
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    model_config = {"extra": "forbid"}


class RegressorModelOverride(BaseModel):
    """Outcome-specific changes to the default regression model.

    Only the fields that are set are applied on top of the default model.

    Attributes
    ----------
    algorithm : str or None, default=None
        Name of the registered model class. Defaults to the algorithm of the
        default model.
    hyperparameters : dict of str to Any or None, default=None
        Hyperparameters forwarded to the model constructor. They are merged
        into the default ones when the algorithm is unchanged, and replace
        them otherwise.

    """

    algorithm: str | None = None
    hyperparameters: dict[str, Any] | None = None
    model_config = {"extra": "forbid"}


# --- REGRESSOR EVALUATION SCHEMAS ---
class RegressorMetricsConfig(BaseModel):
    """Configuration for regression evaluation metrics and bootstrap
    confidence intervals.

    Attributes
    ----------
    metrics : list of str, default=["rmse", "mae"]
        Metric identifiers to compute, must be registered in
        `MetricRegistry`.
    n_bootstraps : int, default=200
        Number of bootstrap resamples used to compute confidence
        intervals.
    ci_level : float, default=0.95
        Confidence level for the computed interval bounds.

    """

    metrics: list[str] = Field(default=["rmse", "mae"])
    n_bootstraps: int = Field(default=200, ge=0)
    ci_level: float = Field(default=0.95, ge=0.0, le=1.0)
    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def validate_metrics(self) -> RegressorMetricsConfig:
        """Validate that all requested metrics are registered.

        Returns
        -------
        RegressorMetricsConfig
            The validated configuration instance.

        Raises
        ------
        ValueError
            If any entry in `metrics` is not a registered metric.

        """
        _validate_metrics_registered(self.metrics)
        return self


class RegressorValidationSubConfig(BaseModel):
    """Configuration grouping the test, cross-validation, and
    recalibration split settings for a regression workflow.

    `recalibration_split` produces a holdout set for future post-hoc
    regression recalibration (e.g. PIT recalibration); no model-level
    recalibration step consumes it yet — `RegressorModelSetup` has no
    `recalibration` field, unlike the classifier's `ModelSetup`.

    Attributes
    ----------
    test_split : SplitTestConfig
        Configuration for the train/test split.
    cross_validation : CrossValConfig or None, default=None
        Configuration for cross-validation, required when `run_mode`
        is "cv" or "audit".
    recalibration_split : SplitRecalibrationConfig or None, default=None
        Configuration for the recalibration split.

    """

    test_split: SplitTestConfig
    cross_validation: CrossValConfig | None = None
    recalibration_split: SplitRecalibrationConfig | None = None
    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def validate_group_strategies(self) -> RegressorValidationSubConfig:
        """Validate that recalibration and test split have same strategy.

        Returns
        -------
        RegressorValidationSubConfig
            The validated configuration instance.

        Raises
        ------
        ValueError
            If `recalibration_split` is set and its `strategy` differs
            from `test_split.strategy`.

        """
        _validate_recalibration_split_strategy(
            self.test_split, self.recalibration_split
        )
        return self

    @model_validator(mode="after")
    def validate_group_columns(self) -> RegressorValidationSubConfig:
        """Validate that recalibration and test split have same group
        columns.

        Returns
        -------
        RegressorValidationSubConfig
            The validated configuration instance.

        Raises
        ------
        ValueError
            If both splits use the "group" strategy but their group
            columns differ.

        """
        _validate_recalibration_split_group_column(
            self.test_split, self.recalibration_split
        )
        return self

    @model_validator(mode="after")
    def validate_group_values(self) -> RegressorValidationSubConfig:
        """Validate that recalibration and test split have different
        values.

        Returns
        -------
        RegressorValidationSubConfig
            The validated configuration instance.

        Raises
        ------
        ValueError
            If both splits use the "group" strategy and share any group
            value.

        """
        _validate_recalibration_split_values(self.test_split, self.recalibration_split)
        return self


class RegressorEvaluationSubConfig(BaseModel):
    """Configuration grouping the metrics and fairness evaluation
    settings for a regression workflow.

    Attributes
    ----------
    metrics : RegressorMetricsConfig
        Configuration for regression evaluation metrics and bootstrap
        confidence intervals.
    fairness : FairnessConfig or None, default=None
        Configuration for subgroup fairness evaluation. If not set, no
        subgroup analysis is performed and the subgroup strata heatmaps are
        not plotted.

    """

    metrics: RegressorMetricsConfig
    fairness: FairnessConfig | None = None
    model_config = {"extra": "forbid"}


class RegressorWorkflowConfig(BaseModel):
    """The master schema for the regression workflow subconfiguration.

    Attributes
    ----------
    random_state : int or None, default=42
        Seed controlling reproducibility of stochastic operations.
    n_jobs : int or None, default=1
        Number of parallel jobs to use during model fitting.
    preprocessing : PreprocessingConfig or None, default=None
        Configuration controlling whether and how preprocessing is
        applied.
    validation : RegressorValidationSubConfig
        Configuration grouping the test, cross-validation, and
        recalibration split settings.
    evaluation : RegressorEvaluationSubConfig
        Configuration grouping the metrics and fairness evaluation
        settings.

    """

    random_state: int | None = Field(default=42, ge=0)
    n_jobs: int | None = Field(default=1, ge=-1)

    preprocessing: PreprocessingConfig | None = None
    validation: RegressorValidationSubConfig
    evaluation: RegressorEvaluationSubConfig

    model_config = {"extra": "forbid"}


# --- REGRESSOR DISPLAY SCHEMAS ---
class RegressorDisplayDefaultsConfig(BaseModel):
    """Default visualization parameters across all regression plot types.

    Attributes
    ----------
    n_bootstraps : int, default=1000
        Number of bootstrap resamples used to compute the confidence bands of
        the coverage, sharpness, and Winkler score curves.
    save : bool, default=True
        Whether to persist generated plots to disk.
    show : bool, default=False
        Whether to display generated plots interactively.
    n_jobs : int or None, default=1
        Number of parallel jobs used to compute bootstrap resamples.
        Independent of `workflow.n_jobs`.
    dist_n_bins : int, default=20
        Number of bins used for the histogram of the target values.
    dist_yscale : {"linear", "log", "symlog", "logit"}, default="linear"
        Scale of the y-axis of the histogram of the target values.
    coverage_levels : list of (int or float), default=[5, 15, ..., 95]
        Nominal central-interval coverage levels, in percent, evaluated by
        the coverage, sharpness, and Winkler score curves. Each level must be
        strictly between 0 and 100.
    pit_n_bins : int, default=20
        Number of bins of the probability integral transform (PIT) histogram.

    """

    n_bootstraps: int = Field(default=1000, ge=0)
    save: bool = True
    show: bool = False
    n_jobs: int | None = Field(default=1, ge=-1)
    dist_n_bins: int = Field(default=20, ge=1)
    dist_yscale: Literal["linear", "log", "symlog", "logit"] = "linear"
    coverage_levels: list[int | float] = Field(
        default_factory=lambda: [*range(5, 100, 10)]
    )
    pit_n_bins: int = Field(default=20, ge=1)
    model_config = {"extra": "forbid"}

    @field_validator("coverage_levels")
    @classmethod
    def validate_coverage_levels(cls, v: list[int | float]) -> list[int | float]:
        """Validate that the coverage levels are percentages.

        Parameters
        ----------
        v : list of (int or float)
            Candidate coverage levels.

        Returns
        -------
        list of (int or float)
            The validated coverage levels.

        Raises
        ------
        ValueError
            If the list is empty or a level is not strictly between 0 and 100.

        """
        if not v:
            raise ValueError("coverage_levels should not be empty")
        invalid = [level for level in v if not 0 < level < 100]
        if invalid:
            raise ValueError(
                f"coverage_levels should be strictly between 0 and 100, got {invalid}"
            )
        return v


class RegressorDisplayConfig(BaseModel):
    """Configuration settings for regression pipeline evaluation graphics
    and themes.

    Attributes
    ----------
    defaults : RegressorDisplayDefaultsConfig
        Default visualization parameters applied across all plot types.
    overrides : dict of str to dict of str to Any, default={}
        Plot-type-specific parameter overrides, keyed by plot type.
    outcome_overrides : dict, default={}
        Outcome-specific plot parameter overrides, keyed by outcome name
        then plot type, as ``{outcome: {plot_type: {param: value}}}``.
    theme : dict of str to Any, or None, default=None
        Parameters of `MedpipeTheme` (e.g. `dpi`, `primary_color`, `font_family`).

    """

    defaults: RegressorDisplayDefaultsConfig = Field(
        default_factory=RegressorDisplayDefaultsConfig
    )
    overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)
    outcome_overrides: dict[str, dict[str, dict[str, Any]]] = Field(
        default_factory=dict
    )
    theme: dict[str, Any] | None = Field(default=None)

    model_config = {"extra": "forbid"}

    @field_validator("theme")
    @classmethod
    def validate_theme_keys(cls, v: dict[str, Any] | None) -> dict[str, Any] | None:
        """Ensure theme parameters are attributes of `MedpipeTheme`.

        Parameters
        ----------
        v : dict of str to Any or None
            Candidate `theme` mapping.

        Returns
        -------
        dict of str to Any or None
            The validated mapping.

        Raises
        ------
        ValueError
            If a key is not a theme parameter.

        """
        _validate_theme_keys(v)
        return v

    @field_validator("overrides", "outcome_overrides")
    @classmethod
    def validate_plot_override_keys(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Ensure plot override identifiers correspond to valid regression
        plot types.

        Parameters
        ----------
        v : dict of str to Any
            Candidate `overrides` or `outcome_overrides` mapping.

        Returns
        -------
        dict of str to Any
            The validated mapping.

        Raises
        ------
        ValueError
            If any plot type key is not a recognized regression plot type.

        """
        valid_plots = {
            "coverage",
            "coverage_curve",
            "sharpness",
            "sharpness_curve",
            "winkler",
            "winkler_curve",
            "winkler_score",
            "marginal_calibration",
            "marginal_calib",
            "pit_histogram",
            "pit",
            "distribution",
            "data_distribution",
            "dist",
            "strata_heatmap",
            "heatmap",
        }
        return _validate_plot_type_keys(v, valid_plots)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RegressorDisplayConfig:
        """Instantiate RegressorDisplayConfig from parsed TOML dictionary.

        Parameters
        ----------
        data : dict of str to Any
            Parsed TOML data for the display configuration section.

        Returns
        -------
        RegressorDisplayConfig
            The constructed configuration instance.

        """
        return cls.model_validate(data)


# --- GLOBAL MEDPIPE REGRESSOR CONFIGURATION SCHEMA ---
class MedpipeRegressorConfig(BaseModel):
    """The master schema for a single-file regression configuration.

    Attributes
    ----------
    meta : MetaConfig
        Project metadata and run-mode configuration.
    data : DataConfig
        Dataset location and predictor/outcome column configuration.
    workflow : RegressorWorkflowConfig
        Preprocessing, validation, and evaluation configuration.
    display : RegressorDisplayConfig or None, default=None
        Visualization configuration, required when `run_mode` is "audit"
        or "eval".
    default_model : RegressorModelSetup
        Default model and hyperparameter setup applied to all outcomes.
    outcome_overrides : dict of str to RegressorModelOverride, default={}
        Per-outcome overrides cascaded onto `default_model`.
    resolved_models : dict of str to RegressorModelSetup, default={}
        Fully resolved per-outcome model configurations, generated
        during validation.

    """

    meta: MetaConfig
    data: DataConfig
    workflow: RegressorWorkflowConfig
    display: RegressorDisplayConfig | None = None

    # The default setup applied to all outcomes
    default_model: RegressorModelSetup

    # Optional overrides keyed by outcome name
    outcome_overrides: dict[str, RegressorModelOverride] = Field(default_factory=dict)

    # Dynamically generated during validation: fully resolved configurations per outcome
    resolved_models: dict[str, RegressorModelSetup] = Field(
        default_factory=dict, init_var=False
    )

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def resolve_cascading_models(self) -> MedpipeRegressorConfig:
        """Cascade default_model settings into outcome_overrides.

        Returns
        -------
        MedpipeRegressorConfig
            The validated configuration instance, with `resolved_models`
            populated.

        """
        resolved = {}
        for outcome in self.data.outcomes:
            base_setup = self.default_model.model_dump()
            if outcome in self.outcome_overrides:
                override_setup = self.outcome_overrides[outcome].model_dump(
                    exclude_unset=True
                )
                base_setup = _cascade_model_setup(base_setup, override_setup)
            resolved[outcome] = RegressorModelSetup(**base_setup)

        self.resolved_models = resolved
        return self

    @model_validator(mode="after")
    def validate_metrics_for_task(self) -> MedpipeRegressorConfig:
        """Validate that the metrics can be computed for the type of outcome.

        Returns
        -------
        MedpipeRegressorConfig
            The validated configuration instance.

        Raises
        ------
        ValueError
            If a metric cannot be computed for the type of outcome, or if a
            distributional metric is used with a model that cannot predict a
            distribution.

        """
        metrics = self.workflow.evaluation.metrics.metrics
        _validate_metrics_for_task(metrics, "regression")
        if any("predict_dist" in _metric_response_methods(name) for name in metrics):
            _validate_metrics_need_distribution(
                {name: model.algorithm for name, model in self.resolved_models.items()}
            )
        return self

    @model_validator(mode="after")
    def validate_preprocessing_columns(self) -> MedpipeRegressorConfig:
        """Validate that preprocessing operations only act on predictors.

        Returns
        -------
        MedpipeRegressorConfig
            The validated configuration instance.

        Raises
        ------
        ValueError
            If an operation refers to a column that is not in
            `data.predictors`.

        """
        _validate_preprocessing_columns(
            self.workflow.preprocessing, self.data.predictors
        )
        return self

    @model_validator(mode="after")
    def validate_algorithms_registered(self) -> MedpipeRegressorConfig:
        """Validate that every resolved algorithm is known to ModelRegistry.

        Returns
        -------
        MedpipeRegressorConfig
            The validated configuration instance.

        Raises
        ------
        ValueError
            If an algorithm cannot be resolved.

        """
        _validate_algorithms_registered(
            {name: model.algorithm for name, model in self.resolved_models.items()}
        )
        return self

    @model_validator(mode="after")
    def validate_cross_validation(self) -> MedpipeRegressorConfig:
        """Check that a cross-validation config is passed with correct
        run modes.

        Returns
        -------
        MedpipeRegressorConfig
            The validated configuration instance.

        Raises
        ------
        ValueError
            If `meta.run_mode` is "cv" or "audit" and
            `workflow.validation.cross_validation` is not set.

        """
        if (
            self.meta.run_mode in ("cv", "audit")
            and self.workflow.validation.cross_validation is None
        ):
            expr = (
                "Cross-validation parameters must be specified "
                "when run_mode is 'cv' or 'audit'"
            )
            raise ValueError(expr)
        return self

    @model_validator(mode="after")
    def validate_audit_and_eval_run_mode(self) -> MedpipeRegressorConfig:
        """Check that audit and eval run modes have a display configuration.

        Returns
        -------
        MedpipeRegressorConfig
            The validated configuration instance.

        Raises
        ------
        ValueError
            If `run_mode` is "audit" or "eval" and `display` is not set.

        """
        run_mode = self.meta.run_mode
        if run_mode in ("audit", "eval") and self.display is None:
            raise ValueError(
                "Display parameters must be specified "
                "when run_mode is 'audit' or 'eval'"
            )
        return self

    @model_validator(mode="after")
    def validate_outcome_overrides_exist_in_outcomes(self) -> MedpipeRegressorConfig:
        """Ensures all outcome names in outcome_overrides are defined in
        data.outcomes.

        Returns
        -------
        MedpipeRegressorConfig
            The validated configuration instance.

        Raises
        ------
        ValueError
            If any key in `outcome_overrides` or
            `display.outcome_overrides` is not present in
            `data.outcomes`.

        """
        valid_outcomes = set(self.data.outcomes)

        if self.outcome_overrides and self.data and self.data.outcomes:
            for override_outcome in self.outcome_overrides:
                if override_outcome not in valid_outcomes:
                    raise ValueError(
                        f"Outcome override '{override_outcome}' is not present "
                        f"in data.outcomes: {self.data.outcomes}"
                    )

        if self.display and self.display.outcome_overrides:
            for override_outcome in self.display.outcome_overrides:
                if override_outcome not in valid_outcomes:
                    raise ValueError(
                        f"Display outcome override '{override_outcome}' is not present "
                        f"in data.outcomes: {self.data.outcomes}"
                    )
        return self
