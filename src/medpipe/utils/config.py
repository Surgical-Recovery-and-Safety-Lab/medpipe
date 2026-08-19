"""
Configuration utilities module.

This module provides configuration schemas.

"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Union

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

VerboseType = Union[VerbosityMode, bool, VerbosityInt]


# ==============================================================================
# CONFIGURATION SCHEMA (pydantic)
# ==============================================================================
# --- TOP-LEVEL MASTER SCHEMAS ---
class MetaConfig(BaseModel):
    """The master schema for the meta section of the configuration file."""

    project_name: str
    run_mode: Literal["fast", "eval", "cv", "audit"] = "audit"
    verbose: VerboseType = Field(
        default="compact",
        description="Console logging verbosity: 'quiet' (0), 'compact' (1), 'info' (2), 'debug' (3).",
    )
    model_config = {"extra": "forbid"}

    @field_validator("project_name")
    @classmethod
    def validate_project_name(cls, name: str) -> str:
        """Validate that project name is not empty."""
        if not name:
            raise ValueError("Project name should not be an empty string.")

        return name


# --- DATA SCHEMAS ---
class DataConfig(BaseModel):
    """The master schema for the data section of the configuration file."""

    path: str
    predictors: list[str]
    outcomes: list[str]
    model_config = {"extra": "allow"}

    @field_validator("path")
    @classmethod
    def validate_path(cls, file: str) -> str:
        """Validate that path is a points to a file."""
        data_path = Path(file)
        suffix = data_path.suffix
        if suffix == "":
            raise ValueError("path should be a file, but got no suffix")
        return file

    @model_validator(mode="after")
    def check_for_target_leakage(self) -> "DataConfig":
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
    name: str  # Matches the exact class name
    columns: list[str]  # The specific columns this transformer applies to
    model_config = {"extra": "allow"}

    @field_validator("columns")
    @classmethod
    def validate_columns(cls, columns: list[str]) -> list[str]:
        """Validate that columns are not empty."""
        if not columns:
            raise ValueError("Columns cannot be an empty list")
        return columns


class PreprocessingConfig(BaseModel):
    preprocess: bool | None = None
    operations: list[PreprocessOperationConfig] | None = None
    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def validate_operations(self) -> "PreprocessingConfig":
        if self.preprocess and not self.operations:
            raise ValueError("Operations must be specified if preprocess is True")
        return self


class SplitTestConfig(BaseModel):
    strategy: Literal["random", "group"] = "random"
    group_column: str | None = None
    values: list[str | int] | None = None
    test_size: float | None = Field(default=None, gt=0.0, lt=1.0)
    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def validate_strategy(self) -> "SplitTestConfig":
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
    strategy: Literal["random", "group"] | None = None
    group_column: str | None = None
    values: list[str | int] | None = None
    recalibration_size: float | None = Field(default=None, gt=0.0, lt=1.0)
    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def validate_strategy(self) -> "SplitRecalibrationConfig":
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
    strategy: Literal["random", "group"]
    grid_search: bool | None = None
    group_column: str | None = None
    n_splits: int | None = Field(default=None, ge=2)
    shuffle: bool | None = None
    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def validate_strategy(self) -> "CrossValConfig":
        if self.strategy == "group" and not self.group_column:
            raise ValueError(
                "The group strategy requires a group column to be specified"
            )
        return self


class ValidationSubConfig(BaseModel):
    test_split: SplitTestConfig
    cross_validation: CrossValConfig | None = None
    recalibration_split: SplitRecalibrationConfig | None = None

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def validate_group_strategies(self) -> "ValidationSubConfig":
        """Validate that recalibration and test split have same strategy."""
        if self.recalibration_split:
            if self.recalibration_split.strategy != self.test_split.strategy:
                raise ValueError("Recalibration and test strategies should match")
        return self

    @model_validator(mode="after")
    def validate_group_columns(self) -> "ValidationSubConfig":
        """Validate that recalibration and test split have same group columns."""
        if self.recalibration_split:
            # Check only when strategy is group
            if (
                self.recalibration_split.strategy == "group"
                and self.test_split.strategy == "group"
            ):
                if (
                    self.recalibration_split.group_column
                    != self.test_split.group_column
                ):
                    raise ValueError(
                        "Recalibration and test group columns should match"
                    )
        return self

    @model_validator(mode="after")
    def validate_group_values(self) -> "ValidationSubConfig":
        """Validate that recalibration and test split have different values."""
        if self.recalibration_split and self.recalibration_split.values is not None:
            if (
                self.recalibration_split.strategy == "group"
                and self.test_split.strategy == "group"
            ):
                for value in self.recalibration_split.values:
                    if value in self.test_split.values:  # type: ignore
                        raise ValueError(
                            "Recalibration and test values should be different"
                        )
        return self


class MetricsConfig(BaseModel):
    metrics: list[str] = Field(default=["roc_auc", "ici"])
    n_bootstraps: int = Field(default=200, ge=0)
    ci_level: float = Field(default=0.95, ge=0.0, le=1.0)
    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def validate_metrics(self) -> "MetricsConfig":
        """Validate input metrics."""
        from medpipe.metrics.core import METRICS

        for metric in self.metrics:
            if metric not in METRICS:
                expr = (
                    f"{metric} was not found in available metric "
                    f"list. Available metrics are {METRICS}"
                )
                raise ValueError(expr)

        return self


class FairnessConfig(BaseModel):
    strata: list[str]
    groups: dict[str, list[list[int | float | str]]] | None = None
    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def validate_group_keys(self) -> "FairnessConfig":
        """Validate group keys are in strata."""
        if self.groups:
            for key in self.groups.keys():
                if key not in self.strata:
                    raise ValueError(f"{key} should be in the strata list")
                if not self.groups[key]:
                    raise ValueError(f"{key} should not have an empty list")
        return self


class EvaluationSubConfig(BaseModel):
    metrics: MetricsConfig
    fairness: FairnessConfig | None = None
    model_config = {"extra": "forbid"}


class WorkflowConfig(BaseModel):
    """The master schema for the workflow subconfiguration file."""

    random_state: int | None = Field(default=42, ge=0)
    n_jobs: int | None = Field(default=1, ge=-1)

    preprocessing: PreprocessingConfig | None = None
    validation: ValidationSubConfig
    evaluation: EvaluationSubConfig

    model_config = {"extra": "forbid"}


# --- MODEL SCHEMAS ---
class RecalibrationConfig(BaseModel):
    recalibrate: bool
    method: Literal["isotonic", "sigmoid", "temperature"] = Field(default="isotonic")
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    model_config = {"extra": "forbid"}


class ModelSetup(BaseModel):
    """Configuration for a model, recalibrator, and their hyperparameters."""

    algorithm: str
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    recalibration: RecalibrationConfig | None = None
    model_config = {"extra": "forbid"}


# --- DISPLAY SCHEMAS ---
class DisplayDefaultsConfig(BaseModel):
    """Default visualization parameters across all plot types."""

    n_bootstraps: int = Field(default=1000, ge=0)
    save: bool = True
    show: bool = False
    n_bins: int = Field(default=10, ge=1)
    strategy: Literal["uniform", "quantile", "spline"] = "uniform"
    model_config = {"extra": "allow"}


class DisplayConfig(BaseModel):
    """Configuration settings for pipeline evaluation graphics and themes."""

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
        """Process legacy flat keys (e.g., calibration_strategy) into defaults."""
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

    @field_validator("overrides", "outcome_overrides")
    @classmethod
    def validate_plot_override_keys(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Ensure plot override identifiers correspond to valid plot types."""
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
            "probability_distribution",
            "dist",
            "dca",
            "dca_curve",
            "strata_heatmap",
            "heatmap",
        }

        for key, val in v.items():
            if isinstance(val, dict) and any(
                isinstance(sub_v, dict) for sub_v in val.values()
            ):
                # Outcome overrides dictionary: outcome_name -> {plot_type: params}
                for plot_key in val.keys():
                    if plot_key.lower() not in valid_plots:
                        raise ValueError(
                            f"Unknown plot override type '{plot_key}'. "
                            f"Valid plot types are: {sorted(list(valid_plots))}"
                        )
            else:
                # Plot type overrides dictionary: plot_type -> params
                if key.lower() not in valid_plots:
                    raise ValueError(
                        f"Unknown plot override type '{key}'. "
                        f"Valid plot types are: {sorted(list(valid_plots))}"
                    )
        return v

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DisplayConfig":
        """Instantiate DisplayConfig from parsed TOML dictionary."""
        return cls.model_validate(data)


# --- GLOBAL MEDPIPE CONFIGURATION SCHEMA ---
class MedpipeConfig(BaseModel):
    """The master schema for a single-file configuration."""

    meta: MetaConfig
    data: DataConfig
    workflow: WorkflowConfig
    display: DisplayConfig | None = None

    # The default setup applied to all outcomes
    default_model: ModelSetup

    # Optional overrides keyed by outcome name
    outcome_overrides: dict[str, ModelSetup] = Field(default_factory=dict)

    # Dynamically generated during validation: fully resolved configurations per outcome
    resolved_models: dict[str, ModelSetup] = Field(default_factory=dict, init_var=False)

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def resolve_cascading_models(self) -> "MedpipeConfig":
        """Cascade default_model settings into outcome_overrides."""
        resolved = {}
        for outcome in self.data.outcomes:
            base_setup = self.default_model.model_dump()

            if outcome in self.outcome_overrides:
                override_setup = self.outcome_overrides[outcome].model_dump(
                    exclude_unset=True
                )

                # 1. Algorithm & Hyperparameters
                if (
                    "algorithm" in override_setup
                    and override_setup["algorithm"] != base_setup["algorithm"]
                ):
                    base_setup["algorithm"] = override_setup["algorithm"]
                    # Algorithm changed: replace hyperparameters entirely
                    # to prevent collisions
                    base_setup["hyperparameters"] = override_setup.get(
                        "hyperparameters", {}
                    )
                elif "hyperparameters" in override_setup:
                    # Same algorithm: deep-merge hyperparameters
                    base_setup["hyperparameters"].update(
                        override_setup["hyperparameters"]
                    )

                # 2. Recalibration
                if "recalibration" in override_setup:
                    override_recal = override_setup["recalibration"]
                    if override_recal is None:
                        # Explicitly disable recalibration for this outcome
                        base_setup["recalibration"] = None
                    elif base_setup.get("recalibration") is None:
                        base_setup["recalibration"] = override_recal
                    else:
                        if (
                            "method" in override_recal
                            and override_recal["method"]
                            != base_setup["recalibration"]["method"]
                        ):
                            base_setup["recalibration"] = override_recal
                        else:
                            if "recalibrate" in override_recal:
                                base_setup["recalibration"]["recalibrate"] = (
                                    override_recal["recalibrate"]
                                )
                            if "method" in override_recal:
                                base_setup["recalibration"]["method"] = override_recal[
                                    "method"
                                ]
                            if "hyperparameters" in override_recal:
                                base_setup["recalibration"]["hyperparameters"].update(
                                    override_recal["hyperparameters"]
                                )

            resolved[outcome] = ModelSetup(**base_setup)

        self.resolved_models = resolved
        return self

    @model_validator(mode="after")
    def validate_recalibration(self) -> "MedpipeConfig":
        """Check recalibration split is specified with recalibration method."""
        if self.default_model.recalibration:  # Recalibration is present
            if not self.workflow.validation.recalibration_split:
                expr = (
                    "Recalibration validation split must be "
                    "specified when a recalibration method is used"
                )
                raise ValueError(expr)
        return self

    @model_validator(mode="after")
    def validate_cross_validation(self) -> "MedpipeConfig":
        """Check that a cross-validation config is passed with correct
        run modes."""
        if self.meta.run_mode != "fast":
            if self.workflow.validation.cross_validation is None:
                expr = (
                    "Cross-validation parameters must be specified "
                    "when run_mode is not 'fast'"
                )
                raise ValueError(expr)
        return self

    @model_validator(mode="after")
    def validate_audit_and_eval_run_mode(self) -> "MedpipeConfig":
        """Check that audit and eval run modes have correct evaluation."""
        run_mode = self.meta.run_mode
        if run_mode == "audit" or run_mode == "eval":
            if self.workflow.evaluation.fairness is None:
                expr = (
                    "Evaluation fairness parameters must be specified "
                    "when run_mode is 'audit' or 'eval'"
                )
                raise ValueError(expr)
            if self.display is None:
                raise ValueError(
                    "Display parameters must be specified "
                    "when run_mode is 'audit' or 'eval'"
                )
        return self

    @model_validator(mode="after")
    def validate_outcome_overrides_exist_in_outcomes(self) -> "MedpipeConfig":
        """Ensures all outcome names in outcome_overrides are defined in data.outcomes."""
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
