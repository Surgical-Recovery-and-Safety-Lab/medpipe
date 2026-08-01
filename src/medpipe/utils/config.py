"""
Configuration utilities module.

This module provides configuration schemas.

"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from medpipe.metrics.core import METRICS


# ==============================================================================
# CONFIGURATION SCHEMA (pydantic)
# ==============================================================================
# --- TOP-LEVEL MASTER SCHEMAS ---
class MetaConfig(BaseModel):
    """The master schema for the meta section of the configuration file."""

    project_name: str
    run_mode: Literal["fast", "eval", "cv", "audit"] = "audit"
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
    model_config = {"extra": "forbid"}

    @field_validator("path")
    @classmethod
    def validate_path(cls, file: str) -> str:
        """Validate that path is a points to a file."""
        data_path = Path(file)
        suffix = data_path.suffix
        if suffix == "":
            raise ValueError("path should be a file, but got no suffix")

        if suffix != ".csv":
            if suffix != ".parquet":
                raise ValueError(
                    f"path should be a .csv or .parquet file, but got suffix {suffix}"
                )
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
    strategy: Literal["random", "group", "search"]
    group_column: str | None = None
    n_splits: int | None = Field(default=None, ge=2)
    shuffle: bool | None = None
    random_state: int | None = Field(default=None, ge=0)
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
    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def validate_metrics(self) -> "MetricsConfig":
        """Validate input metrics."""
        for metric in self.metrics:
            if metric not in METRICS:
                expr = (
                    f"{metric} was not found in available metric "
                    f"list. Available metrics are {METRICS}"
                )
                raise ValueError(expr)

        return self


class CalibrationConfig(BaseModel):
    strategy: Literal["uniform", "quantile", "spline"] = Field(default="uniform")
    n_bootstraps: int = Field(default=200, ge=0)
    model_config = {"extra": "forbid"}


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
    calibration: CalibrationConfig | None = None
    fairness: FairnessConfig | None = None
    model_config = {"extra": "forbid"}


class WorkflowConfig(BaseModel):
    """The master schema for the workflow subconfiguration file."""

    preprocessing: PreprocessingConfig | None = None
    validation: ValidationSubConfig
    evaluation: EvaluationSubConfig

    model_config = {"extra": "forbid"}


# --- MODEL SCHEMAS ---
class RecalibrationConfig(BaseModel):
    method: Literal["isotonic", "sigmoid", "temperature"]
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    model_config = {"extra": "forbid"}


class ModelSetup(BaseModel):
    """Configuration for a model, recalibrator, and their hyperparameters."""

    algorithm: str
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    recalibration: RecalibrationConfig | None = None
    model_config = {"extra": "forbid"}


# --- GLOBAL MEDPIPE CONFIGURATION SCHEMA ---
class MedpipeConfig(BaseModel):
    """The master schema for a single-file configuration."""

    meta: MetaConfig
    data: DataConfig
    workflow: WorkflowConfig

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
            # Start with a copy of the default model setup
            base_setup = self.default_model.model_dump()

            # If the user provided an override for this outcome, update the base
            if outcome in self.outcome_overrides:
                override_setup = self.outcome_overrides[outcome].model_dump(
                    exclude_unset=True
                )

                # Update base algorithm
                if "algorithm" in override_setup:
                    base_setup["algorithm"] = override_setup["algorithm"]

                # Merge predictor hyperparameters
                if "hyperparameters" in override_setup:
                    base_setup["hyperparameters"].update(
                        override_setup["hyperparameters"]
                    )

                # Handle recalibration merge
                if (
                    "recalibration" in override_setup
                    and override_setup["recalibration"]
                ):
                    if base_setup.get("recalibration"):
                        # Both have recalibration, do a deep merge
                        if "method" in override_setup["recalibration"]:
                            base_setup["recalibration"]["method"] = override_setup[
                                "recalibration"
                            ]["method"]
                        if "hyperparameters" in override_setup["recalibration"]:
                            base_setup["recalibration"]["hyperparameters"].update(
                                override_setup["recalibration"]["hyperparameters"]
                            )
                    else:
                        # Base had no recalibration, overwrite entirely
                        base_setup["recalibration"] = override_setup["recalibration"]

            # Save the fully resolved configuration for this outcome
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
    def validate_evaluation(self) -> "MedpipeConfig":
        """Check that audit and eval run modes have correct evaluation."""
        run_mode = self.meta.run_mode
        if run_mode == "audit" or run_mode == "eval":
            if self.workflow.evaluation.calibration is None:
                expr = (
                    "Evaluation calibration parameters must be specified "
                    "when run_mode is 'audit' or 'eval'"
                )
                raise ValueError(expr)
            if self.workflow.evaluation.fairness is None:
                expr = (
                    "Evaluation fairness parameters must be specified "
                    "when run_mode is 'audit' or 'eval'"
                )
                raise ValueError(expr)
        return self

    @model_validator(mode="after")
    def validate_search_cv(self) -> "MedpipeConfig":
        """Check that at least one hyperparamters is a list
        with search cv strategy."""
        workflow_cv_cfg = self.workflow.validation.cross_validation
        if workflow_cv_cfg is not None and workflow_cv_cfg.strategy == "search":
            for outcome, model_setup in self.resolved_models.items():
                hyperparams = model_setup.hyperparameters

                # Check if any value in the hyperparameters dict is a list
                has_list_hyperparam = any(
                    isinstance(v, list) for v in hyperparams.values()
                )

                if not has_list_hyperparam:
                    raise ValueError(
                        f"Cross-validation strategy is set to 'search', but outcome "
                        f"'{outcome}' has no hyperparameter defined as a list for "
                        "GridSearchCV."
                    )

        return self

    @model_validator(mode="after")
    def validate_outcome_overrides_exist_in_outcomes(self) -> "MedpipeConfig":
        """Ensures all outcome names in outcome_overrides are defined in data.outcomes."""
        if self.outcome_overrides and self.data and self.data.outcomes:
            valid_outcomes = set(self.data.outcomes)
            for override_outcome in self.outcome_overrides:
                if override_outcome not in valid_outcomes:
                    raise ValueError(
                        f"Outcome override '{override_outcome}' is not present "
                        f"in data.outcomes: {self.data.outcomes}"
                    )
        return self
