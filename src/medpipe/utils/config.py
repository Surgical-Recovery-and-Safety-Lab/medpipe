"""
Configuration utilities module.

This module provides helper functions for reading configuration files.

Functions:
- parse_version_number: Function that parses a version number.
- read_subconfiguration_file: Reads the contents of a configuration file
    from a path.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal, TypeAlias
from warnings import warn

from pydantic import BaseModel, Field, field_validator, model_validator

from medpipe.metrics.core import METRICS

from .exceptions import file_checks

# Define file specific types
SubConfig: TypeAlias = "DataConfig | HyperparameterConfig | WorkflowConfig"
SubConfigTypes: TypeAlias = Literal["data", "workflow", "hyperparameters"]

# ==============================================================================
# CONFIGURATION SCHEMA (pydantic)
# ==============================================================================
# --- TOP-LEVEL MASTER SCHEMAS ---


class MetaConfig(BaseModel):
    version: str
    project_name: str
    run_mode: Literal["fast", "cv", "audit"] = "audit"
    model_config = {"extra": "forbid"}

    @field_validator("version")
    @classmethod
    def validate_version_format(cls, v: str) -> str:
        """Validate version is formatted as vX.Y.Z"""
        try:
            v_splits = v.split(".")
            assert len(v_splits) == 3
        except:
            raise ValueError(f"Version should be formatted as vX.Y.Z, but got {v}")
        return v

    @field_validator("project_name")
    @classmethod
    def validate_project_name(cls, name: str) -> str:
        """Validate that project name is not empty."""
        if not name:
            raise ValueError("Project name should not be an empty string.")

        return name


class PathsConfig(BaseModel):
    config_dir: str
    model_dir: str
    figure_dir: str
    model_config = {"extra": "forbid"}

    @field_validator("config_dir", "model_dir", "figure_dir")
    @classmethod
    def validate_paths(cls, dir: str) -> str:
        """Validate that paths point to directories."""
        path = Path(dir).expanduser().resolve()

        if path.is_file():
            raise ValueError(f"{path} points to an existing file")

        path.mkdir(parents=True, exist_ok=True)
        return dir


class ModelConfig(BaseModel):
    algorithm: str
    model_config = {"extra": "forbid"}


class RecalibrationConfig(BaseModel):
    method: str | None = None
    model_config = {"extra": "forbid"}


class TopLevelConfig(BaseModel):
    """The master schema for the top-level configuration file."""

    meta: MetaConfig
    paths: PathsConfig
    model: ModelConfig
    recalibration: RecalibrationConfig | None = None
    model_config = {"extra": "forbid"}


# --- DATA SCHEMAS
class DataConfig(BaseModel):
    """The master schema for the data subconfiguration file."""

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


# --- WORKFLOW SCHEMAS
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
    group_column: str | None = None
    n_splits: int = Field(default=2, ge=2)
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

    preprocessing: PreprocessingConfig | None = None
    validation: ValidationSubConfig
    evaluation: EvaluationSubConfig

    model_config = {"extra": "forbid"}


# --- HYPERPARAMETERS SCHEMAS
class PredictorConfig(BaseModel):
    learning_rate: float = Field(default=0.1, gt=0)
    # Allow extra parameters to be passed as keyword argument to predictor
    model_config = {"extra": "allow"}


class RecalibratorConfig(BaseModel):
    # Allow extra parameters to be passed as keyword argument to recalibrator
    model_config = {"extra": "allow"}


class ModelHyperparamSubConfig(BaseModel):
    predictor: PredictorConfig
    recalibrator: RecalibratorConfig | None = None

    model_config = {"extra": "forbid"}


class HyperparameterConfig(BaseModel):
    """The master schema for the hyperparameter subconfiguration file."""

    hyperparameters: ModelHyperparamSubConfig
    model_config = {"extra": "forbid"}


class MedpipeConfig(BaseModel):
    """The master schema for a medpipe pipeline."""

    top_level: TopLevelConfig
    data: DataConfig
    workflow: WorkflowConfig
    hyperparameters: HyperparameterConfig
    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def validate_recalibration(self) -> "MedpipeConfig":
        """Check recalibration split is specified with recalibration method."""
        if self.top_level.recalibration:  # Recalibration is present
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
        if self.top_level.meta.run_mode != "fast":
            if self.workflow.validation.cross_validation is None:
                expr = (
                    "Cross-validation parameters must be specified "
                    "when run_mode is not 'fast'"
                )
                raise ValueError(expr)
        return self


# ==============================================================================
# CONFIGURATION FUNCTIONS
# ==============================================================================


# Define some constants
SUBCONFIG_REGISTRY: dict[SubConfigTypes, type[SubConfig]] = {
    "data": DataConfig,
    "workflow": WorkflowConfig,
    "hyperparameters": HyperparameterConfig,
}


def read_subconfiguration_file(path: str | Path, subtype: SubConfigTypes) -> SubConfig:
    """
    Reads the contents of a configuration file from a path.

    The contents are validated using the pydantic classes defined
    in _types.py.

    Parameters
    ----------
    path: str | Path
        Path to the configuration file.
    subtype: SubConfigTypes {"data", "workflow", "hyperparameters"}
        Subtype of the configuration being read.

    Returns
    -------
    config: SubConfig
        Subconfiguration dictionary.

    Raises
    ------
    TypeError
        If path is not a str or Path.
    FileNotFoundError
        If path does not exist.
    IsADirectoryError
        If path is not a file.
    ValueError
        If path it not a .toml file.
        If subtype is not in {"data", "workflow", "hyperparameters"}.
    tomllib.TOMLDecodeError
        If the file was not read properly.

    """
    if subtype not in SUBCONFIG_REGISTRY.keys():
        valid_options = list(SUBCONFIG_REGISTRY.keys())
        raise ValueError(
            f"Unexpected subtype {subtype}, expecting one of {valid_options}"
        )

    file_checks(path, ".toml")

    with open(path, "rb") as file:
        raw_config = tomllib.load(file)
    subtype_class = SUBCONFIG_REGISTRY[subtype]

    return subtype_class.model_validate(raw_config)


def parse_version_number(version: str) -> list[str]:
    """
    Parses a version number.

    Expecting a version number in the format vX.Y.Z, with
    X the data version,
    Y the workflow version,
    Z the hyperparameters version.

    Parameters
    ----------
    version : str
        Version number to parse.

    Returns
    -------
    v_list : list[str]
        List containing data, workflow, hyperparameters numbers.

    Raises
    ------
    TypeError
        If v_number is not a string.
    ValueError
        If v_number is an empty string.
        If v_number does not have 3 elements.
        If v_number has an empty element.

    Warns
    -----
    UserWarning
        If the version string has more than 3 elements.

    """
    if not isinstance(version, str):
        raise TypeError(f"Version should be a string, but got {type(version)}")

    if not version:
        raise ValueError(
            "Version is empty. Check the version number is formatted as vX.Y.Z"
        )
    v_to_parse = version
    if version[0] == "v":
        # Remove v prefix if present
        v_to_parse = version[1:]

    v_list = v_to_parse.split(".")

    # Safety checks
    v_len = len(v_list)

    if v_len < 3:
        raise ValueError(
            f"Expecting 3 values, but got {v_len}. "
            "Check the version number is formatted as vX.Y.Z"
        )
    elif v_len > 3:
        warn(f"Expecting 3 values, but got {v_len}. Everything after 3 is ignored.")

    else:  # Check that there are not empty elements
        for i, v in enumerate(v_list):
            if not v:
                raise ValueError(
                    f"Element {i} in version is empty. "
                    "Check the version number is formatted as vX.Y.Z"
                )

    return v_list[:3]
