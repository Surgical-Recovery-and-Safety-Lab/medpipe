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

from medpipe.data.sampler import VALID_SAMPLER_FN
from medpipe.data.weighting import VALID_WEIGHTING_FN

from .exceptions import file_checks

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

    @field_validator("config_dir")
    @classmethod
    def validate_config_dir(cls, dir: str) -> str:
        """Validate that config dir is a path and directory."""
        path = Path(dir)
        if path.suffix != "":
            raise ValueError(
                f"config_dir should be a directory, but got suffix {path.suffix}"
            )
        return dir

    @field_validator("model_dir")
    @classmethod
    def validate_model_dir(cls, dir: str) -> str:
        """Validate that model dir is a path and directory."""
        path = Path(dir)
        if path.suffix != "":
            raise ValueError(
                f"model_dir should be a directory, but got suffix {path.suffix}"
            )
        return dir

    @field_validator("figure_dir")
    @classmethod
    def validate_figure_dir(cls, dir: str) -> str:
        """Validate that figure dir is a path and directory."""
        path = Path(dir)
        if path.suffix != "":
            raise ValueError(
                f"figure_dir should be a directory, but got suffix {path.suffix}"
            )
        return dir


class ModelConfig(BaseModel):
    algorithm: str
    model_config = {"extra": "forbid"}


class CalibrationConfig(BaseModel):
    method: Literal["isotonic", "logistic"]
    model_config = {"extra": "forbid"}


class TopLevelConfig(BaseModel):
    """The master schema for the top-level configuration file."""

    meta: MetaConfig
    paths: PathsConfig
    model: ModelConfig
    calibration: CalibrationConfig
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
    def validate_path(cls, dir: str) -> str:
        """Validate that figure dir is a path and directory."""
        data_path = Path(dir)
        if data_path.suffix != "":
            raise ValueError(
                f"path should be a directory, but got suffix {data_path.suffix}"
            )
        return dir

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
    feature_list: list[str]  # The specific columns this transformer applies to
    model_config = {"extra": "allow"}


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
    drop_group_column: bool | None = None
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
            elif type(self.drop_group_column) is not bool:
                raise ValueError(msg + "the drop flag to be specified")

        return self


class SplitRecalibrationConfig(BaseModel):
    strategy: Literal["random", "group"] | None = None
    group_column: str | None = None
    values: list[str | int] | None = None
    recalibration_size: float | None = Field(default=None, gt=0.0, lt=1.0)
    drop_group_column: bool | None = None
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
            elif type(self.drop_group_column) is not bool:
                raise ValueError(msg + "the drop flag to be specified")

        return self


class CrossValConfig(BaseModel):
    strategy: Literal["random", "group"] | None = None
    group_column: str | None = None
    n_splits: int = Field(default=5, ge=2)
    shuffle: bool = True
    random_state: int = Field(default=42, ge=0)
    drop_group_column: bool | None = True
    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def validate_strategy(self) -> "CrossValConfig":
        if self.strategy == "group" and not self.group_column:
            raise ValueError(
                "The group strategy requires a group column to be specified"
            )

        if self.strategy == "group" and not isinstance(self.drop_group_column, bool):
            raise ValueError(
                "The group strategy requires the drop flag to be specified"
            )

        return self


class ValidationSubConfig(BaseModel):
    test_split: SplitTestConfig
    recalibration_split: SplitRecalibrationConfig | None = None
    cross_validation: CrossValConfig | None = None

    model_config = {"extra": "forbid"}


class WorkflowConfig(BaseModel):
    """The master schema for the workflow subconfiguration file."""

    preprocessing: PreprocessingConfig
    validation: ValidationSubConfig

    model_config = {"extra": "forbid"}


# --- HYPERPARAMETERS SCHEMAS
class PredictorConfig(BaseModel):
    learning_rate: float = Field(default=0.1, gt=0)
    # Allow extra parameters to be passed as keyword argument to predictor
    model_config = {"extra": "allow"}


class CalibratorConfig(BaseModel):
    # Allow extra parameters to be passed as keyword argument to calibrator
    model_config = {"extra": "allow"}


class ModelHyperparamSubConfig(BaseModel):
    predictor: PredictorConfig
    calibrator: CalibratorConfig | None = None

    model_config = {"extra": "forbid"}


class WeightingConfig(BaseModel):
    weighting_fn: str | None = Field(default=None)
    model_config = {"extra": "forbid"}

    @field_validator("weighting_fn")
    @classmethod
    def validate_weighting_fn(cls, fn: str | None) -> str | None:
        if fn is not None and fn not in VALID_WEIGHTING_FN:
            raise ValueError(
                f"Unknown weighting function {fn} "
                f"should be one of {VALID_WEIGHTING_FN}"
            )
        return fn


class SamplingConfig(BaseModel):
    sampler_fn: str | None = Field(default=None)
    reduction_factor: float | None = Field(default=None, gt=0.0, lt=1.0)
    hard_percent: float | None = Field(default=None, gt=0.0, lt=1.0)
    model_config = {"extra": "forbid"}

    @field_validator("sampler_fn")
    @classmethod
    def validate_weighting_fn(cls, fn: str | None) -> str | None:
        if fn is not None and fn not in VALID_SAMPLER_FN:
            raise ValueError(
                f"Unknown weighting function {fn} "
                f"should be one of {VALID_SAMPLER_FN}"
            )
        return fn


class BalancingSubConfig(BaseModel):
    weighting: WeightingConfig | None = None
    sampling: SamplingConfig | None = None


class HyperparameterConfig(BaseModel):
    """The master schema for the hyperparameter subconfiguration file."""

    hyperparameters: ModelHyperparamSubConfig
    balancing: BalancingSubConfig | None = None


class MedpipeConfig(BaseModel):
    """The master schema for a medpipe pipeline."""

    top_level: TopLevelConfig
    data: DataConfig
    workflow: WorkflowConfig
    hyperparameters: HyperparameterConfig


# ==============================================================================
# CONFIGURATION FUNCTIONS
# ==============================================================================


# Define some constants
SUBCONFIG_REGISTRY: dict[SubConfigTypes, type[SubConfig]] = {
    "data": DataConfig,
    "workflow": WorkflowConfig,
    "hyperparameters": HyperparameterConfig,
}

# Define file specific types
SubConfig: TypeAlias = DataConfig | HyperparameterConfig | WorkflowConfig
SubConfigTypes: TypeAlias = Literal["data", "workflow", "hyperparameters"]


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
    tomllib.TOMLDecodeError
        If the file was not read properly.

    """
    if subtype not in SUBCONFIG_REGISTRY.keys():
        valid_options = list(SUBCONFIG_REGISTRY.keys())
        raise ValueError(
            f"Unexpected subtype {subtype}, expecting one of " f"{valid_options}"
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

    Warns
    -----
    UserWarning
        If the version string has more than 3 elements.

    """
    if not isinstance(version, str):
        raise TypeError(f"version should be a string, but got {type(version)}")

    v_to_parse = version
    if version[0] == "v":
        # Remove v prefix if present
        v_to_parse = version[1:]

    v_list = v_to_parse.split(".")

    # Safety checks
    v_len = len(v_list)
    if v_len < 3:
        raise ValueError(
            f"Expecting 3 values, but got {v_len}."
            "Check the version number is formatted as vX.Y.Z"
        )
    elif v_len > 3:
        warn(f"Expecting 3 values, but got {v_len}. Everything after 3 is ignored.")

    return v_list[:3]
