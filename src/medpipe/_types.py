"""
Medpipe typings.

This module provides special types defined for the medpipe package.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal, Sequence, TypeAlias, TypeVar

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder, PowerTransformer, StandardScaler

if TYPE_CHECKING:
    from medpipe.data.sampler import VALID_SAMPLER_FN
    from medpipe.data.weighting import VALID_WEIGHTING_FN
    from medpipe.models.calibrators import IsotonicCalibrator, LogisticCalibrator
    from medpipe.models.predictors import HGBClassifier

# ==============================================================================
# CORE PIPELINE & DATA TYPES
# ==============================================================================

# Define data and label types
SingleClassLabels: TypeAlias = Annotated[npt.NDArray[np.integer], Literal["N"]]
MultiClassLabels: TypeAlias = Annotated[npt.NDArray[np.integer], Literal["N", "C"]]
FullProba: TypeAlias = Annotated[npt.NDArray[np.number], Literal["C", "N", 2]]
PosProba: TypeAlias = Annotated[npt.NDArray[np.number], Literal["N", "C"]]
Labels: TypeAlias = SingleClassLabels | MultiClassLabels
PredData: TypeAlias = pd.DataFrame | npt.NDArray  # Predictor class data
Data: TypeAlias = PredData | PosProba  # Pipeline class data

# Define preprocessing types
PreprocessOpConfig: TypeAlias = dict[str, dict[str, list[str]]]
PreprocessOp: TypeAlias = OrdinalEncoder | StandardScaler | PowerTransformer

# Define metric types
CI: TypeAlias = tuple[npt.NDArray, npt.NDArray, npt.NDArray]
CIDict: TypeAlias = dict[str, CI]
MetricDict: TypeAlias = dict[str, list[float]]
ModelMetrics: TypeAlias = dict[str, dict[str, Sequence[float]]]

# Define model types
Classifier: TypeAlias = HistGradientBoostingClassifier
Predictor: TypeAlias = "HGBClassifier"
Regressor: TypeAlias = LogisticRegression | IsotonicRegression
Calibrator: TypeAlias = "IsotonicCalibrator | LogisticCalibrator"
Model: TypeAlias = Classifier | Regressor
R = TypeVar("R", bound=Regressor)  # Generic Type Variable for Regressors
C = TypeVar("C", bound=Classifier)  # Generic Type Variable for Classifiers


# ==============================================================================
# CONFIGURATION SCHEMA (pydantic)
# ==============================================================================
# --- TOP-LEVEL MASTER SCHEMAS ---


class MetaConfig(BaseModel):
    version: str
    project_name: str
    run_mode: Literal["fast", "cv", "audit"] = "audit"
    model_config = {"extra": "forbid"}


class PathsConfig(BaseModel):
    config_dir: str
    model_dir: str
    figure_dir: str
    model_config = {"extra": "forbid"}


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
    features: list[str]
    outcomes: list[str]


# --- WORKFLOW SCHEMAS
class PreprocessOperationConfig(BaseModel):
    name: str  # Matches the exact class name
    feature_list: list[str]  # The specific columns this transformer applies to
    model_config = {"extra": "allow"}


class PreprocessingConfig(BaseModel):
    preprocess: bool = True
    steps: list[PreprocessOperationConfig] = Field(default_factory=list)
    model_config = {"extra": "forbid"}


class TestSplitConfig(BaseModel):
    strategy: Literal["random", "group"] = "random"
    group_column: str | None = None
    values: list[str | int] | None = None
    test_size: float | None = Field(default=None, gt=0.0, le=1.0)
    drop_group_column: bool | None = None
    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def validate_strategy(self) -> "TestSplitConfig":
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


class RecalibrationSplitConfig(BaseModel):
    strategy: Literal["random", "group"] | None = None
    group_column: str | None = None
    values: list[str | int] | None = None
    recalibration_size: float | None = Field(default=None, gt=0.0, le=1.0)
    drop_group_column: bool | None = None
    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def validate_strategy(self) -> "RecalibrationSplitConfig":
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
    n_splits: int = Field(default=5, gt=0)
    shuffle: bool = True
    random_state: int = Field(default=42, gt=0)
    drop_group_column: bool = True
    model_config = {"extra": "forbid"}


class WorkflowConfig(BaseModel):
    """The master schema for the workflow subconfiguration file."""

    preprocessing: PreprocessingConfig
    test_split: TestSplitConfig = Field(..., alias="validation.test_split")
    calibration_split: RecalibrationSplitConfig | None = Field(
        default=None, alias="validation.calibration_split"
    )
    cross_validation: CrossValConfig | None = Field(
        default=None, alias="validation.cross_validation"
    )


# --- HYPERPARAMETERS SCHEMAS
class PredictorConfig(BaseModel):
    learning_rate: float
    # Allow extra parameters to be passed as keyword argument to predictor
    model_config = {"extra": "allow"}


class CalibratorConfig(BaseModel):
    # Allow extra parameters to be passed as keyword argument to calibrator
    model_config = {"extra": "allow"}


class WeightingConfig(BaseModel):
    weighting_fn: str | None = Field(default=None)
    model_config = {"extra": "forbid"}

    @field_validator("weighting_fn")
    @classmethod
    def validate_weighting_fn(cls, fn: str | None) -> str | None:
        if fn is not None and fn not in VALID_WEIGHTING_FN:
            raise ValueError(
                f"Unkown weighting function {fn}, should be one of {VALID_WEIGHTING_FN}"
            )
        return fn


class SamplingConfig(BaseModel):
    sampler_fn: str | None = Field(default=None)
    reduction_factor: float | None = Field(default=None, gt=0.0, lt=1.0)
    hard_percent: float | None = Field(default=None, gt=0.0, lt=1.0)
    model_config = {"extra": "forbid"}

    @field_validator("sampler_fn")
    @classmethod
    def validate_sampler_fn(cls, fn: str | None) -> str | None:
        if fn is not None and fn not in VALID_SAMPLER_FN:
            raise ValueError(
                f"Unkown weighting function {fn}, should be one of {VALID_SAMPLER_FN}"
            )
        return fn


class HyperparameterConfig(BaseModel):
    """The master schema for the hyperparameter subconfiguration file."""

    predictor: PredictorConfig = Field(..., alias="predictor.hyperparameters")
    calibrator: CalibratorConfig | None = Field(
        default=None, alias="calibrator.hyperparameters"
    )
    weighting: WeightingConfig | None = Field(default=None, alias="balancing.weighting")
    sampling: SamplingConfig | None = Field(default=None, alias="balancing.sampling")


class MedpipeConfig(BaseModel):
    """The master schema for a medpipe pipeline."""

    top_level: TopLevelConfig
    data: DataConfig
    workflow: WorkflowConfig
    hyperparameters: HyperparameterConfig


# Define a generic config type as a union
Config: TypeAlias = (
    MedpipeConfig | TopLevelConfig | DataConfig | WorkflowConfig | HyperparameterConfig
)
