"""
Medpipe typings.

This module provides special types defined for the medpipe package.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal, Sequence, TypeAlias, TypeVar

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder, PowerTransformer, StandardScaler

if TYPE_CHECKING:
    from medpipe.models.calibrators import IsotonicCalibrator, LogisticCalibrator
    from medpipe.models.predictors import HGBClassifier
    from medpipe.utils.config import (
        DataConfig,
        HyperparameterConfig,
        MedpipeConfig,
        TopLevelConfig,
        WorkflowConfig,
    )

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

# Define a generic config type as a union
Config: TypeAlias = (
    "MedpipeConfig | TopLevelConfig | DataConfig | WorkflowConfig | HyperparameterConfig"
)
