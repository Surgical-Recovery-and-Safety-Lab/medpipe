"""
Medpipe typings.

This module provides special types defined for the medpipe package.

"""

from __future__ import annotations

from typing import Annotated, Literal, Sequence, TypeAlias, TypeVar

import numpy as np
import numpy.typing as npt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder, PowerTransformer, StandardScaler

# Define label types
SingleClassLabels: TypeAlias = Annotated[npt.NDArray[np.integer], Literal["N"]]
MultiClassLabels: TypeAlias = Annotated[npt.NDArray[np.integer], Literal["N", "C"]]
FProbas: TypeAlias = Annotated[npt.NDArray[np.number], Literal["C", "N", 2]]
PProbas: TypeAlias = Annotated[npt.NDArray[np.number], Literal["N", "C"]]
Labels: TypeAlias = SingleClassLabels | MultiClassLabels

# Define preprocessing types
PreprocessOpConfig: TypeAlias = dict[str, dict[str, list[str]]]
PreprocessOp: TypeAlias = OrdinalEncoder | StandardScaler | PowerTransformer

# Define metric types
CI: TypeAlias = tuple[npt.NDArray, npt.NDArray, npt.NDArray]
CIDict: TypeAlias = dict[str, CI]
MetricDict: TypeAlias = dict[int | str, dict[str, Sequence[float]]]

# Define model types
Classifier: TypeAlias = HistGradientBoostingClassifier
Regressor: TypeAlias = LogisticRegression | IsotonicRegression
Model: TypeAlias = Classifier | Regressor
R = TypeVar("R", bound=Regressor)  # Generic Type Variable for Regressors
