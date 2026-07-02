"""
Medpipe typings.

This module provides special types defined for the medpipe package.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal, TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd

if TYPE_CHECKING:
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
Labels: TypeAlias = SingleClassLabels | MultiClassLabels
PredData: TypeAlias = pd.DataFrame | npt.NDArray  # Predictor class data

# Define a generic config type as a union
Config: TypeAlias = (
    "MedpipeConfig | TopLevelConfig | DataConfig | WorkflowConfig | HyperparameterConfig"
)
