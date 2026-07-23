"""
Medpipe typings.

This module provides special types defined for the medpipe package.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal, TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.sparse import csr_array, csr_matrix

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
TransformedData: TypeAlias = pd.DataFrame | npt.NDArray | csr_matrix | csr_array

# Define a generic config type as a union
Config: TypeAlias = (
    "MedpipeConfig | TopLevelConfig | DataConfig | WorkflowConfig | HyperparameterConfig"
)
