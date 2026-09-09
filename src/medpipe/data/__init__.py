"""
medpipe.data
------------
Data processing, label extraction, dataset splitting, and transformation utilities.

Provides functions for label manipulation, train/test/recalibration splitting,
subgroup mask resolution, custom transformers, and preprocessor registry management.
"""

from medpipe.data.registry import PreprocessorRegistry
from medpipe.data.transformers import BoundedLogitTransformer
from medpipe.data.utils import (
    extract_labels,
    get_split_idx,
    resolve_subgroup_mask,
    split_data,
)

__all__ = [
    # Data manipulation & split utilities
    "extract_labels",
    "get_split_idx",
    "split_data",
    "resolve_subgroup_mask",
    # Preprocessor registry & transformers
    "PreprocessorRegistry",
    "BoundedLogitTransformer",
]
