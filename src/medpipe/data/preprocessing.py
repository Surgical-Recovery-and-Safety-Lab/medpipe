"""
Preprocessing functions module.

This module provides functions to preprocess data before training.

Functions:
- train_test_it: Creates a KFold iterator to split data into test and train sets.
- convert_object_to_categorical: Converts object columns to categoricals.
- fit_preprocess_operations: Fits processing operations to data.
- bin_score: Bins the M3 score into 5 categories.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, PowerTransformer, StandardScaler

from medpipe._types import PreprocessOp, PreprocessOpConfig

if TYPE_CHECKING:
    import numpy.typing as npt


def train_test_it(
    group_k_fold: bool = False, **kwargs: Any
) -> StratifiedKFold | GroupKFold:
    """
    Creates a KFold iterator to split data into test and train sets.

    Parameters
    ----------
    group_k_fold : bool, default: False
        If True, the data will be split using a group and a
        GroupKFold iterator is returned.
    **kwargs: Any
        Extra arguments for the StratifiedKFold or GroupKFold class.

    Returns
    -------
    kfold_it : StratifiedKFold | GroupKFold
        KFold iterator.

    Raises
    ------
    ValueError
        If n_splits is less than 2.

    """
    # Create the correct argument dict for StratifiedKFold
    args_dict = dict()
    for key, value in kwargs.items():
        match key:
            case "random_state":
                if value == -1:
                    value = None
                args_dict.update({key: value})

            case "shuffle":
                args_dict.update({key: value})

            case "n_splits":
                if value < 2:
                    raise ValueError(
                        f"n_splits should be greater than 2, but got {value}"
                    )

                args_dict.update({key: value})

    if not group_k_fold:
        kfold_it = StratifiedKFold(**args_dict)
    else:
        kfold_it = GroupKFold(**args_dict)

    return kfold_it


def convert_object_to_categorical(data: pd.DataFrame) -> pd.DataFrame:
    """
    Converts all object columns of a DataFrame to categoricals.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame to manipulate.

    Returns
    -------
    processed_data : pd.DataFrame
        Processed DataFrame.

    Raises
    ------
    TypeError
        If data is not a pd.DataFrame.

    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"data should be a pd.DataFrame, but got {type(data).__name__}")

    processed_data = data.copy()
    obj_cols = processed_data.select_dtypes(include=["object"]).columns

    # Vectorized conversion for all object columns at once
    if not obj_cols.empty:
        processed_data[obj_cols] = processed_data[obj_cols].astype("category")

    return processed_data


def fit_preprocess_operations(
    data: pd.DataFrame, preprocessing_dict: PreprocessOpConfig
) -> Mapping[str, PreprocessOp | str]:
    """
    Fits processing operations to data.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame to manipulate.
    preprocessing_dict : PreprocessOpConfig
        Dictionary of the operations and the features on which to operate.

    Returns
    -------
    operation_dict : Mapping[str, PreprocessOp | str]
        Dictionary of the different preprocessing objects.

    Raises
    ------
    TypeError
        If data is not a pd.DataFrame.
        If features is not a list[str].
    KeyError
        If a features is not a valid key.
    ValueError
        If preprocess is not a valid preprocessing function.

    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"data should be a pd.DataFrame, but got {type(data).__name__}")

    # Use shallow copy instead of deepcopy for performance
    data_working = data.copy()
    operation_dict: dict[str, PreprocessOp | str] = {}

    # Map names to classes for cleaner logic
    transformers = {
        "ordinal_encoder": OrdinalEncoder,
        "standardise": StandardScaler,
        "power_transform": PowerTransformer,
    }

    for op_name, config in preprocessing_dict.items():
        features = config.get("feature_list")

        if not isinstance(features, list):
            raise TypeError(
                f"features should be a list, but got {type(features).__name__}"
            )

        if features and not isinstance(features[0], str):
            raise TypeError(
                f"features must contain strings, got {type(features[0]).__name__}"
            )

        if op_name == "bin":
            operation_dict[op_name] = "bin"
            continue

        if op_name in transformers:
            # Instantiate, fit, and transform sequentially
            model = transformers[op_name]()
            operation_dict[op_name] = model.fit(data_working[features])

            # Update working data so subsequent fits see transformed values
            data_working[features] = model.transform(data_working[features])
        else:
            raise ValueError(f"{op_name} is an invalid preprocessing function")

    return operation_dict


def bin_score(data: npt.NDArray) -> npt.NDArray:
    """
    Bins the M3 score into 5 categories.

    Parameters
    ----------
    data : npt.NDArray
        M3 score data.

    Returns
    -------
    binned_data : npt.NDArray
        Binned data.

    """
    return np.clip(np.ceil(data), 0, 4).astype(int)
