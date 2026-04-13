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

from copy import deepcopy
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
    if type(data) is not type(pd.DataFrame()):
        raise TypeError(f"data should be a pd.DataFrame, but got {type(data)}")

    # Create a copy of data to work on
    processed_data = data

    for column in data.select_dtypes(include=["object"]).columns:
        processed_data[column] = data[column].astype("category")

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
    if type(data) is not type(pd.DataFrame()):
        raise TypeError(f"data should be a pd.DataFrame, but got {type(data)}")

    # Operation dictionary to store fitted operations
    data_copy = deepcopy(data)
    operation_dict = dict()

    for preprocess in preprocessing_dict.keys():
        features = preprocessing_dict[preprocess]["feature_list"]

        if type(features) is not type([]):
            raise TypeError(f"features should be a list, but got {type(features)}")

        if type(features[0]) is not type(""):
            raise TypeError(
                f"features should be a list(str), but got list({type(features[0])}"
            )

        match preprocess:
            case "ordinal_encoder":
                operation_dict[preprocess] = OrdinalEncoder().fit(data_copy[features])
                data_copy[features] = operation_dict[preprocess].transform(
                    data_copy[features]
                )
            case "standardise":
                operation_dict[preprocess] = StandardScaler().fit(data_copy[features])
                data_copy[features] = operation_dict[preprocess].transform(
                    data_copy[features]
                )
            case "power_transform":
                operation_dict[preprocess] = PowerTransformer().fit(data_copy[features])
                data_copy[features] = operation_dict[preprocess].transform(
                    data_copy[features]
                )
            case "bin":
                operation_dict[preprocess] = "bin"
            case _:
                raise ValueError(f"{preprocess} invalid preprocessing function")

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
    binned_data = np.ceil(data)
    binned_data[binned_data > 4] = 4
    return binned_data
