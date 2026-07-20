"""
Sample functions module.

This module provides data sampling functions.

Functions:
- sample_data: Samples the data using the specified strategy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Type

import imblearn

from medpipe.utils.config import SamplingConfig

if TYPE_CHECKING:
    import numpy.typing as npt
    import pandas as pd


def sample_data(
    X: pd.DataFrame, y: npt.NDArray, config: SamplingConfig | None
) -> tuple[pd.DataFrame, npt.NDArray]:
    """
    Sample the data based on the configuration strategy.

    Available strategies are found in imblearn.over_sampling
    and imblearn.under_sampling modules.

    Parameters
    ----------
    X : pd.DataFrame
        Data to resample.
    y : npt.NDArray
        Labels to resample.
    config : SamplingConfig | None
        Configuration to read parameters from or None.

    Returns
    -------
    X_resampled : pd.DataFrame
        Resampled data or X if sampling configuration is None.
    y_resampled : npt.NDArray
        Resampled labels or y if sampling configuration is None.

    """
    if config is None:
        return X, y
    assert config.strategy

    sampler = _check_strategy(config.strategy)

    # Instantiate with random seed anchoring + extra parameters passed in config
    kwargs = config.model_dump()
    kwargs.pop("strategy")
    sampler = sampler(**kwargs)

    # Resample both matrices simultaneously
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    return X_resampled, y_resampled


def _check_strategy(strategy: str) -> Type:
    """
    Internal function that checks if the strategy type is correct.

    Currently checks the imblearn.under_sampling and
    imblearn.over_sampling modules.

    Parameters
    ----------
    strategy : str
        Strategy to check.

    Returns
    -------
    sampler : Type
        Sampler class.

    Raises
    ------
    ValueError
        If the strategy type is not a valid class in one of the modules.

    """
    if hasattr(imblearn.over_sampling, strategy):
        return getattr(imblearn.over_sampling, strategy)

    if hasattr(imblearn.under_sampling, strategy):
        return getattr(imblearn.under_sampling, strategy)

    raise ValueError(
        f"{strategy} is not found in imblearn.under_sampling, or "
        "imblearn.over_sampling modules, please check that the strategy matches"
    )
