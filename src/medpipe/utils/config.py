"""
Configuration utilities module.

This module provides helper functions for reading configuration files.

Functions:
- parse_version_number: Function that parses a version number.
- read_subconfiguration_file: Reads the contents of a configuration file
    from a path.
- validate_balancing: Checks if sampler and weighting functions are valid.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal, TypeAlias
from warnings import warn

from medpipe._types import (
    BalancingSubConfig,
    DataConfig,
    HyperparameterConfig,
    WorkflowConfig,
)
from medpipe.data.sampler import VALID_SAMPLER_FN
from medpipe.data.weighting import VALID_WEIGHTING_FN

from .exceptions import file_checks

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


def validate_balancing(config: BalancingSubConfig) -> None:
    """
    Checks if sampler and weighting functions are valid.

    Parameters
    ----------
    config: BalancingSubConfig
        Configuration dictionary to verify.

    Returns
    -------
    None
        Nothing is returned.

    Raises
    ------
    ValueError
        If the sampler function is invalid.
        If the weighting function is invalid.

    """
    if config.weighting and config.weighting.weighting_fn:
        if config.weighting.weighting_fn not in VALID_WEIGHTING_FN:
            raise ValueError(
                f"Unknown weighting function {config.weighting.weighting_fn} "
                f"should be one of {VALID_WEIGHTING_FN}"
            )

    if config.sampling and config.sampling.sampler_fn:
        if config.sampling.sampler_fn not in VALID_SAMPLER_FN:
            raise ValueError(
                f"Unknown sampling function {config.sampling.sampler_fn} "
                f"should be one of {VALID_SAMPLER_FN}"
            )
