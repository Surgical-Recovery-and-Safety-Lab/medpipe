"""
Configuration utilities module.

This module provides helper functions for reading configuration files.

Functions:
- get_file_path: Gets a file path from a configuration dictionary.
- parse_version_number: Function that parses a version number.
- get_data_configuration: Chains data configuration to return full configuration.
"""

from pyrisk.utils.exceptions import file_checks
from pyrisk.utils.io import read_toml_configuration


def get_file_path(
    config_dict: dict, v_number: str = "", path_type: str = "io", exists: bool = True
) -> str:
    """
    Gets the path to a file from a configuration dictionary.

    Parameters
    ----------
    config_dict
        Dictionary from a loaded .TOML file.
    v_number : default: ""
        Version number.
    path_type : {"io", "db", "data"}, default: "io"
        Path type in the configuration file.
    exists : default: True
        Flag to indicate if the file should exists.

    Returns
    -------
    file_path : str
        Path to the file.

    """
    if type(config_dict) is not type(dict()):
        raise TypeError(
            f"config_dict should be a dictionary, but got {type(config_dict)}"
        )
    if type(path_type) is not type(""):
        raise TypeError(f"path_type should be a str, but got {type(path_type)}")

    if path_type not in ["io", "db", "data"]:
        raise ValueError(f"path_type should be io, db, or data, but got {path_type}")

    key = path_type + "_parameters"
    if key not in config_dict.keys():
        raise KeyError(f"config_dict should have a {key} key")

    parameters = config_dict[key]

    file_path = (
        parameters["dir"] + parameters["name"] + v_number + parameters["extension"]
    )

    # Run file checks before returning
    file_checks(file_path, parameters["extension"], exists=exists)
    return file_path


def split_version_number(v_number: str):
    """
    Splits a version number into the data and model version numbers.

    Expecting a version number in the format vX.Y.Z-nN, where X.Y.Z is the
    data version number and nN is the model version number.

    Parameters
    ----------
    v_number
        Data version number to split.

    Returns
    -------
    data_v_number : str
        Data version number in the format vX.Y.Z.
    model_v_number : str
        Model version number in the format vnN.

    Raises
    ------
    TypeError
        If v_number is not a string.
    ValueError
        If v_number format is incorrect.

    """
    if type(v_number) is not type(""):
        raise TypeError(f"v_number should be a string, but got {type(v_number)}")

    if "-" not in v_number:
        raise ValueError(
            f"Incorrect version number format, expecting vX.Y.Z-nN but got {v_number}"
        )

    data_v_number, model_v_number = v_number.split("-")  # Split at the - sign

    if data_v_number[0] != "v":
        # Add v in case version number does not have one
        data_v_number = "v" + data_v_number

    return data_v_number, "v" + model_v_number


def parse_data_version_number(v_number: str) -> list[str]:
    """
    Parses a data version number.

    Expecting a version number in the format vX.Y.Z, where X, Y, and Z
    are integers.

    Parameters
    ----------
    v_number
        Data version number to parse.

    Returns
    -------
    v_list : list[str]
        List containing [source, extraction, preprocessing] numbers.

    Raises
    ------
    TypeError
        If v_number is not a string.

    """
    if type(v_number) is not type(""):
        raise TypeError(f"v_number should be a string, but got {type(v_number)}")

    if v_number[0] == "v":
        # Remove v prefix if present
        v_number = v_number[1:]

    return v_number.split(".")


def parse_model_version_number(v_number: str) -> list[str]:
    """
    Parses a model version number.

    Expecting a version number in the format vxX, where x can be any
    letter and X any integer.

    Parameters
    ----------
    v_number
        Model version number to parse.

    Returns
    -------
    v_list : list[str]
        List containing [prediction x, model X] numbers.

    Raises
    ------
    TypeError
        If v_number is not a string.

    """
    if type(v_number) is not type(""):
        raise TypeError(f"v_number should be a string, but got {type(v_number)}")

    if v_number[0] == "v":
        # Remove v prefix if present
        v_number = v_number[1:]

    # Get all letters and digits in case multiple are present
    alpha = "".join([c for c in v_number if c.isalpha()])
    digit = "".join([c for c in v_number if c.isdigit()])

    return [alpha, digit]


def get_configuration(parameters: dict, v_number: str, join_token: str = ".") -> dict:
    """
    Gets the configuration by chaining .toml configurations.

    Parameters
    ----------
    parameters
        Parameters for the configuration chaining.
    v_number
        Version number of the data to recuperate.
    join_token : default: "."
        Token used to join the version numbers. For data configurations use "."
        and for model configurations use "".

    Returns
    -------
    config_dict : dict
        Configuration parameters dictionary.

    Raises
    ------
    TypeError
        If parameters is not a dict.
    ValueError
        If join_token is not "." or "".

    """
    if type(parameters) is not type(dict()):
        raise TypeError(f"parameters should be a dict, but got f{type(parameters)}")

    match join_token:
        case ".":
            v_list = parse_data_version_number(
                v_number
            )  # Parse data version number to identify path

        case "":
            v_list = parse_model_version_number(
                v_number
            )  # Parse model version number to identify path
        case _:
            raise ValueError(f"join_token should be . or '', but got {join_token}")

    config_dict = {}  # Create empty configuration dictionary
    path = parameters["dir"]

    for i in range(len(v_list)):
        file_path = (
            path
            + parameters["name"]
            + f"v{join_token.join(v_list[:i+1])}"
            + parameters["extension"]
        )
        config_dict.update(read_toml_configuration(file_path))

        if i == len(v_list) - 1:
            # Exit early to avoid out of range error
            break

        # Update file path with next version folder
        path += f"v{v_list[i+1]}/"

    return config_dict
