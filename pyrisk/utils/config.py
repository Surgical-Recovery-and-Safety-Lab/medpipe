"""
Configuration utilities module.

This module provides helper functions for reading configuration files.

Functions:
- get_file_path: Gets a file path from a configuration dictionary.
"""

from pyrisk.utils.exceptions import file_checks


def get_file_path(
    config_dict: dict, path_type: str = "io", suffix: str = "", version: bool = True
) -> str:
    """
    Gets the path to a file from a configuration dictionary.

    Parameters
    ----------
    config_dict
        Dictionary from a loaded .TOML file.
    path_type : {"io", "db", "data"}, default: "io"
        Path type in the configuration file.
    suffix : default, ""
        Additional suffix for the path.
    version : default: True
        Flag to add the version number if True.

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
    if type(suffix) is not type(""):
        raise TypeError(f"suffix should be a str, but got {type(suffix)}")

    if path_type not in ["io", "db", "data"]:
        raise ValueError(f"path_type should be io, db, or data, but got {path_type}")

    key = path_type + "_parameters"
    if key not in config_dict.keys():
        raise KeyError(f"config_dict should have a {key} key")

    parameters = config_dict[key]

    if version:
        file_path = (
            parameters["dir"]
            + parameters["name"]
            + config_dict["version"]
            + suffix
            + parameters["extension"]
        )

    else:
        file_path = (
            parameters["dir"] + parameters["name"] + suffix + parameters["extension"]
        )

    # Run file checks before returning
    file_checks(file_path, parameters["extension"])
    return file_path
