"""
Logger functions.

This module provides helper functions for logging.

Functions:
- setup_logger: sets up a logger for a script.
"""

import logging
import logging.config

import pyrisk.utils.exceptions
import pyrisk.utils.io

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {"format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"}
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.FileHandler",
            "level": "ERROR",
            "formatter": "standard",
            "filename": "default_name.log",
            "mode": "a",
        },
    },
    "loggers": {
        "pyrisk": {
            "level": "DEBUG",
            "handlers": ["console", "file"],
            "propagate": False,
        }
    },
    "root": {"level": "WARNING", "handlers": ["console"]},
}


def setup_logger(script_name: str, log_path: str) -> logging.Logger:
    """
    Setups a logger based on a configuration file for logging exceptions.

    Parameters
    ----------
    script_name
        Name of the script to log exceptions from.
    config_file
        Path to the configuration file for the logger.
    log_path
        Path to the folder to store the log file.

    Returns
    -------
    logger : logging.Logger
        Logger object.

    Raises
    ______
    TypeError
        If config_file or log_path are not a str.
    FileNotFoundError
        If config_file or log_path do not exist.
    IsADirectoryError
        If config_file is not a file.
    ValueError
        If config_file extension is not .toml file.
    NotADirectoryError
        If log_path is not a directory.

    """
    try:
        config = pyrisk.utils.io.read_toml_configuration(config_file)

    except (FileNotFoundError, ValueError, IsADirectoryError, TypeError):
        raise

    try:
        pyrisk.utils.exceptions.path_checks(log_path)

    except (FileNotFoundError, TypeError, NotADirectoryError):
        raise

    log_file = f"{log_path}/{script_name}.log"

    # Modify the filename in file handler args
    if "handlers" in config and "file" in config["handlers"]:
        # Update args tuple to set new log file path
        args = list(config["handlers"]["file"]["args"])
        args[0] = log_file  # set new file path
        config["handlers"]["file"]["args"] = args

    logging.config.dictConfig(config)
    return logging.getLogger(script_name)
