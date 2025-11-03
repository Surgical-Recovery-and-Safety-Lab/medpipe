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
    Setups a logger for logging exceptions.

    Parameters
    ----------
    script_name
        Name of the script to log exceptions from.
    log_path
        Path to the folder to store the log file.

    Returns
    -------
    logger : logging.Logger
        Logger object.

    Raises
    ______
    TypeError
        If script_name or log_path are not a str.
    FileNotFoundError
        If log_path do not exist.
    NotADirectoryError
        If log_path is not a directory.

    """
    if type(script_name) is not str:
        raise TypeError(f"{script_name} should be a string")

    try:
        pyrisk.utils.exceptions.path_checks(log_path)

    except (FileNotFoundError, TypeError, NotADirectoryError):
        raise

    log_file = f"{log_path}/{script_name}.log"  # Location to save the log

    # Change the log file destination
    logger_config_dict = LOGGING_CONFIG
    logger_config_dict["handlers"]["file"]["filename"] = log_file

    logging.config.dictConfig(logger_config_dict)  # Configure logger

    return logging.getLogger(script_name)
