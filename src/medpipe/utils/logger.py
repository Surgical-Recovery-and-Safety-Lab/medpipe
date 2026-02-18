"""
Logger functions.

This module provides helper functions for logging.

Functions:
- setup_logger: sets up a logger for a script.
- exception_handler: Function that logs exceptions.
- print_message: Function that prints and logs an info message.
"""

import logging
import logging.config
import pathlib
import sys

import medpipe.utils.exceptions

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {"format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"},
    },
    "handlers": {
        "file": {
            "class": "logging.FileHandler",
            "level": "INFO",
            "formatter": "standard",
            "filename": "default_name.log",
            "mode": "w",
            "encoding": "utf8",
        },
    },
    "loggers": {
        "root": {
            "level": "INFO",
            "handlers": ["file"],
            "propagate": False,
        }
    },
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
        medpipe.utils.exceptions.path_checks(log_path)

    except (FileNotFoundError, TypeError, NotADirectoryError):
        raise

    # Change the log file destination
    logger_config_dict = LOGGING_CONFIG
    logger_config_dict["handlers"]["file"]["filename"] = str(
        pathlib.Path(log_path) / f"{script_name}.log"
    )

    logging.config.dictConfig(logger_config_dict)  # Configure logger

    return logging.getLogger(script_name)


def exception_handler(logger, log_path, log_config, script_name):
    """
    Handles exceptions and logs them.

    Parameters
    ----------
    logger : logging.Logger
        Logger object that logs the exception.
    log_path : str
        Path to the log file being used.
    log_config : dict
        Configuration parameters for the log messages.
    script_name : str
        Name of the script in which the error occurred.

    Returns
    -------
    None
        Nothing is returned.

    """
    logger.exception(log_config["log_message"] + f"{script_name}")
    sys.stderr.write(log_config["print_message"] + f"{log_path}/{script_name}.log\n")


def print_message(message, logger=None, script_name="") -> None:
    """
    Wrapper function to print message or log them.

    If the logger.level is less than 0, only log message.
    If the logger.level is greater than 0, log and print.
    If logger is None, only print.

    Parameters
    ----------
    message : str
        Message to print or log.
    logger : logging.Logger or None, default:None
        Logger to log message. If None message is printed to terminal.
    script_name : str, default: ""
        Script name to know where the message is coming from.

    Returns
    -------
    None
        Nothing is returned.

    """
    if logger:
        if logger.level >= 0:
            # Print to file and screen
            logger.info(f"{message} in {script_name}")
            sys.stdout.write(f"[INFO] {message}\n")
        elif logger.level < 0:
            # Don't print to screen
            logger.info(f"{message} in {script_name}")
    else:
        # Only print to screen
        print(f"[INFO] {message} in {script_name}")
