"""
medpipe.utils.logger
--------------------

Provides centralized logging configuration for the medpipe package.
Supports routing high-level logs to the console and detailed debugging
logs to specific experiment artifact directories.
"""

import logging
import sys
from pathlib import Path
from typing import Union


def get_console_logger(
    name: str = "medpipe", level: int = logging.INFO
) -> logging.Logger:
    """Initialize and return the base console logger for the package.

    This function configures a StreamHandler to output logs to standard output
    with a clean, human-readable format. If the logger already has handlers,
    it avoids duplicating them.

    Parameters
    ----------
    name : str, default="medpipe"
        The name of the logger.
    level : int, default=logging.INFO
        The logging level threshold for the console.

    Returns
    -------
    logging.Logger
        Configured standard library Logger instance.

    """
    # Always ensure the root package logger ("medpipe") has the console handler
    root_logger = logging.getLogger("medpipe")
    root_logger.setLevel(logging.DEBUG)  # Base logger captures all thresholds

    if not root_logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    return logging.getLogger(name)


def add_file_handler(
    logger: logging.Logger,
    log_dir: Union[str, Path],
    filename: str = "execution.log",
    level: int = logging.DEBUG,
) -> None:
    """Attach a FileHandler to an existing logger to archive detailed run logs.

    Parameters
    ----------
    logger : logging.Logger
        The logger instance to attach the handler to.
    log_dir : str or Path
        The directory where the log file should be saved (e.g., an artifact folder).
    filename : str, default="execution.log"
        The name of the log file.
    level : int, default=logging.DEBUG
        The logging level threshold for the file output.

    """
    # Resolve top-level parent logger ("medpipe") regardless of sub-logger name
    top_logger_name = logger.name.split(".")[0]
    target_logger = logging.getLogger(top_logger_name)

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    file_path = log_path / filename

    # Avoid duplicate file handlers if initialized multiple times
    for handler in target_logger.handlers:
        if (
            isinstance(handler, logging.FileHandler)
            and Path(handler.baseFilename) == file_path.resolve()
        ):
            return

    file_handler = logging.FileHandler(file_path, mode="w", encoding="utf-8")
    file_handler.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(module)s:%(lineno)d | %(message)s"
    )
    file_handler.setFormatter(formatter)
    target_logger.addHandler(file_handler)
