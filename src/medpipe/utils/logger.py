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


class CompactProgressFilter(logging.Filter):
    """
    Filter that passes only high-level workflow milestone messages
    during compact mode.
    """

    # Tuple of message prefixes/substrings to allow through in "compact" mode
    PROGRESS_KEYWORDS = (
        "Initialising Medpipe end-to-end",
        "Succesfully loaded Medpipe",
        "Step 1/",
        "Step 2/",
        "Step 3/",
        "Step 4/",
        "Ingesting data from",
        "Starting model fitting across",
        "--- Starting ",
        "--- Finished ",
        "Full Medpipe pipeline execution finished",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        # Always allow WARNING, ERROR, and CRITICAL messages
        if record.levelno >= logging.WARNING:
            return True

        # Check if the log message contains any progress milestone keyword
        msg = record.getMessage()
        return any(keyword in msg for keyword in self.PROGRESS_KEYWORDS)


def set_verbosity(verbose: Union[bool, int, str] = "compact") -> None:
    """
    Globally configure the console log verbosity level and filters.

    Parameters
    ----------
    verbose : str, bool, or int, default="compact"
        - "quiet" / False / logging.WARNING : WARNING level only.
        - "compact" / 1                     : Filtered INFO level (progress lines only).
        - "info" / True / logging.INFO      : Full unfiltered INFO log output.
        - "debug" / 3 / logging.DEBUG       : Complete DEBUG log output.
    """
    root_logger = logging.getLogger("medpipe")

    # Standardize input mode string and handle explicit logging int levels
    if isinstance(verbose, bool):
        mode = "info" if verbose else "quiet"
    elif isinstance(verbose, int):
        mode = {
            logging.WARNING: "quiet",
            logging.INFO: "info",
            logging.DEBUG: "debug",
            0: "quiet",
            1: "compact",
            2: "info",
            3: "debug",
        }.get(verbose, "compact")
    else:
        mode = str(verbose).lower()

    # Resolve level & filter status
    use_compact_filter = False
    if mode in ("quiet", "warning"):
        level = logging.WARNING
    elif mode in ("compact", "progress"):
        level = logging.INFO
        use_compact_filter = True
    elif mode in ("info", "detailed"):
        level = logging.INFO
    elif mode == "debug":
        level = logging.DEBUG
    else:
        level = logging.INFO

    # Update StreamHandler level and filters on root_logger
    for handler in root_logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(
            handler, logging.FileHandler
        ):
            handler.setLevel(level)

            # Detach any existing CompactProgressFilter instances
            handler.filters = [
                f for f in handler.filters if not isinstance(f, CompactProgressFilter)
            ]

            if use_compact_filter:
                handler.addFilter(CompactProgressFilter())


def get_console_logger(
    name: str = "medpipe",
    verbose: Union[bool, int, str, None] = None,
) -> logging.Logger:
    """
    Initialize and return the base console logger for the package with
    configurable verbosity.

    Parameters
    ----------
    name : str, default="medpipe"
        The name of the sub-logger instance.
    verbose : str, bool, or int, default="compact"
        Console verbosity setting ("compact", "info", "debug", "quiet")

    Returns
    -------
    logging.Logger
        Configured standard library Logger instance.
    """
    root_logger = logging.getLogger("medpipe")
    root_logger.setLevel(
        logging.DEBUG
    )  # Base logger captures all thresholds for file logging

    has_console_handler = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in root_logger.handlers
    )

    if not has_console_handler:
        console_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # Apply log level & filter configuration to stream handler
    if verbose is not None:
        set_verbosity(verbose)

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
