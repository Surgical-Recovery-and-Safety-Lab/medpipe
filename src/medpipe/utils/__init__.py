"""
medpipe.utils
-------------
Core utility functions and infrastructure for the Medpipe package.

Exposes configuration schemas, I/O handlers, centralized logging,
component registries, reproducibility management, and validation utilities.
"""

from medpipe.utils.config import MedpipeConfig
from medpipe.utils.io import DataLoaderRegistry, load_data, read_toml_configuration
from medpipe.utils.logger import add_file_handler, get_console_logger
from medpipe.utils.registry import BaseRegistry
from medpipe.utils.reproducibility import ArtifactManager
from medpipe.utils.validation import file_checks, path_checks

__all__ = [  # noqa: RUF022 (grouped by category, not alphabetical)
    # Configuration
    "MedpipeConfig",
    # I/O utilities
    "load_data",
    "read_toml_configuration",
    "DataLoaderRegistry",
    # Logging
    "get_console_logger",
    "add_file_handler",
    # Component Registry
    "BaseRegistry",
    # Reproducibility & Artifacts
    "ArtifactManager",
    # Validation Checks
    "file_checks",
    "path_checks",
]
