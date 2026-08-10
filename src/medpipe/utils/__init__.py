"""
medpipe.utils
-------------
Core utility functions and infrastructure for the Medpipe package.

Exposes configuration schemas, I/O handlers, centralized logging,
component registries, reproducibility management, and assertion utilities.
"""

from medpipe.utils.config import MedpipeConfig
from medpipe.utils.exceptions import (
    array_check,
    array_dim_check,
    file_checks,
    path_checks,
)
from medpipe.utils.io import DataLoaderRegistry, load_data, read_toml_configuration
from medpipe.utils.logger import add_file_handler, get_console_logger
from medpipe.utils.registry import BaseRegistry
from medpipe.utils.reproducibility import ArtifactManager

__all__ = [
    # Configuration
    "MedpipeConfig",
    # I/O utilities
    "load_data",
    "read_toml_configuration",
    # Logging
    "get_console_logger",
    "add_file_handler",
    # Component Registry
    "BaseRegistry",
    # Reproducibility & Artifacts
    "ArtifactManager",
    # Exception & Validation Checks
    "file_checks",
    "path_checks",
    "array_check",
    "array_dim_check",
]
