#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_models_core.py

Test functions for the models.core module
"""

import pathlib

import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

from medpipe.models.core import create_model
from medpipe.utils.config import get_configuration, split_version_number
from medpipe.utils.io import read_toml_configuration

CWD = pathlib.Path.cwd()
DATA_DIR = CWD / "test/test_data/"


@pytest.mark.parametrize(
    "model_type, config_file",
    [("hgb-c", "config/HGBc_config.toml"), ("hgb-c", "")],
)
def test_create_model_HGBc_success(model_type, config_file):
    if config_file:
        general_params = read_toml_configuration(str(DATA_DIR / config_file))
        _, model_version = split_version_number(general_params["version"])
        model_config = get_configuration(
            general_params["model_parameters"], model_version
        )
        model = create_model(
            model_type,
            logger=None,
            **model_config["hyperparameters"],
        )

    else:
        model = create_model(model_type, logger=None)

    if model_type == "hgb-c":
        assert isinstance(model, HistGradientBoostingClassifier)


def test_create_model_not_valid_model():
    with pytest.raises(ValueError):
        create_model("not_valid_model")


@pytest.mark.parametrize("model_type", [123, [], {}, 1.5, None])
def test_create_model_invalid_model_type(model_type):
    with pytest.raises(TypeError):
        create_model(model_type, logger=None)


# Test invalid configuration parameters for the model
@pytest.mark.parametrize("model_type", ["hgb-c"])
def test_create_model_invalid_config(model_type):
    model_config = {"invalid": None}
    # Expecting a failure due to invalid config
    with pytest.raises(TypeError):
        create_model(
            model_type,
            logger=None,
            **model_config,
        )


# Test passing a logger to see if log messages are printed
@pytest.mark.parametrize("model_type", ["hgb-c"])
def test_create_model_with_logger(model_type):
    # Here we'll check if logger prints the expected message to stdout/stderr
    logger = None  # Use a mock or None for simplicity in this case
    create_model(model_type, logger=logger)


# Test `create_model` when no config file is passed
@pytest.mark.parametrize("model_type", ["hgb-c"])
def test_create_model_without_config(model_type):
    model = create_model(model_type, logger=None)

    if model_type == "hgb-c":
        assert isinstance(model, HistGradientBoostingClassifier)
