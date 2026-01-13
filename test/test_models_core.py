#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_models_core.py

Test functions for the models.core module
"""

import pathlib

import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.svm import SVC

from pyrisk.models.AIRiskNN import AIRiskNN
from pyrisk.models.core import create_model
from pyrisk.utils.config import get_configuration, split_version_number
from pyrisk.utils.io import read_toml_configuration

CWD = pathlib.Path.cwd()
DATA_DIR = CWD / "test/test_data/"


@pytest.mark.parametrize(
    "model_type, config_file",
    [
        ("nn", "config/NN_config.toml"),
    ],
)
def test_create_model_NN_success(model_type, config_file):
    general_params = read_toml_configuration(str(DATA_DIR / config_file))
    _, model_version = split_version_number(general_params["version"])
    model_config = get_configuration(general_params["model_parameters"], model_version)
    model = create_model(
        model_type,
        n_features=1,
        logger=None,
        **model_config["architecture"],
    )

    assert isinstance(model, AIRiskNN)


@pytest.mark.parametrize(
    "model_type, config_file",
    [("hgb", "config/HGB_config.toml"), ("hgb", ""), ("svm", "")],
)
def test_create_model_HGB_SVM_success(model_type, config_file):
    if config_file:
        general_params = read_toml_configuration(str(DATA_DIR / config_file))
        _, model_version = split_version_number(general_params["version"])
        model_config = get_configuration(
            general_params["model_parameters"], model_version
        )
        model = create_model(
            model_type,
            n_features=1,
            logger=None,
            **model_config["hyperparameters"],
        )

    else:
        model = create_model(model_type, n_features=1, logger=None)

    if model_type == "svm":
        assert isinstance(model, SVC)
    if model_type == "hgb":
        assert isinstance(model, HistGradientBoostingClassifier)


def test_create_model_not_valid_model():
    with pytest.raises(ValueError):
        create_model("not_valid_model")


@pytest.mark.parametrize("model_type", [123, [], {}, 1.5, None])
def test_create_model_invalid_model_type(model_type):
    with pytest.raises(TypeError):
        create_model(model_type, n_features=1, logger=None)


# Test invalid configuration parameters for the model
@pytest.mark.parametrize("model_type", ["hgb", "svm"])
def test_create_model_invalid_config(model_type):
    model_config = {"invalid": None}
    # Expecting a failure due to invalid config
    with pytest.raises(TypeError):
        create_model(
            model_type,
            n_features=-1,
            logger=None,
            **model_config,
        )


# Test invalid configuration parameters for the nn model
def test_create_model_nn_invalid_config():
    model_config = {"invalid": None}
    # Expecting a failure due to invalid config
    with pytest.raises(KeyError):
        create_model(
            "nn",
            n_features=10,
            logger=None,
            **model_config,
        )


# Test missing `n_features` for NN model
def test_create_model_nn_missing_n_features():
    with pytest.raises(ValueError):
        create_model(
            "nn",
            n_features=-1,  # This is an invalid value for `n_features`
            logger=None,
        )


# Test passing a logger to see if log messages are printed
@pytest.mark.parametrize("model_type", ["hgb", "svm"])
def test_create_model_with_logger(model_type):
    # Here we'll check if logger prints the expected message to stdout/stderr
    logger = None  # Use a mock or None for simplicity in this case
    create_model(model_type, n_features=10, logger=logger)


# Test `create_model` when no config file is passed
@pytest.mark.parametrize("model_type", ["hgb", "svm"])
def test_create_model_without_config(model_type):
    model = create_model(model_type, n_features=10, logger=None)

    if model_type == "hgb":
        assert isinstance(model, HistGradientBoostingClassifier)
    elif model_type == "svm":
        assert isinstance(model, SVC)
