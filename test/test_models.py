#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_models.py

Test functions for the models module
"""

import pathlib

import pytest

from pyrisk.models import create_model
from pyrisk.utils.io import read_toml_configuration

CWD = pathlib.Path.cwd()
DATA_DIR = CWD / "test/test_data/"


@pytest.mark.parametrize(
    "model_type, config_file",
    [
        ("hgb", "test_hgb.toml"),
        ("svm", "test_svm.toml"),
    ],
)
def test_create_model_success(model_type, config_file):
    config_params = read_toml_configuration(str(DATA_DIR / config_file))
    create_model(model_type, **config_params["parameters"])


@pytest.mark.parametrize(
    "model_type",
    [
        "hgb",
        "svm",
    ],
)
def test_create_model_incorrect_keyword_argument(model_type):
    with pytest.raises(TypeError):
        config_params = read_toml_configuration(
            str(DATA_DIR / "incorrect_model_config.toml")
        )
        create_model(model_type, **config_params["parameters"])


def test_create_model_not_str():
    with pytest.raises(TypeError):
        create_model(12)


def test_create_model_not_valid_model():
    with pytest.raises(ValueError):
        create_model("not_valid_model")
