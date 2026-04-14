#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_pipeline_Pipeline.py

Test functions for the Pipeline class.
"""

import pathlib

import pytest

from medpipe.pipeline.pipeline import Pipeline
from medpipe.utils.io import load_data_from_csv, read_toml_configuration

CWD = pathlib.Path.cwd()
DATA_DIR = CWD / "test/test_data/"


@pytest.mark.parametrize(
    "config_file",
    ["config/HGBc_config.toml"],
)
def test_create_Pipeline_success(config_file):
    general_config = read_toml_configuration(str(DATA_DIR / config_file))
    _ = Pipeline(general_config)


@pytest.mark.parametrize(
    "version_number",
    [
        "v0.1.1.2-a.1.0.1",
        "v0.1.0.2-a.2.1.1",
        "v0.1.1.2-a.3.2.1",
        "v0.1.0.2-a.4.0.1",
        "v0.1.0.2-a.5.1.1",
        "v0.1.2.2-a.1.1.1",
        "v0.1.3.2-a.2.1.1",
    ],
)
def test_run_Pipeline_HGBc_single_label(version_number):
    config_file = "config/HGBc_config.toml"
    data = load_data_from_csv("test/test_data/test_data.csv")
    general_config = read_toml_configuration(str(DATA_DIR / config_file))
    general_config["version"] = version_number
    pipeline = Pipeline(general_config)

    pipeline.preprocessor.fit(data)
    X_train, _ = pipeline.get_test_data(data)
    pipeline.run(X_train)


@pytest.mark.parametrize(
    "version_number",
    [
        "v0.1.1.2-b.1.0.1",
        "v0.1.0.2-b.2.1.1",
        "v0.1.1.2-b.3.2.1",
        "v0.1.0.2-b.4.0.1",
        "v0.1.0.2-b.5.1.1",
        "v0.1.2.2-b.5.1.1",
        "v0.1.3.2-b.2.1.1",
    ],
)
def test_run_Pipeline_HGBc_multi_label(version_number):
    config_file = "config/HGBc_config.toml"
    data = load_data_from_csv("test/test_data/test_data.csv")
    general_config = read_toml_configuration(str(DATA_DIR / config_file))
    general_config["version"] = version_number
    pipeline = Pipeline(general_config)

    pipeline.preprocessor.fit(data)
    X_train, _ = pipeline.get_test_data(data)
    pipeline.run(X_train)
