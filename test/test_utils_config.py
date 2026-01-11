#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_utils_config.py

Test functions for the config submodule.
"""
import pathlib

import pytest

from pyrisk.utils.config import (
    get_configuration,
    get_file_path,
    parse_version_number,
    split_version_number,
)
from pyrisk.utils.io import read_toml_configuration

CWD = pathlib.Path.cwd()
DATA_DIR = str(CWD / "test/test_data/")


@pytest.mark.parametrize(
    "file_name, path_type, exists, expected_file_path",
    [
        (
            "config/HGB_config.toml",
            "io",
            False,
            "test/test_data/models/ai_risk_HGB-v0.1.1.3-a.1.1.1.pkl",
        ),
        (
            "config/NN_config.toml",
            "data",
            False,
            "test/test_data/config/data/ai_risk_data_v0.1.1.3-b.1.1.1.2.toml",
        ),
        (
            "config/AI_risk_data_config.toml",
            "data",
            True,
            "test/test_data/config/test_config_v0.2.toml",
        ),
    ],
)
def test_get_file_path_success(file_name, path_type, exists, expected_file_path):
    toml_path = str(CWD / DATA_DIR / file_name)
    general_config = read_toml_configuration(toml_path)
    file_path = get_file_path(
        general_config, general_config["version"], path_type, exists
    )
    assert file_path == expected_file_path


@pytest.mark.parametrize(
    "file_name, path_type, exists",
    [
        (
            "config/HGB_config.toml",
            "db",
            False,
        ),
    ],
)
def test_get_file_path_key_error(file_name, path_type, exists):
    with pytest.raises(KeyError):
        toml_path = str(CWD / DATA_DIR / file_name)
        general_config = read_toml_configuration(toml_path)
        get_file_path(general_config, general_config["version"], path_type, exists)


@pytest.mark.parametrize(
    "file_name, path_type, exists",
    [
        (
            "config/NN_config.toml",
            "invalid_path",
            False,
        ),
    ],
)
def test_get_file_path_value_error(file_name, path_type, exists):
    with pytest.raises(ValueError):
        toml_path = str(CWD / DATA_DIR / file_name)
        general_config = read_toml_configuration(toml_path)
        get_file_path(general_config, general_config["version"], path_type, exists)


@pytest.mark.parametrize(
    "config_dict, path_type",
    [
        (True, False),
        ([1], [2]),
        (12, 42),
        ("", {"key": 1}),
        (True, ""),
        ([1], ""),
        (12, ""),
        ("", ""),
        ({"key": 1}, False),
        ({"key": 1}, [2]),
        ({"key": 1}, 42),
        ({"key": 1}, {"key": 1}),
    ],
)
def test_get_file_path_type_error(config_dict, path_type):
    with pytest.raises(TypeError):
        get_file_path(config_dict, "v0.1", path_type)


@pytest.mark.parametrize(
    "file_name, path_type, exists",
    [
        (
            "config/HGB_config.toml",
            "io",
            True,
        ),
        (
            "config/NN_config.toml",
            "data",
            True,
        ),
    ],
)
def test_get_file_path_exists_error(file_name, path_type, exists):
    with pytest.raises(FileNotFoundError):
        toml_path = str(CWD / DATA_DIR / file_name)
        general_config = read_toml_configuration(toml_path)
        get_file_path(general_config, general_config["version"], path_type, exists)


# Test for split_version_number function
@pytest.mark.parametrize(
    "v_number, expected_data_v_number, expected_model_v_number",
    [
        ("v1.2.3-b.2.45", "v1.2.3", "vb.2.45"),
        ("v2.0.0-a.99", "v2.0.0", "va.99"),
        ("v10.11.12-5.31.1", "v10.11.12", "v5.31.1"),
    ],
)
def test_split_version_number_success(
    v_number, expected_data_v_number, expected_model_v_number
):
    data_v_number, model_v_number = split_version_number(v_number)
    assert data_v_number == expected_data_v_number
    assert model_v_number == expected_model_v_number


@pytest.mark.parametrize(
    "v_number",
    [
        123,  # Non-string type
        [1, 2, 3],
        "1.2.3",  # Missing model version part
    ],
)
def test_split_version_number_error(v_number):
    with pytest.raises((TypeError, ValueError)):
        split_version_number(v_number)


# Test for parse_version_number function
@pytest.mark.parametrize(
    "v_number, expected_parsed_version",
    [
        ("v1.2.3", ["1", "2", "3"]),
        ("10.11.12", ["10", "11", "12"]),
        ("va.0.1", ["a", "0", "1"]),
    ],
)
def test_parse_version_number_success(v_number, expected_parsed_version):
    parsed_version = parse_version_number(v_number)
    assert parsed_version == expected_parsed_version


@pytest.mark.parametrize(
    "v_number",
    [
        123,  # Non-string type
        [1, 2, 3],
    ],
)
def test_parse_version_number_error(v_number):
    with pytest.raises(TypeError):
        parse_version_number(v_number)


@pytest.mark.parametrize(
    "file_name, parameters, version",
    [
        (
            "config/HGB_config.toml",
            "data_parameters",
            "v0.1.1.3",
        ),
        (
            "config/HGB_config.toml",
            "model_parameters",
            "va.1.1.1",
        ),
        (
            "config/NN_config.toml",
            "model_parameters",
            "vb.1.1.1.2",
        ),
        (
            "config/NN_config.toml",
            "data_parameters",
            "v0.1.1.2",
        ),
    ],
)
def test_get_configuration_success(file_name, parameters, version):
    toml_path = str(CWD / DATA_DIR / file_name)
    general_config = read_toml_configuration(toml_path)
    get_configuration(general_config[parameters], version)


@pytest.mark.parametrize(
    "config_dict",
    [
        True,
        [1],
        12,
        "",
    ],
)
def test_get_configuration_type_error(config_dict):
    with pytest.raises(TypeError):
        get_configuration(config_dict, "v0.1")


@pytest.mark.parametrize(
    "v_number",
    [
        123,  # Non-string type
        [1, 2, 3],
    ],
)
def test_get_configuration_version_error(v_number):
    with pytest.raises(TypeError):
        get_configuration({"dir": 1}, v_number)
