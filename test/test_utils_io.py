#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_utils_io.py

Test functions for the utils.io module
"""
import pathlib
import tomllib

import pytest

from pyrisk.utils.io import load_data_from_csv, read_toml_configuration

CWD = pathlib.Path.cwd()
DATA_DIR = str(CWD / "test/test_data/")
TXT_PATH = str(CWD / DATA_DIR / "test_text.txt")
CSV_PATH = str(CWD / DATA_DIR / "test_data.csv")
DF = load_data_from_csv(CSV_PATH)


def test_load_data_from_csv_success():
    load_data_from_csv(CSV_PATH)


def test_load_data_from_csv_not_str():
    with pytest.raises(TypeError):
        load_data_from_csv(12)


def test_load_data_from_csv_not_csv_file():
    with pytest.raises(ValueError):
        load_data_from_csv(TXT_PATH)


def test_load_data_from_csv_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_data_from_csv("not_a_file.csv")


def test_load_data_from_csv_not_a_file():
    with pytest.raises(IsADirectoryError):
        load_data_from_csv(DATA_DIR)


def test_read_toml_configuration_success():
    data_path = str(CWD / "test/test_data/test_config.toml")
    read_toml_configuration(data_path)


def test_read_toml_configuration_not_str():
    with pytest.raises(TypeError):
        read_toml_configuration(12)


def test_read_toml_configuration_not_toml_file():
    with pytest.raises(ValueError):
        read_toml_configuration(TXT_PATH)


def test_read_toml_configuration_incorrect_toml_file():
    with pytest.raises(tomllib.TOMLDecodeError):
        data_path = str(CWD / "test/test_data/not_a_good_toml.toml")
        read_toml_configuration(data_path)


def test_read_toml_configuration_file_not_found():
    with pytest.raises(FileNotFoundError):
        read_toml_configuration("not_a_file.toml")


def test_read_toml_configuration_not_a_file():
    with pytest.raises(IsADirectoryError):
        read_toml_configuration(DATA_DIR)
