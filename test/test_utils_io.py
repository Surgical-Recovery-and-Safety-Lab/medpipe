#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_utils_io.py

Test functions for the utils.io module
"""
import pathlib
import tomllib

import pytest

from pyrisk.utils.io import (
    load_data_from_csv,
    read_toml_configuration,
    save_data_to_csv,
)

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


def test_save_data_to_csv_success(tmp_path):
    save_path = str(tmp_path / "save_data.csv")
    save_data_to_csv(DF, save_path)


@pytest.mark.parametrize(
    "df, save_path",
    [
        (DF, 12),
        (12, CSV_PATH),
    ],
)
def test_save_data_to_csv_wrong_types(df, save_path):
    with pytest.raises(TypeError):
        save_data_to_csv(df, save_path)


def test_save_data_to_csv_not_csv_file():
    with pytest.raises(ValueError):
        save_data_to_csv(DF, TXT_PATH)


def test_save_data_to_csv_not_a_file():
    with pytest.raises(IsADirectoryError):
        save_data_to_csv(DF, DATA_DIR)


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
