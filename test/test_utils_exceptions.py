#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_utils_exceptions.py

Test functions for the utils.exceptions module
"""
import pathlib

import pytest

from pyrisk.utils.exceptions import file_checks, path_checks

CWD = pathlib.Path.cwd()
DATA_DIR = str(CWD / "test/test_data/")


@pytest.mark.parametrize(
    "file_name, extension",
    [
        ("test_text.txt", ".txt"),
        ("test_data.csv", ".csv"),
        ("test_config.toml", ".toml"),
    ],
)
def test_file_checks_success(file_name, extension):
    file_path = str(CWD / DATA_DIR / file_name)
    file_checks(file_path, extension)


def test_file_checks_not_str():
    with pytest.raises(TypeError):
        file_checks(12, ".txt")


@pytest.mark.parametrize(
    "file_name, extension",
    [
        ("test_text.txt", ".csv"),
        ("test_data.csv", ".toml"),
        ("test_config.toml", ".txt"),
    ],
)
def test_file_checks_not_extension_file(file_name, extension):
    with pytest.raises(ValueError):
        file_path = str(CWD / DATA_DIR / file_name)
        file_checks(file_path, extension)


@pytest.mark.parametrize(
    "file_name, extension",
    [
        ("not_a_file.txt", ".txt"),
        ("not_a_file.csv", ".csv"),
        ("not_a_file.toml", ".toml"),
    ],
)
def test_file_checks_file_not_found(file_name, extension):
    with pytest.raises(FileNotFoundError):
        file_checks(file_name, extension)


@pytest.mark.parametrize(
    "extension",
    [
        (".txt"),
        (".csv"),
        (".toml"),
    ],
)
def test_file_checks_not_a_file(extension):
    with pytest.raises(IsADirectoryError):
        file_checks(DATA_DIR, extension)


def test_path_checks_success():
    path_checks(DATA_DIR)


def test_path_checks_not_str():
    with pytest.raises(TypeError):
        path_checks(12)


def test_path_checks_dir_not_found():
    with pytest.raises(FileNotFoundError):
        data_path = str(CWD / "not_a_dir/")
        path_checks(data_path)


@pytest.mark.parametrize(
    "file_name",
    [
        ("test_text.txt"),
        ("test_data.csv"),
        ("test_config.toml"),
    ],
)
def test_path_checks_is_a_file(file_name):
    with pytest.raises(NotADirectoryError):
        data_path = str(CWD / DATA_DIR / file_name)
        path_checks(data_path)
