#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_utils_exceptions.py

Test functions for the utils.exceptions module
"""
import pathlib

import pytest

from pyrisk.utils.exceptions import path_checks

CWD = pathlib.Path.cwd()


@pytest.mark.parametrize(
    "file_name, extension",
    [
        ("test_text.txt", ".txt"),
        ("test_data.csv", ".csv"),
        ("test_config.toml", ".toml"),
    ],
)
def test_path_checks_success(file_name, extension):
    data_path = str(CWD / "test/test_data/" / file_name)
    path_checks(data_path, extension)


def test_path_checks_not_str():
    with pytest.raises(TypeError):
        path_checks(12, ".txt")


@pytest.mark.parametrize(
    "file_name, extension",
    [
        ("test_text.txt", ".csv"),
        ("test_data.csv", ".toml"),
        ("test_config.toml", ".txt"),
    ],
)
def test_path_checks_not_extension_file(file_name, extension):
    with pytest.raises(ValueError):
        data_path = str(CWD / "test/test_data/" / file_name)
        path_checks(data_path, extension)


@pytest.mark.parametrize(
    "file_name, extension",
    [
        ("not_a_file.txt", ".txt"),
        ("not_a_file.csv", ".csv"),
        ("not_a_file.toml", ".toml"),
    ],
)
def test_path_checks_file_not_found(file_name, extension):
    with pytest.raises(FileNotFoundError):
        path_checks(file_name, extension)
