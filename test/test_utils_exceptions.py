#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_utils_exceptions.py

Test functions for the utils.exceptions module
"""

import pathlib

import numpy as np
import pytest
from pandas import Series

from medpipe.utils.exceptions import (
    array_check,
    array_dim_check,
    file_checks,
    path_checks,
)

CWD = pathlib.Path.cwd()
DATA_DIR = str(CWD / "test/test_data/")


@pytest.mark.parametrize(
    "file_name, extension, exists",
    [
        ("test_text.txt", ".txt", True),
        ("test_data.csv", ".csv", True),
        ("config/HGBc_config.toml", ".toml", True),
        ("not_a_file.csv", ".csv", False),
    ],
)
def test_file_checks_success(file_name, extension, exists):
    file_path = str(CWD / DATA_DIR / file_name)
    file_checks(file_path, extension, exists)


def test_file_checks_not_str():
    with pytest.raises(TypeError):
        file_checks(12, ".txt")


@pytest.mark.parametrize(
    "file_name, extension",
    [
        ("test_text.txt", ".csv"),
        ("test_data.csv", ".toml"),
        ("config/HGBc_config.toml", ".txt"),
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


def test_path_checks_create_dir(tmp_path):
    data_path = tmp_path / "not_a_dir/"
    path_checks(str(data_path))
    assert data_path.exists()


@pytest.mark.parametrize(
    "file_name",
    [42, 3.14, [1, 2, 3], None],
)
def test_path_checks_not_str(file_name):
    with pytest.raises(TypeError):
        path_checks(file_name)


@pytest.mark.parametrize(
    "file_name",
    ["test_text.txt", "test_data.csv", "config/HGBc_config.toml"],
)
def test_path_checks_is_a_file(file_name):
    with pytest.raises(NotADirectoryError):
        data_path = str(CWD / DATA_DIR / file_name)
        path_checks(data_path)


@pytest.mark.parametrize(
    "file_name",
    ["test_text.tx", "test_data.cs", "config/HGBc_config.tom"],
)
def test_path_checks_file_not_found(file_name):
    with pytest.raises(FileNotFoundError):
        data_path = str(CWD / DATA_DIR / file_name)
        path_checks(data_path)


@pytest.mark.parametrize(
    "arr",
    [
        np.array([]),
        np.array([1]),
        np.array([[1, 2], [1, 2]]),
        np.array([[1], [2]]),
        [1, 2, 3],  # List
        Series([1, 2]),  # Series
    ],
)
def test_array_check_success(arr):
    array_check(arr)


@pytest.mark.parametrize(
    "not_arr",
    [
        5,  # Int
        3.14,  # Float
        "not_an_arr",  # Str
        {"A", 1},  # Dict
        (42, 3.14, "pi"),  # tuple
    ],
)
def test_array_check_not_array(not_arr):
    with pytest.raises(TypeError):
        array_check(not_arr)


@pytest.mark.parametrize(
    "arr1, arr2, dim",
    [
        (np.array([]), np.array([]), None),
        (np.array([1]), np.array([2]), None),
        (np.array([1, 2]), np.array([2, 3]), None),
        (np.array([1, 2]), np.array([2, 3]), 0),
        (np.array([[1], [2]]), np.array([[1], [3]]), None),
        (np.array([[1], [2]]), np.array([[1], [3]]), 1),
        (np.array([[1, 2], [1, 2]]), np.array([[1, 4], [2, 3]]), None),
        (np.array([[1, 2], [1, 2]]), np.array([[1, 4], [2, 3]]), 0),
        (np.array([[1, 2], [1, 2]]), np.array([[1, 4], [2, 3]]), 1),
    ],
)
def test_array_dim_check_success(arr1, arr2, dim):
    array_dim_check(arr1, arr2, dim)


@pytest.mark.parametrize(
    "arr1, arr2, dim",
    [
        (np.array([]), np.array([1]), None),
        (np.array([1]), np.array([2, 2]), None),
        (np.array([1, 2]), np.array([3]), 0),
        (np.array([[1], [2]]), np.array([[1, 1], [2, 2]]), None),
        (np.array([[1], [2]]), np.array([[1, 1], [2, 2]]), 1),
        (np.array([[1, 2], [1, 2]]), np.array([1, 4]), None),
    ],
)
def test_array_dim_check_not_equal(arr1, arr2, dim):
    with pytest.raises(ValueError):
        array_dim_check(arr1, arr2, dim)


@pytest.mark.parametrize(
    "arr1, arr2, dim",
    [
        (np.array([]), np.array([1]), 1),
        (np.array([1]), np.array([2, 2]), 1),
        (np.array([1, 2]), np.array([2, 3]), 1),
        (np.array([[1], [2]]), np.array([[1], [3]]), 2),
    ],
)
def test_array_dim_check_index_error(arr1, arr2, dim):
    with pytest.raises(IndexError):
        array_dim_check(arr1, arr2, dim)


@pytest.mark.parametrize(
    "dim",
    [
        3.14,  # Float
        "not_an_arr",  # Str
        [1, 2, 3],  # List
        ("arr", 1),  # Tuple
    ],
)
def test_array_check_bad_dim(dim):
    with pytest.raises(TypeError):
        array_dim_check(np.array([]), np.array([]), dim)
