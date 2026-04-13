#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_data_utils.py

Test functions for the data.utils module
"""

import pathlib

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from medpipe.data.utils import (
    convert_data,
    downcast_dtypes,
    extract_labels,
    get_data_from_idx,
    get_validation_idx,
)
from medpipe.utils.io import load_data_from_csv

CWD = pathlib.Path.cwd()
DATA_DIR = str(CWD / "test/test_data/")
DATA_FILE = str(CWD / DATA_DIR / "test_data.csv")
SAMPLE_DATA = pd.DataFrame(
    {
        "age": [25, 30, 35, 19, 80, 47, 20, 42, 69],
        "sex": ["M", "F", "M", "M", "M", "F", "M", "M", "F"],
        "dummy": [10, 20, 30, 10, 30, 40, 10, 20, 30],
        "M3_score": [0.1, 5.4, 2.9, 4.0, 0.0, 3.0, 1.9, 2.3, 0.0],
    }
)


@pytest.mark.parametrize(
    "labels",
    [
        ["AGE"],
        ["AGE", "SEX"],
    ],
)
def test_extract_labels_success(labels):
    data = load_data_from_csv(DATA_FILE)
    extract_labels(data, labels)


@pytest.mark.parametrize(
    "data, labels",
    [
        (load_data_from_csv(DATA_FILE), "AGE"),  # str
        (load_data_from_csv(DATA_FILE), [1, 2]),  # list(int)
        (load_data_from_csv(DATA_FILE), 3.14),  # float
        ("string", ["AGE"]),  # str
        (3.14, ["AGE"]),  # float
    ],
)
def test_extract_labels_type_error(data, labels):
    with pytest.raises(TypeError):
        extract_labels(data, labels)


def test_extract_labels_key_error():
    with pytest.raises(KeyError):
        data = load_data_from_csv(DATA_FILE)
        extract_labels(data, ["NOT_A_KEY"])


@pytest.mark.parametrize("val_size", [0.1, 0.2, 0.9])
def test_get_validation_idx_val_size_success(val_size):
    train_idx, val_idx = get_validation_idx(np.arange(100), val_size=val_size)
    assert len(train_idx) == np.round(100 * (1 - val_size))
    assert len(val_idx) == np.round(100 * val_size)


@pytest.mark.parametrize(
    "groups, train_idx_true, val_idx_true",
    [
        (SAMPLE_DATA["dummy"], np.array([0, 1, 2, 3, 4, 6, 7, 8]), np.array([5])),
        (SAMPLE_DATA["sex"], np.array([1, 5, 8]), np.array([0, 2, 3, 4, 6, 7])),
    ],
)
def test_get_validation_idx_groups_success(groups, train_idx_true, val_idx_true):
    train_idx, val_idx = get_validation_idx(np.arange(len(SAMPLE_DATA)), groups=groups)
    assert (train_idx == train_idx_true).all()
    assert (val_idx == val_idx_true).all()


@pytest.mark.parametrize(
    "groups, group_vals, train_idx_true, val_idx_true",
    [
        (
            SAMPLE_DATA["dummy"],
            [10, 20],
            np.array([2, 4, 5, 8]),
            np.array([0, 1, 3, 6, 7]),
        ),
        (SAMPLE_DATA["sex"], ["F"], np.array([0, 2, 3, 4, 6, 7]), np.array([1, 5, 8])),
    ],
)
def test_get_validation_idx_groups_val_success(
    groups, group_vals, train_idx_true, val_idx_true
):
    train_idx, val_idx = get_validation_idx(
        np.arange(len(SAMPLE_DATA)), groups=groups, group_vals=group_vals
    )
    assert (train_idx == train_idx_true).all()
    assert (np.sort(val_idx) == val_idx_true).all()


@pytest.mark.parametrize("val_size", [-0.5, 1.2])
def test_get_validation_idx_value_error(val_size):
    with pytest.raises(ValueError):
        _, _ = get_validation_idx(np.arange(100), val_size=val_size)


def test_get_validation_idx_groups_value_error():
    with pytest.raises(ValueError):
        _, _ = get_validation_idx(np.arange(100), groups=np.array([1, 2]))


@pytest.mark.parametrize("val_size", [1, [1, 2], "string", (1, "a"), {"a": 1}])
def test_get_validation_idx_val_size_type_error(val_size):
    with pytest.raises(TypeError):
        _, _ = get_validation_idx(np.arange(100), val_size=val_size)


@pytest.mark.parametrize("groups", [1, [1, 2], "string", (1, "a"), {"a": 1}])
def test_get_validation_idx_groups_type_error(groups):
    with pytest.raises(TypeError):
        _, _ = get_validation_idx(np.arange(100), groups=groups)


@pytest.mark.parametrize("group_vals", [1, 0.2, "a", (1, "a"), {"a": 1}])
def test_get_validation_idx_group_vals_type_error(group_vals):
    with pytest.raises(TypeError):
        _, _ = get_validation_idx(
            np.arange(len(SAMPLE_DATA)),
            groups=SAMPLE_DATA["dummy"],
            group_vals=group_vals,
        )


def test_downcast_dtypes_success():
    downcast_data = downcast_dtypes(SAMPLE_DATA)
    assert pd.api.types.is_integer_dtype(downcast_data["age"])
    assert pd.api.types.is_object_dtype(downcast_data["sex"])
    assert pd.api.types.is_integer_dtype(downcast_data["dummy"])
    assert pd.api.types.is_float_dtype(downcast_data["M3_score"])


@pytest.mark.parametrize(
    "data",
    [
        pd.DataFrame({"ints": [1, 2, 3]}),
        pd.DataFrame({"floats": [0.1, 0.2, 0.4]}),
        pd.DataFrame({"ints": [1, 2], "floats": [0.1, 0.2]}),
        np.array([1, 2, 3]),
    ],
)
def test_convert_data_successfull_conversion(data):
    converted_data = convert_data(data)
    assert isinstance(converted_data, np.ndarray)


@pytest.mark.parametrize(
    "data",
    [
        pd.DataFrame({"objects": ["a", "b"]}),
        pd.DataFrame({"ints": [1, 2], "objects": ["a", "b"]}),
        SAMPLE_DATA,
    ],
)
def test_convert_data_successfull_ignore(data):
    converted_data = convert_data(data)
    assert isinstance(converted_data, pd.DataFrame)


@pytest.mark.parametrize(
    "data, idx",
    [
        (pd.DataFrame({"ints": [1, 2, 3]}), [0, 2]),
        (pd.DataFrame({"floats": [0.1, 0.2, 0.4]}), [-1]),
        (pd.DataFrame({"ints": [1, 2], "floats": [0.1, 0.2]}), [0, 1]),
        (SAMPLE_DATA, np.arange(6)),
    ],
)
def test_get_data_from_idx_dataframe(data, idx):
    indexed_data = get_data_from_idx(data, idx)
    assert_frame_equal(indexed_data, data.iloc[idx])


@pytest.mark.parametrize(
    "data, idx",
    [
        (np.array([0, 1, 2, 3]), [0, 2]),
        (np.array([0, 1, 2, 3]), [-1]),
        (np.array([0, 1, 2, 3, 4, 5, 6, 7, 0]), np.arange(6)),
    ],
)
def test_get_data_from_idx_array(data, idx):
    indexed_data = get_data_from_idx(data, idx)
    assert (indexed_data == data[idx]).all()


def test_get_data_from_empty_idx():
    indexed_data = get_data_from_idx(SAMPLE_DATA)
    assert_frame_equal(indexed_data, SAMPLE_DATA)


@pytest.mark.parametrize(
    "data, idx",
    [
        (pd.DataFrame({"ints": [1, 2, 3]}), [0.0, 2.0]),
        (pd.DataFrame({"floats": [0.1, 0.2, 0.4]}), ["a"]),
        (pd.DataFrame({"ints": [1, 2], "floats": [0.1, 0.2]}), {"a": 1}),
        ([1, 2, 3], [0]),
        (SAMPLE_DATA, "TypeError"),
    ],
)
def test_get_data_type_error(data, idx):
    with pytest.raises(TypeError):
        get_data_from_idx(data, idx)
