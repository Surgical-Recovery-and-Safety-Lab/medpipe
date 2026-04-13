#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_data_preprocessing.py

Test functions for the data.preprocessing module
"""

import pathlib

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold, StratifiedKFold

from medpipe.data.preprocessing import (
    bin_score,
    convert_object_to_categorical,
    downcast_dtypes,
    extract_labels,
    fit_preprocess_operations,
    get_validation_idx,
    train_test_it,
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


def test_convert_object_to_categorical_success():
    data = load_data_from_csv(DATA_FILE)
    convert_object_to_categorical(data)


@pytest.mark.parametrize(
    "data",
    [
        "string",  # str
        3.14,  # float
        42,  # int
    ],
)
def test_convert_object_to_categorical_type_error(data):
    with pytest.raises(TypeError):
        convert_object_to_categorical(data)


@pytest.mark.parametrize(
    "group_k_fold, kwargs, expected_type",
    [
        (
            False,
            {"n_splits": 5, "shuffle": True, "random_state": 42},
            StratifiedKFold,
        ),
        (
            True,
            {"n_splits": 5, "shuffle": True, "random_state": 42},
            GroupKFold,
        ),
    ],
)
def test_train_test_it_success(group_k_fold, kwargs, expected_type):
    kfold_it = train_test_it(group_k_fold=group_k_fold, **kwargs)
    assert isinstance(kfold_it, expected_type)
    assert kfold_it.n_splits == kwargs["n_splits"]
    assert kfold_it.shuffle == kwargs["shuffle"]
    assert kfold_it.random_state == kwargs["random_state"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_splits": 1},  # n_splits less than 2
        {"n_splits": 0},  # n_splits less than 2
        {"n_splits": -1},  # n_splits less than 2
    ],
)
def test_train_test_it_value_error(kwargs):
    with pytest.raises(ValueError):
        _ = train_test_it(group_k_fold=False, **kwargs)
        return None


def test_preprocess_data_success():
    preprocessing_dict = {
        "ordinal_encoder": {"feature_list": ["sex"]},
        "standardise": {"feature_list": ["age"]},
        "power_transform": {"feature_list": ["age", "dummy"]},
        "bin": {"feature_list": ["M3_score"]},
    }
    operation_dict = fit_preprocess_operations(SAMPLE_DATA, preprocessing_dict)
    assert isinstance(operation_dict, type({}))
    assert len(operation_dict.keys()) == 4


@pytest.mark.parametrize(
    "data, preprocessing_dict",
    [
        (None, {"ordinal_encoder": {"feature_list": ["age"]}}),  # Invalid data type
        (
            SAMPLE_DATA,
            {"ordinal_encoder": {"feature_list": "age"}},
        ),  # Invalid features type (not a list)
        (
            SAMPLE_DATA,
            {"ordinal_encoder": {"feature_list": ["age", 1]}},
        ),  # Invalid feature in list
        (
            SAMPLE_DATA,
            {"invalid_function": {"feature_list": ["age"]}},
        ),  # Invalid preprocess function
        (
            SAMPLE_DATA,
            {"ordinal_encoder": {"feature_list": ["invalid_feature"]}},
        ),  # Feature does not exist
    ],
)
def test_preprocess_data_errors(data, preprocessing_dict):
    with pytest.raises((TypeError, KeyError, ValueError)):
        fit_preprocess_operations(data, preprocessing_dict)


def test_bin_score_success():
    m3_data = bin_score(SAMPLE_DATA["M3_score"])
    assert (m3_data == [1.0, 4.0, 3.0, 4.0, 0.0, 3.0, 2.0, 3.0, 0.0]).all()


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
