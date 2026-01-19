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

from pyrisk.data.preprocessing import (
    bin_score,
    convert_object_to_categorical,
    extract_labels,
    fit_preprocess_operations,
    test_train_it,
)
from pyrisk.utils.io import load_data_from_csv

CWD = pathlib.Path.cwd()
DATA_DIR = str(CWD / "test/test_data/")
DATA_FILE = str(CWD / DATA_DIR / "test_data.csv")
SAMPLE_DATA = pd.DataFrame(
    {
        "age": [25, 30, 35, 19, 80, 47],
        "sex": ["M", "F", "M", "M", "M", "F"],
        "dummy": [10, 20, 30, 10, 30, 40],
        "M3_score": [0.1, 5.4, 2.9, 4.0, 0.0, 3.0],
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
    "temporal_k_fold, kwargs, expected_type",
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
def test_train_it_success(temporal_k_fold, kwargs, expected_type):
    kfold_it = test_train_it(temporal_k_fold=temporal_k_fold, **kwargs)
    assert isinstance(kfold_it, expected_type)
    assert kfold_it.n_splits == kwargs["n_splits"]
    assert kfold_it.shuffle == kwargs["shuffle"]
    assert kfold_it.random_state == kwargs["random_state"]
    return None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_splits": 1},  # n_splits less than 2
        {"n_splits": 0},  # n_splits less than 2
        {"n_splits": -1},  # n_splits less than 2
    ],
)
def test_train_it_value_error(kwargs):
    with pytest.raises(ValueError):
        _ = test_train_it(temporal_k_fold=False, **kwargs)
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
    assert (m3_data == [1.0, 4.0, 3.0, 4.0, 0.0, 3.0]).all()
