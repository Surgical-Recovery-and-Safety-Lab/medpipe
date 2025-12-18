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
from sklearn.preprocessing import OrdinalEncoder, PowerTransformer, StandardScaler

from pyrisk.data.preprocessing import (
    convert_object_to_categorical,
    extract_labels,
    preprocess_data,
    test_train_it,
)
from pyrisk.utils.io import load_data_from_csv

CWD = pathlib.Path.cwd()
DATA_DIR = str(CWD / "test/test_data/")
DATA_FILE = str(CWD / DATA_DIR / "test_data.csv")
SAMPLE_DATA = pd.DataFrame(
    {"age": [25, 30, 35], "sex": ["M", "F", "M"], "dummy": [10, 20, 30]}
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
    }
    processed_data, _ = preprocess_data(SAMPLE_DATA, preprocessing_dict)
    assert isinstance(processed_data, pd.DataFrame)

    sex_data = OrdinalEncoder().fit_transform(
        np.ravel(SAMPLE_DATA["sex"]).reshape(-1, 1)
    )
    dummy_data = PowerTransformer().fit_transform(
        np.ravel(SAMPLE_DATA["dummy"]).reshape(-1, 1)
    )
    age_data = PowerTransformer().fit_transform(
        StandardScaler().fit_transform(np.ravel(SAMPLE_DATA["age"]).reshape(-1, 1))
    )

    assert (np.squeeze(processed_data["sex"].to_numpy()) == sex_data).all
    assert (np.squeeze(processed_data["dummy"].to_numpy()) == dummy_data).all
    assert (np.squeeze(processed_data["age"].to_numpy()) == age_data).all


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
        preprocess_data(data, preprocessing_dict)
