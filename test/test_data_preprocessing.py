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
from sklearn.preprocessing import LabelEncoder, PowerTransformer, StandardScaler

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
SAMPLE_DATA = pd.DataFrame({"age": [25, 30, 35], "sex": ["M", "F", "M"]})


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
        kfold_it = test_train_it(temporal_k_fold=False, **kwargs)
        return None


@pytest.mark.parametrize(
    "preprocess, method",
    [
        ("label_encoder", LabelEncoder),
        ("standardise", StandardScaler),
        ("power_transform", PowerTransformer),
    ],
)
def test_preprocess_data_success(preprocess, method):
    if preprocess == "label_encoder":
        features = ["sex"]  # Column that can be processed for each type
    else:
        features = ["age"]
    processed_data = preprocess_data(SAMPLE_DATA, features, preprocess)

    assert isinstance(
        processed_data, pd.DataFrame
    )  # Ensure the result is a pd.DataFrame

    if preprocess == "label_encoder":
        assert processed_data[features].iloc[0].dtype == int
        assert (
            np.squeeze(processed_data[features].to_numpy())
            == method().fit_transform(np.ravel(SAMPLE_DATA[features]))
        ).all
    else:
        assert (
            np.squeeze(processed_data[features].to_numpy())
            == method().fit_transform(SAMPLE_DATA[features])
        ).all


@pytest.mark.parametrize(
    "data, features, preprocess",
    [
        (None, ["age"], "label_encoder"),  # Invalid data type
        (SAMPLE_DATA, "age", "label_encoder"),  # Invalid features type (not a list)
        (SAMPLE_DATA, ["age", 1], "label_encoder"),  # Invalid feature in list
        (SAMPLE_DATA, ["age"], "invalid_function"),  # Invalid preprocess function
    ],
)
def test_preprocess_data_errors(data, features, preprocess):
    with pytest.raises((TypeError, KeyError, ValueError)):
        preprocess_data(data, features, preprocess)


@pytest.mark.parametrize(
    "data, features",
    [
        (None, ["age"]),  # Invalid data type
        (SAMPLE_DATA, "age"),  # Invalid features type (not a list)
        (SAMPLE_DATA, ["age", "non_existing_feature"]),  # Feature doesn't exist in data
    ],
)
def test_preprocess_data_invalid_input(data, features):
    with pytest.raises((TypeError, KeyError)):
        preprocess_data(data, features, "label_encoder")
