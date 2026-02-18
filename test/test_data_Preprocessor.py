#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_data_Preprocessor.py

Test functions for the Preprocessor class.
"""
import pathlib
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import OrdinalEncoder, PowerTransformer, StandardScaler

from pyrisk.data.preprocessing import bin_score, fit_preprocess_operations
from pyrisk.data.Preprocessor import Preprocessor

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
CONFIG = {
    "preprocess": True,
    "ordinal_encoder": {"feature_list": ["sex"]},
    "standardise": {"feature_list": ["age"]},
    "power_transform": {"feature_list": ["age", "dummy"]},
    "bin": {"feature_list": ["M3_score"]},
}


def test_fit_transform():
    preprocessor = Preprocessor(deepcopy(CONFIG), logger=None)
    processed_data = preprocessor.fit_transform(SAMPLE_DATA)
    assert isinstance(processed_data, pd.DataFrame)

    test_data = deepcopy(SAMPLE_DATA)
    test_data["sex"] = OrdinalEncoder().fit_transform(
        np.ravel(test_data["sex"]).reshape(-1, 1)
    )
    test_data["age"] = StandardScaler().fit_transform(
        np.ravel(test_data["age"]).reshape(-1, 1)
    )
    test_data[["age", "dummy"]] = PowerTransformer().fit_transform(
        test_data[["age", "dummy"]]
    )
    test_data["M3_score"] = bin_score(test_data["M3_score"])

    assert (processed_data.to_numpy() == test_data.to_numpy()).all()
    assert (processed_data["age"].to_numpy() == test_data["age"].to_numpy()).all()
    assert (processed_data["dummy"].to_numpy() == test_data["dummy"].to_numpy()).all()
    assert (processed_data["age"].to_numpy() == test_data["age"].to_numpy()).all()
    assert (
        processed_data["M3_score"].to_numpy() == test_data["M3_score"].to_numpy()
    ).all()


def test_fit():
    preprocessor = Preprocessor(deepcopy(CONFIG), logger=None)
    preprocessor.fit(SAMPLE_DATA)

    operation_dict = fit_preprocess_operations(SAMPLE_DATA, preprocessor.transform_seq)

    for key in operation_dict.keys():
        assert key in preprocessor.operations.keys()


def test_transform():
    preprocessor = Preprocessor(deepcopy(CONFIG), logger=None)
    preprocessor.fit(SAMPLE_DATA)
    processed_data = preprocessor.transform(SAMPLE_DATA)

    test_data = deepcopy(SAMPLE_DATA)
    test_data["sex"] = OrdinalEncoder().fit_transform(
        np.ravel(test_data["sex"]).reshape(-1, 1)
    )
    test_data["age"] = StandardScaler().fit_transform(
        np.ravel(test_data["age"]).reshape(-1, 1)
    )
    test_data[["age", "dummy"]] = PowerTransformer().fit_transform(
        test_data[["age", "dummy"]]
    )
    test_data["M3_score"] = bin_score(test_data["M3_score"])

    assert (processed_data.to_numpy() == test_data.to_numpy()).all()
    assert (processed_data["age"].to_numpy() == test_data["age"].to_numpy()).all()
    assert (processed_data["dummy"].to_numpy() == test_data["dummy"].to_numpy()).all()
    assert (processed_data["age"].to_numpy() == test_data["age"].to_numpy()).all()
    assert (
        processed_data["M3_score"].to_numpy() == test_data["M3_score"].to_numpy()
    ).all()
