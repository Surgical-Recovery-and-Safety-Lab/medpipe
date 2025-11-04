#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_data_preprocessing.py

Test functions for the data.preprocessing module
"""
import pathlib

import pytest

from pyrisk.data.preprocessing import split_test_train
from pyrisk.utils.io import load_data_from_csv

CWD = pathlib.Path.cwd()
DATA_DIR = str(CWD / "test/test_data/")
DATA_FILE = str(CWD / DATA_DIR / "test_data.csv")


def test_split_test_train_success_no_additionals():
    data = load_data_from_csv(DATA_FILE)
    split_test_train(data)


@pytest.mark.parametrize(
    "train_split, random_state",
    [
        (0.2, 12),
        (0.9, 30),
        (0.5, 1),
    ],
)
def test_split_test_train_success(train_split, random_state):
    data = load_data_from_csv(DATA_FILE)
    split_test_train(data, train_split, random_state)


@pytest.mark.parametrize(
    "train_split",
    [
        -0.2,
        1.5,
    ],
)
def test_split_test_train_bad_split(train_split):
    with pytest.raises(ValueError):
        data = load_data_from_csv(DATA_FILE)
        split_test_train(data, train_split)


@pytest.mark.parametrize(
    "train_split, random_state",
    [
        ("not_split", 12),
        (1, 30),
        (0.5, "str"),
        (0.5, 0.2),
    ],
)
def test_split_test_train_type_errors(train_split, random_state):
    with pytest.raises(TypeError):
        data = load_data_from_csv(DATA_FILE)
        split_test_train(data, train_split, random_state)


@pytest.mark.parametrize(
    "data",
    [
        -0.2,
        "string",
        1,
    ],
)
def test_split_test_train_bad_data(data):
    with pytest.raises(TypeError):
        split_test_train(data)
