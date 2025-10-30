#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_utils.io.py

Test functions for the utils.io module
"""
import pathlib

import pytest

from pyrisk.utils.io import load_data_from_csv

CWD = pathlib.Path.cwd()


def test_load_data_from_csv_success():
    data_path = str(CWD / "test/test_data/test_data.csv")
    load_data_from_csv(data_path)


def test_load_data_from_csv_not_str():
    with pytest.raises(TypeError):
        load_data_from_csv(12)


def test_load_data_from_csv_not_csv_file():
    with pytest.raises(ValueError):
        data_path = str(CWD / "test/test_data/not_a_csv.txt")
        load_data_from_csv(data_path)


def test_load_data_from_csv_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_data_from_csv("not_a_file.csv")
