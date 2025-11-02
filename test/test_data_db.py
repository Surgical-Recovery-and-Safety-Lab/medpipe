#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_data_db.py

Test functions for the data.db module
"""
import pathlib

import pytest

from pyrisk.data.db import parquet_to_duckdb

CWD = pathlib.Path.cwd()
PARQUET_PATH = str(CWD / "test/test_data/test_data.parquet")
DUCKDB_PATH = str(CWD / "test/test_data/test_data.duckdb")


def test_parquet_to_duckdb_success():
    parquet_to_duckdb(PARQUET_PATH, DUCKDB_PATH)


@pytest.mark.parametrize(
    "parquet_path, duckdb_path",
    [
        (str(CWD / "test/test_data/test_data.parquet"), 12),
        (12, str(CWD / "test/test_data/test_data.duckdb")),
    ],
)
def test_parquet_to_duckdb_not_str(parquet_path, duckdb_path):
    with pytest.raises(TypeError):
        parquet_to_duckdb(parquet_path, duckdb_path)


@pytest.mark.parametrize(
    "parquet_path, duckdb_path",
    [
        (
            str(CWD / "test/test_data/test_data.parquet"),
            str(CWD / "test/test_data/test_data.parquet"),
        ),
        (
            str(CWD / "test/test_data/test_data.duckdb"),
            str(CWD / "test/test_data/test_data.duckdb"),
        ),
    ],
)
def test_parquet_to_duckdb_not_extension_file(parquet_path, duckdb_path):
    with pytest.raises(ValueError):
        parquet_to_duckdb(parquet_path, duckdb_path)


def test_parquet_to_duckdb_file_not_found():
    with pytest.raises(FileNotFoundError):
        parquet_path = str(CWD / "test/test_data/not_exist.parquet")
        parquet_to_duckdb(parquet_path, DUCKDB_PATH)
