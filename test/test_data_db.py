#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_data_db.py

Test functions for the data.db module
"""
import pathlib

import pytest

from pyrisk.data.db import extract_data_from_db, parquet_to_db

CWD = pathlib.Path.cwd()
PARQUET_PATH = str(CWD / "test/test_data/test_data.parquet")
DB_PATH = str(CWD / "test/test_data/test_data.db")
TXT_PATH = str(CWD / "test/test_data/test_text.txt")
QUERY = "SELECT * FROM main"


def test_parquet_to_db_success(tmp_path):
    db_path = str(tmp_path / "save_db.db")
    parquet_to_db(PARQUET_PATH, db_path)


@pytest.mark.parametrize(
    "parquet_path, db_path",
    [
        (PARQUET_PATH, 12),
        (12, DB_PATH),
    ],
)
def test_parquet_to_db_not_str(parquet_path, db_path):
    with pytest.raises(TypeError):
        parquet_to_db(parquet_path, db_path)


@pytest.mark.parametrize(
    "parquet_path, db_path",
    [(PARQUET_PATH, TXT_PATH), (TXT_PATH, DB_PATH)],
)
def test_parquet_to_db_not_extension_file(parquet_path, db_path):
    with pytest.raises(ValueError):
        parquet_to_db(parquet_path, db_path)


def test_parquet_to_db_file_not_found():
    with pytest.raises(FileNotFoundError):
        parquet_path = str(CWD / "test/test_data/not_exist.parquet")
        parquet_to_db(parquet_path, DB_PATH)


def test_parquet_to_db_not_a_file():
    with pytest.raises(IsADirectoryError):
        parquet_path = str(CWD / "test/test_data/")
        parquet_to_db(parquet_path, DB_PATH)


def test_extract_data_from_db_success():
    extract_data_from_db(DB_PATH, QUERY)


@pytest.mark.parametrize(
    "db_path, query",
    [
        (12, QUERY),
        (DB_PATH, 12),
    ],
)
def test_extract_data_from_db_not_str(db_path, query):
    with pytest.raises(TypeError):
        extract_data_from_db(db_path, query)


def test_extract_data_from_db_not_extension_file():
    with pytest.raises(ValueError):
        extract_data_from_db(TXT_PATH, QUERY)


def test_extract_data_from_db_file_not_found():
    with pytest.raises(FileNotFoundError):
        db_path = str(CWD / "test/test_data/not_exist.db")
        extract_data_from_db(db_path, QUERY)


def test_extract_data_from_db_not_a_file():
    with pytest.raises(IsADirectoryError):
        db_path = str(CWD / "test/test_data/")
        extract_data_from_db(db_path, QUERY)
