#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_utils_logger.py

Test functions for the utils.logger module
"""

import pathlib

import pytest

from medpipe.utils.logger import setup_logger

CWD = pathlib.Path.cwd()
DATA_DIR = str(CWD / "test/test_data/")
TXT_FILE = str(CWD / DATA_DIR / "test_text.txt")


def test_setup_logger_success(tmp_path):
    setup_logger("script_name", str(tmp_path))


@pytest.mark.parametrize(
    "script_name, log_path",
    [
        (12, DATA_DIR),
        ("script_name", 12),
    ],
)
def test_setup_logger_not_str(script_name, log_path):
    with pytest.raises(TypeError):
        setup_logger(script_name, log_path)


def test_setup_logger_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        log_path = str(tmp_path / "not_a_dir.txt")
        setup_logger("script_name", log_path)


def test_setup_logger_is_a_file():
    with pytest.raises(NotADirectoryError):
        setup_logger("script_name", TXT_FILE)
