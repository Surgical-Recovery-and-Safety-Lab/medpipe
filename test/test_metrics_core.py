#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_metrics_core.py

Test functions for the metrics.core submodule.
"""
import pathlib
from unittest.mock import MagicMock, patch

import pytest

from pyrisk.metrics.core import print_metrics, print_metrics_CI
from pyrisk.utils.logger import print_message

CWD = pathlib.Path.cwd()
DATA_DIR = str(CWD / "test/test_data/")


# Helper function to mock print_message for testing
def mock_print_message(message, logger, script_name):
    return message  # Just return the message for comparison in tests


# Test: Check that the function prints or logs the correct metrics for single label
def test_print_metrics_single_label(capsys):
    metric_dict = {
        "accuracy": [0.95],
        "f1": [0.92],
        "precision": [0.93],
        "recall": [0.94],
        "auroc": [0.96],
        "ap": [0.97],
    }
    label_list = ["Label1"]

    print_metrics(metric_dict, label_list)

    # Verify the printed messages
    expected_messages = "[INFO]   Label1 metrics: in metrics/core\n[INFO]     Accuracy: 0.950 in metrics/core\n[INFO]     F1: 0.920 in metrics/core\n[INFO]     Precision: 0.930 in metrics/core\n[INFO]     Recall: 0.940 in metrics/core\n[INFO]     AUROC: 0.960 in metrics/core\n[INFO]     AP: 0.970 in metrics/core\n"

    captured = capsys.readouterr()
    assert captured.out == expected_messages
