#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test functions for the data.utils module
"""

import pathlib
from re import escape
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from medpipe.data.utils import (
    convert_data,
    downcast_dtypes,
    extract_labels,
    get_data_from_idx,
    get_split_idx,
    split_data,
)


class TestExtractLabels:
    """Test class for the extract_labels function."""

    @pytest.fixture
    def mock_data(self) -> pd.DataFrame:
        """Generate mock data for tests."""
        return pd.DataFrame(
            {
                "age": [25, 30, 35, 19, 80, 47, 20, 42, 69],
                "sex": ["M", "F", "M", "M", "M", "F", "M", "M", "F"],
                "any_comp": [True, True, False, False, False, True, True, False, False],
                "op_year": [2024, 2024, 2024, 2023, 2023, 2022, 2022, 2022, 2023],
            }
        )

    @pytest.mark.parametrize(
        "labels",
        [
            ["any_comp"],
            ["op_year", "any_comp"],
        ],
    )
    def test_extract_labels_success(
        self, labels: list[str], mock_data: pd.DataFrame
    ) -> None:
        """Test successful function call."""
        X, y = extract_labels(mock_data, labels)

        assert_frame_equal(X, mock_data.drop(columns=labels))
        assert (y == mock_data[labels].to_numpy()).all()

    @pytest.mark.parametrize("labels", [3.14, 42, {}, ()])
    def test_extract_labels_incorrect_labels_type(
        self, mock_data: pd.DataFrame, labels: Any
    ) -> None:
        """Test case when labels type is incorrect."""
        match_expr = f"labels should be a list, but got {type(labels)}"

        with pytest.raises(TypeError, match=match_expr):
            extract_labels(mock_data, labels)

    @pytest.mark.parametrize("labels", [[3.14], [42], [{}], [()]])
    def test_extract_labels_incorrect_label_list(
        self, mock_data: pd.DataFrame, labels: list[Any]
    ) -> None:
        """Test case when labels type is not list of string."""
        match_expr = f"labels should contain strings, but got {type(labels[0])}"

        with pytest.raises(TypeError, match=match_expr):
            extract_labels(mock_data, labels)

    @pytest.mark.parametrize("data", [3.14, 42, {}, (), []])
    def test_extract_labels_incorrect_data_type(self, data: Any) -> None:
        """Test case when data type is incorrect."""
        match_expr = f"data should be a pd.DataFrame, but got {type(data)}"

        with pytest.raises(TypeError, match=match_expr):
            extract_labels(data, ["a"])

    def test_extract_labels_invalid_label(self, mock_data: pd.DataFrame) -> None:
        """Test case when label is not in data."""
        with pytest.raises(ValueError, match=f"invalid_label was not found in data"):
            extract_labels(mock_data, ["invalid_label"])


class TestGetSplitIdx:
    """Test class for the get_split_idx function."""

    def _generate_mock_data(
        self, data_type: Literal["str", "int"]
    ) -> tuple[npt.NDArray, pd.Series | npt.NDArray, list[str] | list[int]]:
        """Generate mock data for tests."""
        idx_list = np.arange(6)
        if data_type == "int":
            column = np.array([2024, 2022, 2024, 2023, 2022, 2023])
            values = [2024]
        else:
            column = pd.Series(
                [
                    "primary",
                    "tertiary",
                    "primary",
                    "secondaray",
                    "secondary",
                    "tertiary",
                ]
            )
            values = ["primary", "tertiary"]
        return (idx_list, column, values)

    @pytest.mark.parametrize(
        "data_type, true_train_idx",
        [("str", np.array([3, 4])), ("int", np.array([1, 3, 4, 5]))],
    )
    def test_get_split_idx_success(
        self, data_type: Literal["int", "str"], true_train_idx: npt.NDArray
    ) -> None:
        """Test successful function call."""
        train_idx, test_idx = get_split_idx(*self._generate_mock_data(data_type))

        assert (train_idx == true_train_idx).all()
        assert (
            test_idx == np.setdiff1d(np.arange(6), true_train_idx, assume_unique=True)
        ).all()

    @pytest.mark.parametrize("column", [3.14, 42, {}, [], ()])
    def test_get_split_idx_invalid_column_type(self, column: Any) -> None:
        """Test case when column is invalid type."""
        match_expr = f"column should be pd.Series or np.array, got {type(column)}"
        with pytest.raises(TypeError, match=match_expr):
            get_split_idx(np.arange(6), column, [2024])

    @pytest.mark.parametrize("values", [3.14, 42, {}, ()])
    def test_get_split_idx_invalid_value_type(self, values: Any) -> None:
        """Test case when column is invalid type."""
        match_expr = f"values should be list or np.array, got {type(values)}"
        with pytest.raises(TypeError, match=match_expr):
            get_split_idx(np.arange(6), np.zeros((6, 1)), values)

    @pytest.mark.parametrize(
        "column, values", [(np.zeros((6, 1)), [1]), (np.array(["a"] * 6), ["b"])]
    )
    def test_get_split_idx_value_not_in_column(
        self, column: npt.NDArray, values: list[str] | list[int]
    ) -> None:
        """Test case when value is not in column."""
        match_expr = f"{values} not present in column"
        with pytest.raises(ValueError, match=escape(match_expr)):
            get_split_idx(np.arange(6), column=column, values=values)


"""
def test_extract_labels_key_error():
    with pytest.raises(KeyError):
        data = load_data_from_csv(DATA_FILE)
        extract_labels(data, ["NOT_A_KEY"])


@pytest.mark.parametrize("val_size", [0.1, 0.2, 0.9])
def test_get_validation_idx_val_size_success(val_size):
    train_idx, val_idx = get_validation_idx(np.arange(100), val_size=val_size)
    assert len(train_idx) == np.round(100 * (1 - val_size))
    assert len(val_idx) == np.round(100 * val_size)


@pytest.mark.parametrize(
    "groups, train_idx_true, val_idx_true",
    [
        (SAMPLE_DATA["dummy"], np.array([0, 1, 2, 3, 4, 6, 7, 8]), np.array([5])),
        (SAMPLE_DATA["sex"], np.array([1, 5, 8]), np.array([0, 2, 3, 4, 6, 7])),
    ],
)
def test_get_validation_idx_groups_success(groups, train_idx_true, val_idx_true):
    train_idx, val_idx = get_validation_idx(np.arange(len(SAMPLE_DATA)), groups=groups)
    assert (train_idx == train_idx_true).all()
    assert (val_idx == val_idx_true).all()


@pytest.mark.parametrize(
    "groups, group_vals, train_idx_true, val_idx_true",
    [
        (
            SAMPLE_DATA["dummy"],
            [10, 20],
            np.array([2, 4, 5, 8]),
            np.array([0, 1, 3, 6, 7]),
        ),
        (SAMPLE_DATA["sex"], ["F"], np.array([0, 2, 3, 4, 6, 7]), np.array([1, 5, 8])),
    ],
)
def test_get_validation_idx_groups_val_success(
    groups, group_vals, train_idx_true, val_idx_true
):
    train_idx, val_idx = get_validation_idx(
        np.arange(len(SAMPLE_DATA)), groups=groups, group_vals=group_vals
    )
    assert (train_idx == train_idx_true).all()
    assert (np.sort(val_idx) == val_idx_true).all()


@pytest.mark.parametrize("val_size", [-0.5, 1.2])
def test_get_validation_idx_value_error(val_size):
    with pytest.raises(ValueError):
        _, _ = get_validation_idx(np.arange(100), val_size=val_size)


def test_get_validation_idx_groups_value_error():
    with pytest.raises(ValueError):
        _, _ = get_validation_idx(np.arange(100), groups=np.array([1, 2]))


@pytest.mark.parametrize("val_size", [1, [1, 2], "string", (1, "a"), {"a": 1}])
def test_get_validation_idx_val_size_type_error(val_size):
    with pytest.raises(TypeError):
        _, _ = get_validation_idx(np.arange(100), val_size=val_size)


@pytest.mark.parametrize("groups", [1, [1, 2], "string", (1, "a"), {"a": 1}])
def test_get_validation_idx_groups_type_error(groups):
    with pytest.raises(TypeError):
        _, _ = get_validation_idx(np.arange(100), groups=groups)


@pytest.mark.parametrize("group_vals", [1, 0.2, "a", (1, "a"), {"a": 1}])
def test_get_validation_idx_group_vals_type_error(group_vals):
    with pytest.raises(TypeError):
        _, _ = get_validation_idx(
            np.arange(len(SAMPLE_DATA)),
            groups=SAMPLE_DATA["dummy"],
            group_vals=group_vals,
        )


def test_downcast_dtypes_success():
    downcast_data = downcast_dtypes(SAMPLE_DATA)
    assert pd.api.types.is_integer_dtype(downcast_data["age"])
    assert pd.api.types.is_object_dtype(downcast_data["sex"])
    assert pd.api.types.is_integer_dtype(downcast_data["dummy"])
    assert pd.api.types.is_float_dtype(downcast_data["M3_score"])


@pytest.mark.parametrize(
    "data",
    [
        pd.DataFrame({"ints": [1, 2, 3]}),
        pd.DataFrame({"floats": [0.1, 0.2, 0.4]}),
        pd.DataFrame({"ints": [1, 2], "floats": [0.1, 0.2]}),
        np.array([1, 2, 3]),
    ],
)
def test_convert_data_successfull_conversion(data):
    converted_data = convert_data(data)
    assert isinstance(converted_data, np.ndarray)


@pytest.mark.parametrize(
    "data",
    [
        pd.DataFrame({"objects": ["a", "b"]}),
        pd.DataFrame({"ints": [1, 2], "objects": ["a", "b"]}),
        SAMPLE_DATA,
    ],
)
def test_convert_data_successfull_ignore(data):
    converted_data = convert_data(data)
    assert isinstance(converted_data, pd.DataFrame)


@pytest.mark.parametrize(
    "data, idx",
    [
        (pd.DataFrame({"ints": [1, 2, 3]}), [0, 2]),
        (pd.DataFrame({"floats": [0.1, 0.2, 0.4]}), [-1]),
        (pd.DataFrame({"ints": [1, 2], "floats": [0.1, 0.2]}), [0, 1]),
        (SAMPLE_DATA, np.arange(6)),
    ],
)
def test_get_data_from_idx_dataframe(data, idx):
    indexed_data = get_data_from_idx(data, idx)
    assert_frame_equal(indexed_data, data.iloc[idx])


@pytest.mark.parametrize(
    "data, idx",
    [
        (np.array([0, 1, 2, 3]), [0, 2]),
        (np.array([0, 1, 2, 3]), [-1]),
        (np.array([0, 1, 2, 3, 4, 5, 6, 7, 0]), np.arange(6)),
    ],
)
def test_get_data_from_idx_array(data, idx):
    indexed_data = get_data_from_idx(data, idx)
    assert (indexed_data == data[idx]).all()


def test_get_data_from_empty_idx():
    indexed_data = get_data_from_idx(SAMPLE_DATA)
    assert_frame_equal(indexed_data, SAMPLE_DATA)


@pytest.mark.parametrize(
    "data, idx",
    [
        (pd.DataFrame({"ints": [1, 2, 3]}), [0.0, 2.0]),
        (pd.DataFrame({"floats": [0.1, 0.2, 0.4]}), ["a"]),
        (pd.DataFrame({"ints": [1, 2], "floats": [0.1, 0.2]}), {"a": 1}),
        ([1, 2, 3], [0]),
        (SAMPLE_DATA, "TypeError"),
    ],
)
def test_get_data_type_error(data, idx):
    with pytest.raises(TypeError):
        get_data_from_idx(data, idx)
"""
