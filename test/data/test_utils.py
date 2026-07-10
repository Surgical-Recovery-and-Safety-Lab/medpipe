#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test functions for the data.utils module
"""

from re import escape
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from medpipe._types import Labels
from medpipe.data.utils import convert_dtypes, extract_labels, get_split_idx, split_data


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


class TestSplitData:
    """Test class for the split_data function."""

    def _generate_mock_data(self) -> tuple[pd.DataFrame, Labels]:
        """Generate mock data for tests."""
        features = pd.DataFrame(
            {
                "age": [25, 30, 35, 19, 80, 47, 20, 42],
                "sex": ["M", "F", "M", "M", "M", "F", "M", "M"],
                "op_year": [2024, 2024, 2024, 2023, 2023, 2022, 2022, 2023],
            }
        )
        labels = np.array([True, False, False, False, True, True, False, False])
        return (features, labels)

    @pytest.mark.parametrize(
        "test_size, recalibration_size, train_len, test_len",
        [
            (0.5, None, 4, 4),
            (0.25, None, 6, 2),
            (0.75, None, 2, 6),
            (None, 0.5, 4, 4),
            (None, 0.25, 6, 2),
            (None, 0.75, 2, 6),
            (0.5, 1.0, 4, 4),  # Check that test size is used
        ],
    )
    def test_split_data_success_random(
        self,
        test_size: float | None,
        recalibration_size: float | None,
        train_len: int,
        test_len: int,
    ) -> None:
        """Test successful function call with random strategy."""
        features, labels = self._generate_mock_data()

        X_train, y_train, X_test, y_test = split_data(
            features,
            labels,
            strategy="random",
            test_size=test_size,
            recalibration_size=recalibration_size,
        )

        assert len(X_train) == train_len
        assert len(y_train) == train_len
        assert len(X_test) == test_len
        assert len(y_test) == test_len

    @pytest.mark.parametrize(
        "column, values, train_len, test_len",
        [
            ("op_year", [2024], 5, 3),
            ("sex", ["F"], 6, 2),
        ],
    )
    def test_split_data_success_group(
        self,
        column: str,
        values: list[str] | list[int],
        train_len: int,
        test_len: int,
    ) -> None:
        """Test successful function call with group strategy."""
        features, labels = self._generate_mock_data()

        X_train, y_train, X_test, y_test = split_data(
            features, labels, strategy="group", group_column=column, values=values
        )

        assert len(X_train) == train_len
        assert len(y_train) == train_len
        assert len(X_test) == test_len
        assert len(y_test) == test_len

    @pytest.mark.parametrize(
        "features",
        [
            3.14,
            42,
            {},
            [],
            (),
            np.array([]),
        ],
    )
    def test_split_data_incorrect_features_type(self, features: Any) -> None:
        """Test case when features is not pd.DataFrame."""
        match_expr = f"features should be a pd.DataFrame, but got {type(features)}"
        with pytest.raises(TypeError, match=match_expr):
            split_data(features, np.zeros((6, 1)), "random")  # type: ignore

    @pytest.mark.parametrize(
        "labels",
        [
            3.14,
            42,
            {},
            [],
            (),
            pd.Series([]),
        ],
    )
    def test_split_data_incorrect_labels_type(self, labels: Any) -> None:
        """Test case when features is not pd.DataFrame."""
        match_expr = f"labels should be a np.array, but got {type(labels)}"
        with pytest.raises(TypeError, match=match_expr):
            split_data(pd.DataFrame({}), labels, "random")

    @pytest.mark.parametrize(
        "group_column, values",
        [
            (None, [0]),
            ("group", None),
            (None, None),
        ],
    )
    def test_split_data_missing_group_args(
        self, group_column: str | None, values: list[int] | None
    ) -> None:
        """Test case when group_column or values are missing in group strategy."""
        with pytest.raises(
            ValueError,
            match="group_column and values must be specified with group strategy",
        ):
            split_data(
                pd.DataFrame({}),
                np.array([]),
                strategy="group",
                group_column=group_column,
                values=values,
            )

    def test_split_data_missing_random_args(self) -> None:
        """Test case when test_size or recalibration_size are missing in random strategy."""
        with pytest.raises(
            ValueError,
            match="test_size or recalibration_size must be specified with random strategy",
        ):
            split_data(pd.DataFrame({}), np.array([]), "random")

    def test_split_data_incorrect_strategy(self) -> None:
        """Test case when strategy is invalid."""
        with pytest.raises(
            ValueError, match="strategy should be random or group, but got invalid"
        ):
            split_data(pd.DataFrame({}), np.array([]), "invalid")  # type: ignore


class TestConvertDtypes:
    """Test class for the convert_dtypes function."""

    @pytest.mark.parametrize(
        "X, dtypes",
        [
            ({"col1": ["a", "b"]}, [pd.CategoricalDtype()]),
            ({"col1": [1, 2]}, [pd.Int64Dtype()]),
            (
                {"col1": [1, 2], "col2": ["1", "2"]},
                [pd.Int64Dtype(), pd.Int64Dtype()],
            ),
        ],
    )
    def test_convert_dtypes_success(
        self,
        X: dict[str, list[int | str]],
        dtypes: list[pd.CategoricalDtype | pd.Int64Dtype],
    ) -> None:
        """Test successful function call."""
        df = convert_dtypes(pd.DataFrame(X))

        for i, key in enumerate(X.keys()):
            assert df[key].dtype.type == dtypes[i].type

    @pytest.mark.parametrize("X", [3.14, 42, "llama", [], (), {}])
    def test_convert_dtypes_incorrect_type(self, X: Any) -> None:
        """Test case when X is not a pd.DataFrame."""
        match_expr = f"Input X should be a pd.DataFrame, but got {type(X)}"
        with pytest.raises(TypeError, match=match_expr):
            convert_dtypes(X)
