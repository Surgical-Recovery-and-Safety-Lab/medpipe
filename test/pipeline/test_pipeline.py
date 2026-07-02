#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline class tests suite.
"""

from pathlib import Path
from typing import Any, Generator, Literal, TypeAlias
from unittest.mock import patch

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from sklearn.base import check_is_fitted
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline

from medpipe._types import Labels
from medpipe.pipeline.pipeline import MedpipePipeline
from medpipe.utils.config import PreprocessOperationConfig
from medpipe.utils.io import load_data, read_toml_configuration

# ==============================================================================
# Fixtures for all tests
# ==============================================================================

MockData: TypeAlias = tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
DataPrep: TypeAlias = tuple[
    npt.NDArray,
    Labels,
    npt.NDArray | None,
    Labels | None,
    npt.NDArray | None,
    StratifiedKFold | GroupKFold,
]


@pytest.fixture(autouse=True)
def shield_local_filesystem() -> Generator[Any, Any, Any]:
    """Automatically prevent any test in this file from creating folders on disk."""
    with (
        patch("pathlib.Path.mkdir") as mock_path_mkdir,
        patch("os.makedirs") as mock_os_makedirs,
    ):
        # Relinquish control back to the test runner loop
        yield mock_path_mkdir, mock_os_makedirs


@pytest.fixture
def example_config_dir() -> Path:
    """Provide the location of the example configuration files."""
    base_dir = Path(__file__).parent.parent.parent

    return base_dir / "config-examples/"


@pytest.fixture
def mp_pipeline(example_config_dir: Path) -> MedpipePipeline:
    """Create a Medpipe Pipeline for tests."""
    return MedpipePipeline(example_config_dir / "HGBc_config.toml", logger=None)


@pytest.fixture
def mock_data() -> MockData:
    """Generate some mock data for the some tests."""
    X_train = pd.DataFrame(
        {
            "SEX": ["F", "M", "M", "F", "M", "M", "F"],
            "AGE": [20, 43, 84, 19, 43, 28, 71],
            "PRIOR_CANCER": [0, 1, 0, 1, 0, 0, 0],
            "ADMISSION_ACUITY": [
                "Acute",
                "Elective",
                "Acute",
                "Acute",
                "Elective",
                "Elective",
                "Elective",
            ],
            "ADMISSION_SOURCE": [
                "Routine",
                "Routine",
                "Routine",
                "Transfer",
                "Routine",
                "Transfer",
                "Routine",
            ],
            "CATEGORY_LEVEL_1": [
                "Plastics",
                "Plastics",
                "Plastics",
                "General Surgery",
                "General Surgery",
                "General Surgery",
                "Plastics",
            ],
            "OP_SEVERITY": [1, 3, 2, 1, 1, 2, 4],
            "OP_YEAR": [2022, 2021, 2023, 2022, 2021, 2022, 2023],
            "DHB_NAME": [
                "Auckland",
                "Christchurch",
                "Wellington",
                "Auckland",
                "Auckland",
                "Wellington",
                "Christchurch",
            ],
        }
    )
    X_test = pd.DataFrame(
        {
            "SEX": ["M", "M", "F"],
            "AGE": [69, 43, 24],
            "PRIOR_CANCER": [0, 1, 0],
            "ADMISSION_ACUITY": ["Acute", "Elective", "Elective"],
            "ADMISSION_SOURCE": ["Transfer", "Transfer", "Routine"],
            "CATEGORY_LEVEL_1": ["Plastics", "General Surgery", "General Surgery"],
            "OP_SEVERITY": [1, 4, 2],
            "OP_YEAR": [2024, 2024, 2024],
            "DHB_NAME": ["Auckland", "Wellington", "Auckland"],
        }
    )
    X_recal = pd.DataFrame(
        {
            "SEX": ["M", "F", "F"],
            "AGE": [23, 45, 72],
            "PRIOR_CANCER": [0, 1, 1],
            "ADMISSION_ACUITY": ["Elective", "Acute", "Elective"],
            "ADMISSION_SOURCE": ["Routine", "Routine", "Routine"],
            "CATEGORY_LEVEL_1": ["Plastics", "General Surgery", "Plastics"],
            "OP_SEVERITY": [1, 2, 2],
            "OP_YEAR": [2023, 2023, 2023],
            "DHB_NAME": ["Wellington", "Christchurch", "Christchurch"],
        }
    )
    return (X_train, X_test, X_recal)


@pytest.fixture
def cv_data_prep(mp_pipeline: MedpipePipeline, mock_data: MockData) -> DataPrep:
    """Run preparation code before calling _cv_fit."""
    # Get different split data sets
    X_train, X_test, X_recal = mock_data
    y_train = np.array([1, 1, 1, 1, 0, 0, 0])
    y_recal = np.array([0, 1, 1])

    X_train, X_test, X_recal, groups = mp_pipeline._drop_group_columns(
        X_train, X_test, X_recal
    )
    X_train, X_recal = mp_pipeline._prepare_features(X_train, X_recal)
    # Create cross-validation generator
    cv_generator = mp_pipeline._get_cv_generator()

    return (X_train, y_train, X_recal, y_recal, groups, cv_generator)


@pytest.fixture
def mock_cv_results() -> dict[str, npt.NDArray]:
    """Generate mock cv_results."""
    cv_results = {
        "test_auroc": np.array([0.94, 0.99, 0.952]),
        "test_ici": np.array([0.01, 0.0223, 0.001]),
        "recal_auroc": np.array([0.9900, 0.93, 0.92]),
        "recal_ici": np.array([0.01, 0.01, 0.00]),
    }
    return cv_results


# ==============================================================================
# Test classes
# ==============================================================================


class TestPipeline:
    """Test class for the MedpipePipeline class"""

    def test_create_pipeline_from_file(self, example_config_dir: Path) -> None:
        """Test successful pipeline creation from configuration file."""
        pipe = MedpipePipeline(example_config_dir / "HGBc_config.toml", logger=None)
        assert pipe.version == "v0.1.1"
        assert pipe.predictor_algo == "HistGradientBoostingClassifier"
        assert pipe.recalibrator_method == "IsotonicRegression"
        assert pipe.n_outcomes == 1
        assert isinstance(pipe.preprocessor, Pipeline)

    def test_create_pipeline_from_config(self, example_config_dir: Path) -> None:
        """Test successful pipeline creation from MedpipeConfig."""
        config = read_toml_configuration(example_config_dir / "HGBc_config.toml")
        pipe = MedpipePipeline(config, logger=None)
        assert pipe.version == "v0.1.1"
        assert pipe.predictor_algo == "HistGradientBoostingClassifier"
        assert pipe.recalibrator_method == "IsotonicRegression"
        assert pipe.n_outcomes == 1
        assert isinstance(pipe.preprocessor, Pipeline)


class TestCheckOp:
    """Test class for the _check_op function of the MedpipePipeline class."""

    @pytest.mark.parametrize("op", ("StandardScaler", "SimpleImputer"))
    def test_pipeline_check_operation_success(
        self, mp_pipeline: MedpipePipeline, op: str
    ) -> None:
        """Test successful function call."""
        assert mp_pipeline._check_operation(op)

    def test_pipeline_check_operation_invalid_op(
        self, mp_pipeline: MedpipePipeline
    ) -> None:
        """Test case when invalid operation is provided."""
        match_expr = f"invalid is not found in sklearn.preprocessing or "
        "sklearn.impute, please check that the operation matches"

        with pytest.raises(ValueError, match=match_expr):
            mp_pipeline._check_operation("invalid")


class TestSetPreprocessingSteps:
    """Test class for the _set_preprocesing_steps function of the
    MedpipePipeline class."""

    def test_pipeline_set_preprocessing_steps_success(
        self, mp_pipeline: MedpipePipeline
    ) -> None:
        """Test successful function call."""
        preprocessing_config = mp_pipeline.medpipe_config.workflow.preprocessing
        pipe = mp_pipeline._set_preprocessing_steps()

        if preprocessing_config:
            assert isinstance(pipe, Pipeline)

            if preprocessing_config.operations is not None:

                for i in range(len(preprocessing_config.operations)):
                    op_config = preprocessing_config.operations[i]

                    # Check the Pipeline steps are correct
                    step = pipe.steps[i]
                    assert isinstance(step, tuple)
                    assert isinstance(step[1], ColumnTransformer)
                    assert step[0] == f"transformer_{i+1}"

                    # Check the ColumnTransformers are correct
                    transformer = step[1].transformers
                    assert isinstance(transformer, list)
                    assert transformer[0][0] == f"op_{i+1}"
                    assert isinstance(
                        transformer[0][1],
                        mp_pipeline._check_operation(op_config.name),
                    )

        else:
            assert pipe is None

    def test_pipeline_set_preprocessing_steps_None(
        self, mp_pipeline: MedpipePipeline
    ) -> None:
        """Test that None is returned correctly."""
        mp_pipeline.medpipe_config.workflow.preprocessing.preprocess = False  # type: ignore
        assert mp_pipeline._set_preprocessing_steps() == None

        mp_pipeline.medpipe_config.workflow.preprocessing = None
        assert mp_pipeline._set_preprocessing_steps() == None

    @pytest.mark.parametrize(
        "operations, columns",
        [
            ([{"name": "OrdinalEncoder", "columns": ["SEX"]}], [["SEX"]]),
            (
                [
                    {"name": "OrdinalEncoder", "columns": ["SEX"]},
                    {"name": "StandardScaler", "columns": ["SEX"]},
                ],
                [["SEX"], ["op_1__SEX"]],
            ),
            (
                [
                    {"name": "OrdinalEncoder", "columns": ["SEX", "PRIOR_CANCER"]},
                    {"name": "StandardScaler", "columns": ["SEX"]},
                    {"name": "PowerTransformer", "columns": ["SEX", "PRIOR_CANCER"]},
                ],
                [
                    ["SEX", "PRIOR_CANCER"],
                    ["op_1__SEX"],
                    ["op_2__op_1__SEX", "remainder__op_1__PRIOR_CANCER"],
                ],
            ),
        ],
    )
    def test_pipeline_set_preprocessing_steps_ct_columns(
        self,
        mp_pipeline: MedpipePipeline,
        operations: list[dict[str, str | list[str]]],
        columns: list[list[str]],
    ) -> None:
        """Test that the ColumnTransformer get the correct column names."""
        preprocessing_config = mp_pipeline.medpipe_config.workflow.preprocessing

        if preprocessing_config:
            preprocessing_config.operations = [
                PreprocessOperationConfig.model_validate(operation)
                for operation in operations
            ]  # Reset operations
            pipe = mp_pipeline._set_preprocessing_steps()

            assert isinstance(pipe, Pipeline)

            for i in range(len(preprocessing_config.operations)):
                # Check the ColumnTransformers columns are correct
                transformer = pipe.steps[i][1].transformers
                assert transformer[0][2] == columns[i]


class TestHasPreprocessor:
    """Test class for the _has_preprocessor function of the
    MedpipePipeline class."""

    def test_pipeline_has_preprocessor_success(
        self, mp_pipeline: MedpipePipeline
    ) -> None:
        """Test successful function call."""
        # Should be true in the default configuration file
        assert mp_pipeline._has_preprocessor() == True

    def test_pipeline_has_preprocessor_False(
        self, mp_pipeline: MedpipePipeline
    ) -> None:
        """Test cases when _has_preprocesor return False."""
        mp_pipeline.medpipe_config.workflow.preprocessing.preprocess = False  # type: ignore

        assert mp_pipeline._has_preprocessor() == False

        mp_pipeline.medpipe_config.workflow.preprocessing = None

        assert mp_pipeline._has_preprocessor() == False


class TestGetDataSets:
    """Test class for the _get_data_sets function of the
    MedpipePipeline class."""

    def test_pipeline_get_data_sets_success(self, mp_pipeline: MedpipePipeline) -> None:
        """Test successful function call."""
        mock_data = load_data(mp_pipeline.medpipe_config.data.path)

        X_train, y_train, X_test, y_test, X_recal, y_recal, groups = (
            mp_pipeline._get_data_sets(mock_data)
        )

        # Create the column list to check
        column_list = mp_pipeline.medpipe_config.data.predictors

        # Check columns are correct
        assert (X_train.columns == column_list).all()
        assert (X_test.columns == column_list).all()

        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)

        if mp_pipeline.medpipe_config.workflow.validation.recalibration_split:
            assert X_recal is not None
            assert y_recal is not None
            assert (X_recal.columns == column_list).all()
            assert len(X_recal) == len(y_recal)

        # Check group_column and X_train
        assert mp_pipeline.medpipe_config.workflow.validation.cross_validation
        if mp_pipeline.medpipe_config.workflow.validation.cross_validation.group_column:
            assert groups is not None
            assert len(groups) == len(X_train)

    def test_pipeline_get_data_sets_no_recal(
        self, mp_pipeline: MedpipePipeline
    ) -> None:
        """Test case when there is no recalibration."""
        mock_data = load_data(mp_pipeline.medpipe_config.data.path)

        validation_config = mp_pipeline.medpipe_config.workflow.validation
        validation_config.recalibration_split = None

        _, _, _, _, X_recal, y_recal, _ = mp_pipeline._get_data_sets(mock_data)

        assert X_recal is None
        assert y_recal is None

    def test_pipeline_get_data_sets_no_groups(
        self, mp_pipeline: MedpipePipeline
    ) -> None:
        """Test case when there are no cross-validation groups."""
        mock_data = load_data(mp_pipeline.medpipe_config.data.path)

        validation_config = mp_pipeline.medpipe_config.workflow.validation
        assert validation_config.cross_validation
        validation_config.cross_validation.group_column = None

        _, _, _, _, _, _, groups = mp_pipeline._get_data_sets(mock_data)

        assert groups is None

    @pytest.mark.parametrize(
        "data",
        [
            3.14,
            42,
            "llama",
            [],
            {},
            (),
            np.array([]),
        ],
    )
    def test_pipeline_get_data_sets_invalid_data(
        self, mp_pipeline: MedpipePipeline, data: Any
    ) -> None:
        """Test case when the data is not pd.DataFrame."""
        match_expr = f"data should be a pd.DataFrame, but got {type(data)}"
        with pytest.raises(TypeError, match=match_expr):
            mp_pipeline._get_data_sets(data)


class TestDropGroupColumns:
    """Test class for the _drop_group_columns function of the
    MedpipePipeline class."""

    def test_pipeline_drop_group_columns_success(
        self,
        mp_pipeline: MedpipePipeline,
        mock_data: MockData,
    ) -> None:
        """Test successful function call."""
        # Get mock data and drop columns
        X_train, X_test, X_recal = mock_data
        X_train, X_test, X_recal, groups = mp_pipeline._drop_group_columns(
            X_train, X_test, X_recal
        )

        # Create the column list to check
        column_list = [
            "SEX",
            "AGE",
            "PRIOR_CANCER",
            "ADMISSION_ACUITY",
            "ADMISSION_SOURCE",
            "CATEGORY_LEVEL_1",
            "OP_SEVERITY",
        ]
        # Check everything is correct
        assert (X_train.columns == column_list).all()
        assert (X_test.columns == column_list).all()
        assert X_recal is not None
        assert (X_recal.columns == column_list).all()
        assert groups is not None
        assert len(groups) == len(X_train)

    def test_pipeline_drop_group_columns_no_recal(
        self,
        mp_pipeline: MedpipePipeline,
        mock_data: MockData,
    ) -> None:
        """Test case when X_recal is None."""
        # Get mock data and drop columns
        X_train, X_test, X_recal = mock_data
        X_recal = None  # Set X_recal to None
        X_train, X_test, X_recal, _ = mp_pipeline._drop_group_columns(
            X_train, X_test, X_recal
        )

        column_list = [
            "SEX",
            "AGE",
            "PRIOR_CANCER",
            "ADMISSION_ACUITY",
            "ADMISSION_SOURCE",
            "CATEGORY_LEVEL_1",
            "OP_SEVERITY",
        ]
        assert X_recal is None
        assert (X_train.columns == column_list).all()
        assert (X_test.columns == column_list).all()

    def test_pipeline_drop_group_columns_no_cv_groups(
        self,
        mp_pipeline: MedpipePipeline,
        mock_data: MockData,
    ) -> None:
        """Test case when cross-validation groups are None."""
        # Get mock data and drop columns
        X_train, X_test, X_recal = mock_data

        assert mp_pipeline.medpipe_config.workflow.validation.cross_validation
        mp_pipeline.medpipe_config.workflow.validation.cross_validation.strategy = (
            "random"  # Set cross-validation strategy to random to have no groups
        )
        X_train, X_test, X_recal, groups = mp_pipeline._drop_group_columns(
            X_train, X_test, X_recal
        )

        column_list = [
            "SEX",
            "AGE",
            "PRIOR_CANCER",
            "ADMISSION_ACUITY",
            "ADMISSION_SOURCE",
            "CATEGORY_LEVEL_1",
            "OP_SEVERITY",
            "DHB_NAME",  # DHB_NAME is not dropped
        ]

        assert groups is None
        assert (X_train.columns == column_list).all()
        assert (X_test.columns == column_list).all()
        assert X_recal is not None
        assert (X_recal.columns == column_list).all()

    def test_pipeline_drop_group_columns_no_test_groups(
        self,
        mp_pipeline: MedpipePipeline,
        mock_data: MockData,
    ) -> None:
        """Test case when test_split groups are None."""
        # Get mock data and drop columns
        X_train, X_test, X_recal = mock_data

        mp_pipeline.medpipe_config.workflow.validation.test_split.strategy = (
            "random"  # Set test split strategy to random to have no groups
        )
        X_train, X_test, X_recal, groups = mp_pipeline._drop_group_columns(
            X_train, X_test, X_recal
        )

        column_list = [
            "SEX",
            "AGE",
            "PRIOR_CANCER",
            "ADMISSION_ACUITY",
            "ADMISSION_SOURCE",
            "CATEGORY_LEVEL_1",
            "OP_SEVERITY",
            "OP_YEAR",  # OP_YEAR is not dropped
        ]

        assert groups is not None
        assert (X_train.columns == column_list).all()
        assert (X_test.columns == column_list).all()
        assert X_recal is not None
        assert (X_recal.columns == column_list).all()

    def test_pipeline_drop_group_columns_no_groups(
        self,
        mp_pipeline: MedpipePipeline,
        mock_data: MockData,
    ) -> None:
        """Test case when test_split and cross-validation groups are None."""
        # Get mock data and drop columns
        X_train, X_test, X_recal = mock_data

        assert mp_pipeline.medpipe_config.workflow.validation.cross_validation
        mp_pipeline.medpipe_config.workflow.validation.test_split.strategy = (
            "random"  # Set test split strategy to random to have no groups
        )
        mp_pipeline.medpipe_config.workflow.validation.cross_validation.strategy = (
            "random"  # Set cross-validation strategy to random to have no groups
        )
        X_train, X_test, X_recal, groups = mp_pipeline._drop_group_columns(
            X_train, X_test, X_recal
        )

        column_list = [
            "SEX",
            "AGE",
            "PRIOR_CANCER",
            "ADMISSION_ACUITY",
            "ADMISSION_SOURCE",
            "CATEGORY_LEVEL_1",
            "OP_SEVERITY",
            "OP_YEAR",  # OP_YEAR is not dropped
            "DHB_NAME",  # DHB_NAME is not dropped
        ]

        assert groups is None
        assert (X_train.columns == column_list).all()
        assert (X_test.columns == column_list).all()
        assert X_recal is not None
        assert (X_recal.columns == column_list).all()


class TestPrepareFeatures:
    """Test class for the _prepare_features function of the
    MedpipePipeline class."""

    def test_pipeline_prepare_features_success(
        self,
        mp_pipeline: MedpipePipeline,
        mock_data: MockData,
    ) -> None:
        """Test successful function call."""
        X_train, _, X_recal = mock_data

        X_train, X_recal = mp_pipeline._prepare_features(X_train, X_recal)

        assert isinstance(X_train, np.ndarray)
        assert isinstance(X_recal, np.ndarray)

        if mp_pipeline.preprocessor:
            check_is_fitted(mp_pipeline.preprocessor)

    def test_pipeline_prepare_features_no_recal(
        self,
        mp_pipeline: MedpipePipeline,
        mock_data: MockData,
    ) -> None:
        """Test case when X_recal is None."""
        X_train, _, X_recal = mock_data
        X_recal = None

        X_train, X_recal = mp_pipeline._prepare_features(X_train, X_recal)

        assert isinstance(X_train, np.ndarray)
        assert X_recal is None

        if mp_pipeline.preprocessor:
            check_is_fitted(mp_pipeline.preprocessor)

    def test_pipeline_prepare_features_no_preprocessing(
        self,
        mp_pipeline: MedpipePipeline,
        mock_data: MockData,
    ) -> None:
        """Test case when there is no preprocessor."""
        X_train, _, X_recal = mock_data
        mp_pipeline.preprocessor = None  # Set preprocessor to None

        X_train, X_recal = mp_pipeline._prepare_features(X_train, X_recal)

        assert isinstance(X_train, np.ndarray)
        assert isinstance(X_recal, np.ndarray)


class TestGetCvGenerator:
    """Test class for the _get_cv_generator function of the
    MedpipePipeline class."""

    @pytest.mark.parametrize("strategy", ["random", "group"])
    def test_pipeline_get_cv_generator_success(
        self, mp_pipeline: MedpipePipeline, strategy: Literal["random", "group"]
    ) -> None:
        """Test successful function call."""
        cv_config = mp_pipeline.medpipe_config.workflow.validation.cross_validation
        assert cv_config
        cv_config.strategy = strategy  # Test both strategies
        cv_generator = mp_pipeline._get_cv_generator()

        if strategy == "random":
            assert isinstance(cv_generator, StratifiedKFold)
        if strategy == "group":
            assert isinstance(cv_generator, GroupKFold)
        assert cv_generator.n_splits == cv_config.n_splits
        assert cv_generator.shuffle == cv_config.shuffle
        assert cv_generator.random_state == cv_config.random_state


class TestCvFit:
    """Test class for the _cv_fit function of the
    MedpipePipeline class."""

    def test_pipeline_cv_fit_success(
        self,
        mp_pipeline: MedpipePipeline,
        cv_data_prep: DataPrep,
    ) -> None:
        """Test successful function call."""
        X_train, y_train, X_recal, y_recal, groups, cv_generator = cv_data_prep
        outcome = "MORTALITY_30D"
        mp_pipeline.folds[outcome] = {}  # Patch because done in run function

        cv_results = mp_pipeline._cv_fit(
            X_train, y_train, outcome, cv_generator, groups, X_recal, y_recal
        )

        check_is_fitted(mp_pipeline.recalibrator[outcome])

        metrics = mp_pipeline.medpipe_config.workflow.evaluation.metrics.metrics
        for metric in metrics:
            assert "test_" + metric in cv_results.keys()

    def test_pipeline_cv_fit_stratified_cv(
        self,
        mp_pipeline: MedpipePipeline,
        cv_data_prep: DataPrep,
    ) -> None:
        """Test successful function call."""
        X_train, y_train, X_recal, y_recal, groups, cv_generator = cv_data_prep
        outcome = "MORTALITY_30D"
        mp_pipeline.folds[outcome] = {}  # Patch because done in run function

        # Change cross-validation generator
        assert mp_pipeline.medpipe_config.workflow.validation.cross_validation
        mp_pipeline.medpipe_config.workflow.validation.cross_validation.strategy = (
            "random"
        )
        cv_generator = mp_pipeline._get_cv_generator()
        groups = None

        cv_results = mp_pipeline._cv_fit(
            X_train, y_train, outcome, cv_generator, groups, X_recal, y_recal
        )

        check_is_fitted(mp_pipeline.recalibrator[outcome])

        metrics = mp_pipeline.medpipe_config.workflow.evaluation.metrics.metrics
        for metric in metrics:
            assert "test_" + metric in cv_results.keys()


class TestCrossValidateAndFit:
    """Test class for the _cross_validate_and_fit function of the
    MedpipePipeline class."""

    def test_pipeline_cross_validate_and_fit_success(
        self,
        mp_pipeline: MedpipePipeline,
        cv_data_prep: DataPrep,
    ) -> None:
        """Test successful function call."""
        X_train, y_train, _, _, groups, cv_generator = cv_data_prep
        outcome = "MORTALITY_30D"
        mp_pipeline.folds[outcome] = {}  # Patch because done in run function

        mp_pipeline._cross_validate_and_fit(
            outcome, X_train, y_train.ravel(), cv_generator, groups
        )

        check_is_fitted(mp_pipeline.predictor[outcome])

    def test_pipeline_cross_validate_and_fit_stratified_cv(
        self,
        mp_pipeline: MedpipePipeline,
        cv_data_prep: DataPrep,
    ) -> None:
        """Test successful function call."""
        X_train, y_train, _, _, groups, cv_generator = cv_data_prep
        outcome = "MORTALITY_30D"
        mp_pipeline.folds[outcome] = {}  # Patch because done in run function

        # Change cross-validation generator
        assert mp_pipeline.medpipe_config.workflow.validation.cross_validation
        mp_pipeline.medpipe_config.workflow.validation.cross_validation.strategy = (
            "random"
        )
        cv_generator = mp_pipeline._get_cv_generator()
        groups = None

        mp_pipeline._cross_validate_and_fit(
            outcome, X_train, y_train.ravel(), cv_generator, groups
        )

        check_is_fitted(mp_pipeline.predictor[outcome])


class TestSaveFoldOutputs:
    """Test class for the _save_fold_outputs function of the
    MedpipePipeline class."""

    def test_pipeline_save_fold_outputs_success(
        self,
        mp_pipeline: MedpipePipeline,
        cv_data_prep: DataPrep,
    ) -> None:
        """Test successful function call."""
        X_train, y_train, _, _, groups, cv_generator = cv_data_prep
        outcome = "MORTALITY_30D"
        mp_pipeline.folds[outcome] = {}  # Patch because done in run function

        cv_results = mp_pipeline._cross_validate_and_fit(
            outcome, X_train, y_train.ravel(), cv_generator, groups
        )

        mp_pipeline._save_fold_outputs(outcome, cv_results, X_train, groups)

        # Check that fold predictions have been saved
        assert groups is not None
        assert outcome in mp_pipeline.predictor_train_outputs.keys()

        for group in groups:
            assert group in mp_pipeline.predictor_train_outputs[outcome]
            assert group in mp_pipeline.folds[outcome]

    def test_pipeline_save_fold_outputs_stratified_cv(
        self,
        mp_pipeline: MedpipePipeline,
        cv_data_prep: DataPrep,
    ) -> None:
        """Test successful function call."""
        X_train, y_train, _, _, groups, cv_generator = cv_data_prep
        outcome = "MORTALITY_30D"
        mp_pipeline.folds[outcome] = {}  # Patch because done in run function

        # Change cross-validation generator
        assert mp_pipeline.medpipe_config.workflow.validation.cross_validation
        mp_pipeline.medpipe_config.workflow.validation.cross_validation.strategy = (
            "random"
        )
        cv_generator = mp_pipeline._get_cv_generator()
        groups = None

        cv_results = mp_pipeline._cross_validate_and_fit(
            outcome, X_train, y_train.ravel(), cv_generator, groups
        )

        mp_pipeline._save_fold_outputs(outcome, cv_results, X_train, groups)

        # Check that fold predictions have been saved
        assert outcome in mp_pipeline.predictor_train_outputs.keys()

        for i in range(
            mp_pipeline.medpipe_config.workflow.validation.cross_validation.n_splits
        ):
            assert i in mp_pipeline.predictor_train_outputs[outcome]
            assert i in mp_pipeline.folds[outcome]


class TestPrintFoldMetrics:
    """Test class for the _print_fold_metrics function of the
    MedpipePipeline class."""

    def test_pipeline_print_fold_metrics_success(
        self,
        capsys,
        mp_pipeline: MedpipePipeline,
        mock_cv_results: dict[str, npt.NDArray],
    ) -> None:
        """Test successful function call."""
        mp_pipeline.metrics = ["auroc", "ici"]
        mp_pipeline.folds = {
            "MORTALITY_30D": {"Auckland": 0, "Christchurch": 1, "Wellington": 2}
        }

        mp_pipeline._print_fold_metrics(mock_cv_results, "MORTALITY_30D")

        # Capture output
        captured = capsys.readouterr()
        msg = (
            "Outcome: MORTALITY_30D\n"
            "  Fold: Auckland\nAUROC: 0.940\nICI: 0.010\n"
            "  Fold: Christchurch\nAUROC: 0.990\nICI: 0.022\n"
            "  Fold: Wellington\nAUROC: 0.952\nICI: 0.001\n"
        )
        assert msg in captured.out

    def test_pipeline_print_fold_metrics_no_recal(
        self,
        capsys,
        mp_pipeline: MedpipePipeline,
        mock_cv_results: dict[str, npt.NDArray],
    ) -> None:
        """Test case when there is no recalibration."""
        mp_pipeline.metrics = ["auroc", "ici"]
        mp_pipeline.folds = {
            "MORTALITY_30D": {"Auckland": 0, "Christchurch": 1, "Wellington": 2}
        }
        mp_pipeline.recalibrator = {}

        mp_pipeline._print_fold_metrics(mock_cv_results, "MORTALITY_30D")

        # Capture output
        captured = capsys.readouterr()
        msg = (
            "Outcome: MORTALITY_30D\n"
            "  Fold: Auckland\nAUROC: 0.940\nICI: 0.010\n"
            "  Fold: Christchurch\nAUROC: 0.990\nICI: 0.022\n"
            "  Fold: Wellington\nAUROC: 0.952\nICI: 0.001\n"
        )
        assert msg in captured.out


class TestExtractFoldResults:
    """Test class for the _extract_fold_results function of the
    MedpipePipeline class."""

    def test_pipeline_extract_fold_results_success(
        self,
        mp_pipeline: MedpipePipeline,
        mock_cv_results: dict[str, npt.NDArray],
    ) -> None:
        """Test successful function call."""
        mp_pipeline.metrics = ["auroc", "ici"]
        results = mp_pipeline._extract_fold_results(mock_cv_results)

        assert results.shape == (2, 3)


class TestRun:
    """Test class for the run function of the MedpipePipeline class."""

    def test_pipeline_run_success(self, mp_pipeline: MedpipePipeline) -> None:
        """Test successful function call."""
        mp_pipeline.run()

    def test_pipeline_run_no_preprocessing(self, mp_pipeline: MedpipePipeline) -> None:
        """Test successful function call with no preprocessing."""
        mp_pipeline.preprocessor = None
        mp_pipeline.run()

    def test_pipeline_run_no_recalibration(self, mp_pipeline: MedpipePipeline) -> None:
        """Test successful function call with no preprocessing."""
        mp_pipeline.recalibrator_method = None
        mp_pipeline.recalibrator = {}
        mp_pipeline.run()

    def test_pipeline_run_no_cv(self, mp_pipeline: MedpipePipeline) -> None:
        """Test successful function call with no cross-validation."""
        mp_pipeline.medpipe_config.top_level.meta.run_mode = "fast"
        mp_pipeline.run()
