#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline class helper functions test suites.
"""

from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from fixtures import (
    DataPrep,
    MockData,
    cv_data_prep,
    example_config_dir,
    mock_cv_results,
    mock_data,
    mp_pipeline,
    shield_local_filesystem,
)
from sklearn.base import check_is_fitted
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline

from medpipe._types import TransformedData
from medpipe.pipeline.pipeline import MedpipePipeline
from medpipe.utils.config import PreprocessOperationConfig
from medpipe.utils.io import load_data, read_toml_configuration

# ==============================================================================
# Test classes for helper functions
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
    """Test class for the get_data_sets function of the
    MedpipePipeline class."""

    def test_pipeline_get_data_sets_success(self, mp_pipeline: MedpipePipeline) -> None:
        """Test successful function call."""
        mock_data = load_data(mp_pipeline.medpipe_config.data.path)

        X_train, y_train, X_test, y_test, X_recal, y_recal, groups = (
            mp_pipeline.get_data_sets(mock_data)
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

        _, _, _, _, X_recal, y_recal, _ = mp_pipeline.get_data_sets(mock_data)

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

        _, _, _, _, _, _, groups = mp_pipeline.get_data_sets(mock_data)

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
            mp_pipeline.get_data_sets(data)

    def test_pipeline_get_data_sets_missing_column(
        self, mp_pipeline: MedpipePipeline, mock_data: MockData
    ) -> None:
        """Test case when columns are not in data."""
        with pytest.raises(
            ValueError, match="Some outcomes or predictors are not in the data"
        ):
            mp_pipeline.get_data_sets(pd.concat(mock_data))


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
        assert X_test is not None
        assert (X_test.columns == column_list).all()
        assert X_recal is not None
        assert (X_recal.columns == column_list).all()
        assert groups is not None
        assert len(groups) == len(X_train)

    def test_pipeline_drop_group_columns_no_cv(
        self,
        mp_pipeline: MedpipePipeline,
        mock_data: MockData,
    ) -> None:
        """Test case when no cross-validation parameters are provided."""
        # Get mock data and drop columns
        X_train, X_test, X_recal = mock_data
        mp_pipeline.medpipe_config.workflow.validation.cross_validation = None
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
            "DHB_NAME",  # Not dropped
        ]
        # Check everything is correct
        assert (X_train.columns == column_list).all()
        assert X_test is not None
        assert (X_test.columns == column_list).all()
        assert X_recal is not None
        assert (X_recal.columns == column_list).all()
        assert groups is None

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
        assert X_test is not None
        assert (X_test.columns == column_list).all()

    def test_pipeline_drop_group_columns_no_test(
        self,
        mp_pipeline: MedpipePipeline,
        mock_data: MockData,
    ) -> None:
        """Test case when X_test is None."""
        # Get mock data and drop columns
        X_train, _, _ = mock_data
        X_train, X_test, X_recal, _ = mp_pipeline._drop_group_columns(
            X_train, None, None
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
        assert (X_train.columns == column_list).all()
        assert X_test is None
        assert X_recal is None

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
        assert X_test is not None
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
        assert X_test is not None
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
        assert X_test is not None
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

        with pytest.raises(ValueError, match="No preprocessor was found"):
            X_train, X_recal = mp_pipeline._prepare_features(X_train, X_recal)


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


class TestPrepareData:
    """Test class for the _prepare_data function of
    the MedpipePipeline class."""

    def test_pipeline_prepare_data_success_with_fit(
        self, mp_pipeline: MedpipePipeline, mock_data: MockData
    ) -> None:
        """Test successful function call is True."""
        X_train, _, X_recal = mock_data

        X_train, X_recal = mp_pipeline._prepare_data(X_train, X_recal, fit=True)

        assert mp_pipeline.preprocessor is not None
        check_is_fitted(mp_pipeline.preprocessor)
        assert isinstance(X_train, np.ndarray)
        assert isinstance(X_recal, np.ndarray)

    def test_pipeline_prepare_data_success_no_fit(
        self, mp_pipeline: MedpipePipeline, mock_data: MockData
    ) -> None:
        """Test successful function call when fit is False."""
        X_train, _, X_recal = mock_data

        _ = mp_pipeline.fit_transform(
            X_train.drop(["DHB_NAME", "OP_YEAR"], axis=1)
        )  # Fit before calling _prepare_data
        X_train, X_recal = mp_pipeline._prepare_data(X_train, X_recal, fit=False)

        assert isinstance(X_train, np.ndarray)
        assert isinstance(X_recal, np.ndarray)

    @pytest.mark.parametrize("X_recal, call_count", [(None, 1), (pd.DataFrame({}), 2)])
    def test_pipeline_prepare_data_success_no_preprocess(
        self,
        mocker,
        mp_pipeline: MedpipePipeline,
        mock_data: MockData,
        X_recal: TransformedData | None,
        call_count: int,
    ) -> None:
        """Test successful function call with no preprocessing."""
        if X_recal is not None:
            X_train, _, X_recal = mock_data
        else:
            X_train, _, _ = mock_data

        # Patch functions
        convert_patch = mocker.patch("medpipe.pipeline.pipeline.convert_dtypes")

        mp_pipeline.preprocessor = None
        _, _ = mp_pipeline._prepare_data(X_train, X_recal)
        assert convert_patch.call_count == call_count

    @pytest.mark.parametrize("X", [3.14, 42, "llama", [], {}, ()])
    def test_pipeline_prepare_data_incorrect_X(
        self, X: Any, mp_pipeline: MedpipePipeline
    ) -> None:
        """Test case when X is of incorrect type."""
        match_expr = f"Input X should be pd.DataFrame, but got {type(X)}"
        with pytest.raises(TypeError, match=match_expr):
            mp_pipeline._prepare_data(X, pd.DataFrame({}))

    @pytest.mark.parametrize("X_recal", [3.14, 42, "llama", [], {}, ()])
    def test_pipeline_prepare_data_incorrect_X_recal(
        self, X_recal: Any, mp_pipeline: MedpipePipeline
    ) -> None:
        """Test case when X_recal is of incorrect type."""
        match_expr = f"Input X_recal should be pd.DataFrame, but got {type(X_recal)}"
        with pytest.raises(TypeError, match=match_expr):
            mp_pipeline._prepare_data(pd.DataFrame({}), X_recal)


class TestPrintTestMetrics:
    """Test class for the _print_test_metrics function of
    the MedpipePipeline class."""

    @pytest.mark.parametrize(
        "metrics, results, recal_results, count",
        [
            (["log_loss"], np.array([0.5]), np.array([0.8]), 1),
            (["log_loss"], np.array([0.5]), None, 0),
            (["log_loss", "accuracy"], np.array([0.5, 0.6]), np.array([0.8, 0.7]), 2),
        ],
    )
    def test_print_test_metrics_success(
        self,
        capsys,
        mp_pipeline: MedpipePipeline,
        metrics: list[str],
        results: npt.NDArray,
        recal_results: npt.NDArray | None,
        count: int,
    ) -> None:
        """Test successful function call."""
        msg = "Outcome: MORTALITY_30D\n"

        mp_pipeline.metrics = metrics
        mp_pipeline._print_test_metrics(results, "MORTALITY_30D", recal_results)

        captured = capsys.readouterr()
        printed = captured.out

        assert msg in printed
        assert printed.count("  Recalibrated:") == count


class TestValidatePredictInputs:
    """Test class for the _validate_predict_inputs function of
    the MedpipePipeline class."""

    @pytest.mark.parametrize(
        "outcomes, estimator_type",
        [
            ("MORTALITY_30D", "predictor"),
            (["ANY_COMP"], "recalibrator"),
            ("all", "predictor"),
            (["ANY_COMP", "MORTALITY_30D"], ["predictor", "recalibrator"]),
            (["ANY_COMP", "MORTALITY_30D"], "predictor"),
        ],
    )
    def test_pipeline_validate_predict_inputs_success(
        self,
        mp_pipeline: MedpipePipeline,
        outcomes: str | list[str],
        estimator_type: Literal["predictor", "recalibrator"] | list[str],
    ) -> None:
        """Test successful function call."""
        valid_outcomes, estimators = mp_pipeline._validate_predict_inputs(
            outcomes, estimator_type
        )

        assert isinstance(valid_outcomes, list)
        assert isinstance(estimators, list)
        assert len(valid_outcomes) == len(estimators)

    @pytest.mark.parametrize(
        "outcomes, estimator_type",
        [
            ("MORTALITY_30D", "predictor"),
            ("ANY_COMP", "recalibrator"),
            (["ANY_COMP", "MORTALITY_30D"], "predictor"),
            (["ANY_COMP", "MORTALITY_30D"], "recalibrator"),
        ],
    )
    def test_pipeline_validate_predict_estimator_type_list_check(
        self,
        mp_pipeline: MedpipePipeline,
        outcomes: str | list[str],
        estimator_type: Literal["predictor", "recalibrator"],
    ) -> None:
        """Test contents of estimators when a string is passed."""
        _, estimators = mp_pipeline._validate_predict_inputs(outcomes, estimator_type)

        for est_type in estimators:
            assert est_type == estimator_type

    def test_pipeline_validate_predict_inputs_outcomes_all_check(
        self,
        mp_pipeline: MedpipePipeline,
    ) -> None:
        """Test contents of valid_outcomes when outcomes is all."""
        valid_outcomes, _ = mp_pipeline._validate_predict_inputs("all", "predictor")

        assert valid_outcomes == mp_pipeline.outcomes

    @pytest.mark.parametrize(
        "outcomes",
        [(3.12, 42, (), {})],
    )
    def test_pipeline_validate_predict_inputs_incorrect_outcomes(
        self, mp_pipeline: MedpipePipeline, outcomes: Any
    ) -> None:
        """Test case when outcomes is invalid."""
        match_expr = (
            "Input outcomes should be a string, 'all', or a list of strings, "
            f"but got {type(outcomes)}"
        )
        with pytest.raises(TypeError, match=match_expr):
            mp_pipeline._validate_predict_inputs(outcomes, "predictor")

    @pytest.mark.parametrize(
        "estimator_type",
        [(3.12, 42, (), {}, "llama")],
    )
    def test_pipeline_validate_predict_inputs_incorrect_estimator_types(
        self, mp_pipeline: MedpipePipeline, estimator_type: Any
    ) -> None:
        """Test case when estimator_types is invalid."""
        match_expr = (
            "Input estimator_type should be a 'predictor', 'recalibrator' "
            f"or list of strings, but got {type(estimator_type)}"
        )
        with pytest.raises(TypeError, match=match_expr):
            mp_pipeline._validate_predict_inputs("all", estimator_type)

    @pytest.mark.parametrize(
        "outcomes, estimator_type",
        [
            (["MORTALITY_30D", "ANY_COMP"], ["predictor"]),
            (["ANY_COMP"], ["recalibrator", "recalibrator"]),
        ],
    )
    def test_pipeline_validate_predict_inputs_mismatch_len(
        self,
        mp_pipeline: MedpipePipeline,
        outcomes: str | list[str],
        estimator_type: Literal["predictor", "recalibrator"] | list[str],
    ) -> None:
        """Test case when outcomes and estimator_type length mismatch."""
        match_expr = (
            "Inputs outcomes and estimator_type should be the same length, "
            f"but got {len(outcomes)} and {len(estimator_type)}"
        )
        with pytest.raises(ValueError, match=match_expr):
            mp_pipeline._validate_predict_inputs(outcomes, estimator_type)
