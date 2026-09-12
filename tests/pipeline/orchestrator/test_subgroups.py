"""
Tests for MedpipeOrchestrator.extract_stratum_subgroup and
get_subgroup_specs.
"""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from medpipe.pipeline.orchestrator import MedpipeOrchestrator


class TestExtractStratumSubgroup:
    """Unit test suite for MedpipeOrchestrator.extract_stratum_subgroup."""

    @pytest.fixture
    def mock_orchestrator(self) -> MedpipeOrchestrator:
        """Create a lightweight MedpipeOrchestrator instance with mocked dependencies."""
        orchestrator = object.__new__(MedpipeOrchestrator)
        orchestrator.logger = MagicMock()
        return orchestrator

    @pytest.fixture
    def sample_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Provide aligned feature (X) and label (y) DataFrames."""
        indices = pd.Index([101, 102, 103, 104, 105])
        X = pd.DataFrame(
            {
                "AGE": [20, 45, 70, 35, 80],
                "SEX": ["F", "M", "F", "M", "F"],
                "FEAT1": [1.1, 2.2, 3.3, 4.4, 5.5],
            },
            index=indices,
        )
        y = pd.DataFrame(
            {
                "readmission": [0, 1, 0, 1, 1],
                "mortality": [0, 0, 1, 0, 1],
            },
            index=indices,
        )
        return X, y

    def test_extract_subgroup_with_x_and_y_success(
        self,
        mock_orchestrator: MedpipeOrchestrator,
        sample_data: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        """Verify slicing both features (X) and target labels (y) preserves alignment and copy independence."""
        X, y = sample_data

        X_sub, y_sub = mock_orchestrator.extract_stratum_subgroup(
            X=X, column="SEX", group="M", y=y
        )

        assert isinstance(X_sub, pd.DataFrame)
        assert isinstance(y_sub, pd.DataFrame)
        assert len(X_sub) == 2
        assert len(y_sub) == 2

        assert list(X_sub.index) == [102, 104]
        assert list(y_sub.index) == [102, 104]

        # Ensure returned objects are deep copies
        X_sub.loc[102, "FEAT1"] = 999.0
        assert X.loc[102, "FEAT1"] == 2.2

    def test_extract_subgroup_without_y_returns_none(
        self,
        mock_orchestrator: MedpipeOrchestrator,
        sample_data: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        """Verify call with y=None returns (X_subgroup, None)."""
        X, _ = sample_data

        X_sub, y_sub = mock_orchestrator.extract_stratum_subgroup(
            X=X, column="AGE", group=(18, 50), y=None
        )

        assert len(X_sub) == 3
        assert y_sub is None

    def test_extract_subgroup_range_tuple(
        self,
        mock_orchestrator: MedpipeOrchestrator,
        sample_data: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        """Verify continuous numerical group resolution using tuple bounds."""
        X, y = sample_data

        X_sub, y_sub = mock_orchestrator.extract_stratum_subgroup(
            X=X, column="AGE", group=(60, 90), y=y
        )

        assert y_sub is not None
        assert len(X_sub) == 2
        assert list(X_sub.index) == [103, 105]
        assert list(y_sub.index) == [103, 105]

    def test_zero_matches_logs_warning_and_returns_empty_dataframes(
        self,
        mock_orchestrator: MedpipeOrchestrator,
        sample_data: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        """Verify zero matched rows triggers logger warning and returns empty DataFrames."""
        X, y = sample_data

        X_sub, y_sub = mock_orchestrator.extract_stratum_subgroup(
            X=X, column="SEX", group="UNKNOWN", y=y
        )

        assert y_sub is not None
        assert len(X_sub) == 0
        assert len(y_sub) == 0
        assert list(X_sub.columns) == list(X.columns)
        assert list(y_sub.columns) == list(y.columns)

        mock_orchestrator.logger.warning.assert_called_once()  # type: ignore
        log_msg = mock_orchestrator.logger.warning.call_args[0][0]  # type: ignore
        assert "returned 0 samples" in log_msg

    def test_missing_column_raises_key_error(
        self,
        mock_orchestrator: MedpipeOrchestrator,
        sample_data: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        """Verify KeyError is raised when the stratification column is missing from X."""
        X, y = sample_data

        with pytest.raises(KeyError, match="Stratum column 'INVALID_COL' not found"):
            mock_orchestrator.extract_stratum_subgroup(
                X=X, column="INVALID_COL", group="M", y=y
            )


class TestGetSubgroupSpecs:
    """Unit test suite for MedpipeOrchestrator.get_subgroup_specs."""

    def test_get_subgroup_specs_with_strata_and_custom_groups(self):
        """Test resolving subgroup specs when both strata and custom range
        groups are defined."""
        orchestrator = object.__new__(MedpipeOrchestrator)

        fairness_cfg = MagicMock()
        fairness_cfg.strata = ["SEX", "AGE"]
        fairness_cfg.groups = {"AGE": [[18, 50], [51, 120]]}

        orchestrator.config = MagicMock()
        orchestrator.config.workflow.evaluation.fairness = fairness_cfg

        specs = orchestrator.get_subgroup_specs()

        assert specs == {
            "SEX": "SEX",
            "AGE": [[18, 50], [51, 120]],
        }

    def test_get_subgroup_specs_strata_only(self):
        """Test resolving subgroup specs when strata are present without
        custom groups."""
        orchestrator = object.__new__(MedpipeOrchestrator)

        fairness_cfg = MagicMock()
        fairness_cfg.strata = ["SEX", "ETHNICITY"]
        fairness_cfg.groups = None

        orchestrator.config = MagicMock()
        orchestrator.config.workflow.evaluation.fairness = fairness_cfg

        specs = orchestrator.get_subgroup_specs()

        assert specs == {
            "SEX": "SEX",
            "ETHNICITY": "ETHNICITY",
        }

    def test_get_subgroup_specs_missing_fairness_config(self):
        """Test returning empty dict when fairness config or strata list
        is missing."""
        orchestrator = object.__new__(MedpipeOrchestrator)
        orchestrator.config = MagicMock()
        orchestrator.config.workflow.evaluation.fairness = None

        specs = orchestrator.get_subgroup_specs()

        assert specs == {}

    def test_get_subgroup_specs_missing_workflow_or_evaluation(self):
        """Test returning empty dict when workflow or evaluation section is
        missing entirely."""
        orchestrator = object.__new__(MedpipeOrchestrator)
        orchestrator.config = MagicMock()
        orchestrator.config.workflow = None

        specs = orchestrator.get_subgroup_specs()

        assert specs == {}
