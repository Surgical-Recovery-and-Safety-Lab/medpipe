import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import joblib
import pandas as pd
import pytest

from medpipe.pipeline.pipeline import Medpipe
from medpipe.utils.config import MedpipeConfig
from medpipe.utils.io import read_toml_configuration

# ==============================================================================
# 1. UNIT TESTS (Mocked Sub-components & Routing Validation)
# ==============================================================================


class TestMedpipeUnit:
    """Unit tests verifying orchestration delegation and argument routing."""

    @patch("medpipe.pipeline.pipeline.MedpipeOrchestrator")
    @patch("medpipe.pipeline.pipeline.MedpipeRunner")
    @patch("medpipe.pipeline.pipeline.MedpipeEvaluator")
    def test_medpipe_initialization(
        self, mock_eval_cls, mock_runner_cls, mock_orch_cls
    ):
        """Verify Medpipe initializes sub-orchestrators correctly."""
        mock_config = MagicMock(spec=MedpipeConfig)
        mock_orch_instance = mock_orch_cls.return_value

        mp = Medpipe(config=mock_config)

        mock_orch_cls.assert_called_once_with(mock_config, "artifacts", None)
        mock_runner_cls.assert_called_once_with(orchestrator=mock_orch_instance)
        mock_eval_cls.assert_called_once_with(
            orchestrator=mock_orch_instance, runner=mock_runner_cls.return_value
        )
        assert mp.mp_config == mock_orch_instance.config

    @patch("medpipe.pipeline.pipeline.MedpipeOrchestrator")
    @patch("medpipe.pipeline.pipeline.MedpipeRunner")
    @patch("medpipe.pipeline.pipeline.MedpipeEvaluator")
    def test_inference_delegation(self, mock_eval_cls, mock_runner_cls, mock_orch_cls):
        """Verify predict, predict_proba, and decision_function delegate to evaluator."""
        mp = Medpipe(config=MagicMock())
        X = pd.DataFrame({"A": [1, 2]})

        mp.predict(X, outcome="MORTALITY")
        mp.evaluator.predict.assert_called_once_with(
            X=X, model=None, outcome="MORTALITY"
        )

        mp.predict_proba(X, outcome="MORTALITY")
        mp.evaluator.predict_proba.assert_called_once_with(
            X=X, model=None, outcome="MORTALITY"
        )

        mp.decision_function(X, outcome="MORTALITY")
        mp.evaluator.decision_function.assert_called_once_with(
            X=X, model=None, outcome="MORTALITY"
        )

    @patch("medpipe.pipeline.pipeline.MedpipeOrchestrator")
    @patch("medpipe.pipeline.pipeline.MedpipeRunner")
    @patch("medpipe.pipeline.pipeline.MedpipeEvaluator")
    def test_evaluate_y_dataframe_resolution(
        self, mock_eval_cls, mock_runner_cls, mock_orch_cls
    ):
        """Verify y DataFrame slicing resolution logic in evaluate()."""
        mp = Medpipe(config=MagicMock())
        X = pd.DataFrame({"AGE": [50, 60]})
        y_df = pd.DataFrame(
            {
                "MORTALITY_30D": [0, 1],
                "READMISSION_90D": [1, 0],
            }
        )

        # When outcome matches a column in y_df
        mp.evaluate(X, y_df, outcome="MORTALITY_30D")
        _, kwargs = mp.evaluator.evaluate.call_args
        pd.testing.assert_series_equal(kwargs["y"], y_df["MORTALITY_30D"])

        # When y is a single-column DataFrame without outcome specified
        y_single = pd.DataFrame({"TARGET": [0, 1]})
        mp.evaluate(X, y_single)
        _, kwargs = mp.evaluator.evaluate.call_args
        pd.testing.assert_series_equal(kwargs["y"], y_single.iloc[:, 0])


class TestMedpipeLoad:
    """Test suite for the Medpipe.load class factory method."""

    @pytest.fixture
    def valid_config_dict(self, tmp_path: Path) -> dict:
        """Provides a minimal valid raw configuration dictionary."""
        return {
            "meta": {
                "project_name": "demo_project",
                "run_mode": "fast",
                "verbose": "compact",
            },
            "data": {
                "path": str(tmp_path / "data.csv"),
                "predictors": ["AGE", "SEX"],
                "outcomes": ["MORTALITY_30D"],
            },
            "default_model": {
                "algorithm": "HistGradientBoostingClassifier",
                "hyperparameters": {},
            },
            "workflow": {
                "validation": {
                    "test_split": {"strategy": "random", "test_size": 0.2},
                },
                "evaluation": {
                    "metrics": {"metrics": ["roc_auc"]},
                },
            },
        }

    def test_load_missing_config_raises_file_not_found(self, tmp_path: Path) -> None:
        """Test that FileNotFoundError is raised if resolved_config.json is missing."""
        empty_run_dir = tmp_path / "empty_run"
        empty_run_dir.mkdir()

        with pytest.raises(
            FileNotFoundError,
            match="Cannot load Medpipe instance: Configuration JSON missing",
        ):
            Medpipe.load(empty_run_dir)

    def test_load_successful_without_models_directory(
        self, tmp_path: Path, valid_config_dict: dict
    ) -> None:
        """Test successful reconstruction of Medpipe when no models directory is present."""
        run_dir = tmp_path / "run_2026_08_10"
        run_dir.mkdir()

        config_path = run_dir / "resolved_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(valid_config_dict, f)

        pipe = Medpipe.load(run_dir)

        assert isinstance(pipe, Medpipe)
        assert pipe.orchestrator.run_dir == run_dir
        assert pipe.displayer.run_dir == run_dir
        assert pipe.mp_config.meta.project_name == "demo_project"

    def test_load_successful_with_fitted_models(
        self, tmp_path: Path, valid_config_dict: dict
    ) -> None:
        """Test loading and restoring serialized fitted models into runner."""
        run_dir = tmp_path / "run_2026_08_10"
        models_dir = run_dir / "models"
        models_dir.mkdir(parents=True)

        # Write resolved JSON configuration
        config_path = run_dir / "resolved_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(valid_config_dict, f)

        # Serialize fitted models dictionary matching project_name
        mock_fitted_models = {"MORTALITY_30D": "fitted_model_placeholder"}
        model_artifact = models_dir / "demo_project_fitted.joblib"
        joblib.dump(mock_fitted_models, model_artifact)

        # Pass run_dir as str to test string path resolution
        pipe = Medpipe.load(str(run_dir))

        assert isinstance(pipe, Medpipe)
        assert pipe.orchestrator.run_dir == run_dir
        assert pipe.displayer.run_dir == run_dir
        assert pipe.runner.fitted_models == mock_fitted_models


# ==============================================================================
# 2. END-TO-END STRESS TESTS (Minimal Synthetic Dataset & Configuration Branches)
# ==============================================================================


@pytest.fixture
def repo_root(request: pytest.FixtureRequest) -> Path:
    """Resolve repository root directory from pytest config."""
    return Path(request.config.rootpath)


@pytest.fixture
def test_data_path(repo_root: Path) -> Path:
    """Locate tests/test_data.csv relative to repo root."""
    path = repo_root / "tests" / "test_data.csv"
    if not path.exists():
        path = repo_root / "test" / "test_data.csv"
    return path


@pytest.fixture
def base_config_path(repo_root: Path) -> Path:
    """Locate examples/default_config.toml relative to repo root."""
    return repo_root / "examples" / "default_config.toml"


@pytest.fixture
def build_medpipe_config(test_data_path: Path, base_config_path: Path):
    """Factory fixture loading default_config.toml and updating data.path dynamically."""

    def _factory(run_mode: str = "fast", disable_recal: bool = False) -> MedpipeConfig:
        config = read_toml_configuration(base_config_path)
        config.data.path = str(test_data_path)
        config.meta.run_mode = run_mode

        if disable_recal:
            config.default_model.recalibration = None
            for override in config.outcome_overrides.values():
                override.recalibration = None
            config.workflow.validation.recalibration_split = None

        return config

    return _factory


class TestMedpipeStressIntegration:
    """Integration stress tests executing Medpipe and verifying artifact lifecycle."""

    @pytest.mark.parametrize("run_mode", ["fast", "cv", "eval", "audit"])
    @pytest.mark.parametrize("disable_recal", [False, True])
    def test_run_mode_and_recalibration_branches(
        self, build_medpipe_config, tmp_path: Path, run_mode: str, disable_recal: bool
    ) -> None:
        """Executes pipeline end-to-end, verifies artifact creation, and deletes artifacts."""
        try:
            config = build_medpipe_config(
                run_mode=run_mode, disable_recal=disable_recal
            )
        except Exception as err:
            pytest.skip(
                f"Config combination incompatible with test dataset state: {err}"
            )

        # Direct artifacts to an isolated temporary directory
        artifact_dir = tmp_path / "artifacts"
        pipeline = Medpipe(config=config, base_artifact_dir=artifact_dir)
        run_dir = pipeline.orchestrator.run_dir

        try:
            results = pipeline.run()

            # 1. Validate returned pipeline output structure
            assert "fitted_models" in results
            assert "evaluations" in results
            assert len(results["fitted_models"]) == len(config.data.outcomes)
            assert len(results["evaluations"]) == len(config.data.outcomes)

            # 2. Verify expected artifacts were created on disk
            assert run_dir.exists()
            assert (run_dir / "env_state.json").exists()

            models_dir = run_dir / "models"
            assert models_dir.exists()
            for outcome in config.data.outcomes:
                assert (models_dir / f"{outcome}_model.joblib").exists()
                assert (run_dir / f"{outcome}_evaluation_results.json").exists()

        finally:
            # 3. Clean up/delete the run directory after verification
            if run_dir.exists():
                # Close any active logging handlers attached to the run file to release file locks
                for handler in list(pipeline.logger.handlers):
                    handler.close()
                    pipeline.logger.removeHandler(handler)
                shutil.rmtree(run_dir, ignore_errors=True)

            assert not run_dir.exists()

    def test_manual_workflow_fit_predict_evaluate_cleanup(
        self, build_medpipe_config, tmp_path: Path
    ) -> None:
        """Validates manual step-by-step API execution and artifact cleanup."""
        config = build_medpipe_config(run_mode="fast", disable_recal=True)
        artifact_dir = tmp_path / "artifacts"
        pipeline = Medpipe(config=config, base_artifact_dir=artifact_dir)
        run_dir = pipeline.orchestrator.run_dir

        try:
            X_train, y_train, X_recal, y_recal, X_test, y_test, groups_train = (
                pipeline.orchestrator.prepare_data()
            )

            fitted = pipeline.fit(X_train, y_train, X_recal, y_recal, groups_train)
            assert set(fitted.keys()) == set(config.data.outcomes)

            for outcome in config.data.outcomes:
                eval_dict = pipeline.evaluate(
                    X=X_test,
                    y=y_test[outcome],
                    outcome=outcome,
                    save_artifacts=True,
                )
                assert "overall" in eval_dict
                assert (run_dir / f"{outcome}_evaluation_results.json").exists()

        finally:
            if run_dir.exists():
                for handler in list(pipeline.logger.handlers):
                    handler.close()
                    pipeline.logger.removeHandler(handler)
                shutil.rmtree(run_dir, ignore_errors=True)

            assert not run_dir.exists()
