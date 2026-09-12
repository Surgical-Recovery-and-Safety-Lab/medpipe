"""
Tests for MedpipeDisplayer's internal statistical computation helpers:
_compute_roc_data, _compute_precision_recall_data, _compute_reliability_data,
_compute_dca_data, and the _save_figure artifact-persistence helper.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np

from medpipe.visualisation.displayer import MedpipeDisplayer


class TestComputeRocData:
    """Tests for the internal `_compute_roc_data` statistical helper method."""

    def test_compute_roc_data_success_1d_probas(
        self, mock_orchestrator, sample_binary_data
    ) -> None:
        """Test ROC data calculation with 1D predicted probabilities and
        bootstrapping."""
        y_true, probas = sample_binary_data
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)

        fpr, tpr, roc_auc, lower_ci, upper_ci = displayer._compute_roc_data(
            y_true=y_true, probas=probas, n_bootstraps=50
        )

        assert isinstance(fpr, np.ndarray)
        assert isinstance(tpr, np.ndarray)
        assert isinstance(roc_auc, float)
        assert 0.0 <= roc_auc <= 1.0
        assert isinstance(lower_ci, np.ndarray)
        assert isinstance(upper_ci, np.ndarray)
        assert len(lower_ci) == len(fpr)
        assert len(upper_ci) == len(fpr)

    def test_compute_roc_data_2d_probas(
        self, mock_orchestrator, sample_binary_data
    ) -> None:
        """Test ROC data calculation when probabilities are 2D (n_samples, 2)."""
        y_true, probas_1d = sample_binary_data
        probas_2d = np.column_stack((1 - probas_1d, probas_1d))
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)

        _, _, roc_auc, lower_ci, upper_ci = displayer._compute_roc_data(
            y_true=y_true, probas=probas_2d, n_bootstraps=10
        )

        assert isinstance(roc_auc, float)
        assert lower_ci is not None
        assert upper_ci is not None

    def test_compute_roc_data_disabled_bootstraps(
        self, mock_orchestrator, sample_binary_data
    ) -> None:
        """Test that n_bootstraps <= 0 returns None for confidence interval arrays."""
        y_true, probas = sample_binary_data
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)

        fpr, tpr, roc_auc, lower_ci, upper_ci = displayer._compute_roc_data(
            y_true=y_true, probas=probas, n_bootstraps=0
        )

        assert isinstance(fpr, np.ndarray)
        assert isinstance(tpr, np.ndarray)
        assert isinstance(roc_auc, float)
        assert lower_ci is None
        assert upper_ci is None

    def test_compute_roc_data_some_bootstraps_skipped(self, mock_orchestrator) -> None:
        """Test that single-class bootstrap resamples are skipped but the
        method still succeeds when at least one resample has both classes.

        This exact (y, seed, n_bootstraps) combination was verified to
        deterministically produce a mix of skipped and valid resamples.
        """
        y_true = np.array([0, 0, 0, 0, 1])
        probas = np.array([0.1, 0.2, 0.3, 0.4, 0.9])
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)

        _, _, roc_auc, lower_ci, upper_ci = displayer._compute_roc_data(
            y_true=y_true, probas=probas, n_bootstraps=30, random_state=0
        )

        assert isinstance(roc_auc, float)
        assert lower_ci is not None
        assert upper_ci is not None

    def test_compute_roc_data_all_bootstraps_skipped(self, mock_orchestrator) -> None:
        """Test that if every bootstrap resample happens to be single-class,
        the method falls back to None CI bounds instead of raising.

        The RNG is mocked to always resample index 0 (a single class) so
        this is deterministic rather than relying on chance.
        """
        y_true = np.array([0, 1])
        probas = np.array([0.2, 0.8])
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)

        fake_rng = MagicMock()
        fake_rng.choice.return_value = np.array([0, 0])

        with patch("numpy.random.default_rng", return_value=fake_rng):
            _fpr, _tpr, roc_auc, lower_ci, upper_ci = displayer._compute_roc_data(
                y_true=y_true, probas=probas, n_bootstraps=5
            )

        assert isinstance(roc_auc, float)
        assert lower_ci is None
        assert upper_ci is None


class TestComputePrecisionRecallData:
    """Tests for the internal `_compute_precision_recall_data` helper method."""

    def test_compute_precision_recall_data_success_1d_probas(
        self, mock_orchestrator, sample_binary_data
    ) -> None:
        """Test PR data calculation with 1D predicted probabilities and
        bootstrapping."""
        y_true, probas = sample_binary_data
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)

        (
            precision,
            recall,
            ap_score,
            baseline,
            lower_ci,
            upper_ci,
        ) = displayer._compute_precision_recall_data(
            y_true=y_true, probas=probas, n_bootstraps=50
        )

        assert isinstance(precision, np.ndarray)
        assert isinstance(recall, np.ndarray)
        assert isinstance(ap_score, float)
        assert isinstance(baseline, float)
        assert 0.0 <= baseline <= 1.0
        assert lower_ci is not None
        assert upper_ci is not None
        assert len(lower_ci) == len(recall)

    def test_compute_precision_recall_data_2d_probas(
        self, mock_orchestrator, sample_binary_data
    ) -> None:
        """Test PR data calculation when probabilities are 2D (n_samples, 2)."""
        y_true, probas_1d = sample_binary_data
        probas_2d = np.column_stack((1 - probas_1d, probas_1d))
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)

        _, _, ap_score, _baseline, lower_ci, upper_ci = (
            displayer._compute_precision_recall_data(
                y_true=y_true, probas=probas_2d, n_bootstraps=10
            )
        )

        assert isinstance(ap_score, float)
        assert lower_ci is not None
        assert upper_ci is not None

    def test_compute_precision_recall_data_disabled_bootstraps(
        self, mock_orchestrator, sample_binary_data
    ) -> None:
        """Test that n_bootstraps <= 0 returns None for confidence interval arrays."""
        y_true, probas = sample_binary_data
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)

        _, _, ap_score, _baseline, lower_ci, upper_ci = (
            displayer._compute_precision_recall_data(
                y_true=y_true, probas=probas, n_bootstraps=0
            )
        )

        assert isinstance(ap_score, float)
        assert lower_ci is None
        assert upper_ci is None

    def test_compute_precision_recall_data_some_bootstraps_skipped(
        self, mock_orchestrator
    ) -> None:
        """Test that single-class bootstrap resamples are skipped but the
        method still succeeds when at least one resample has both classes."""
        y_true = np.array([0, 0, 0, 0, 1])
        probas = np.array([0.1, 0.2, 0.3, 0.4, 0.9])
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)

        _, _, ap_score, _baseline, lower_ci, upper_ci = (
            displayer._compute_precision_recall_data(
                y_true=y_true, probas=probas, n_bootstraps=30, random_state=0
            )
        )

        assert isinstance(ap_score, float)
        assert lower_ci is not None
        assert upper_ci is not None

    def test_compute_precision_recall_data_all_bootstraps_skipped(
        self, mock_orchestrator
    ) -> None:
        """Test that if every bootstrap resample happens to be single-class,
        the method falls back to None CI bounds instead of raising. The RNG
        is mocked to always resample index 0 for determinism."""
        y_true = np.array([0, 1])
        probas = np.array([0.2, 0.8])
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)

        fake_rng = MagicMock()
        fake_rng.choice.return_value = np.array([0, 0])

        with patch("numpy.random.default_rng", return_value=fake_rng):
            _, _, ap_score, _baseline, lower_ci, upper_ci = (
                displayer._compute_precision_recall_data(
                    y_true=y_true, probas=probas, n_bootstraps=5
                )
            )

        assert isinstance(ap_score, float)
        assert lower_ci is None
        assert upper_ci is None


class TestComputeReliabilityData:
    """Tests for the internal `_compute_reliability_data` helper method."""

    def test_compute_reliability_data_uniform_strategy(
        self, mock_orchestrator, sample_binary_data
    ) -> None:
        """Test calibration curve computation with the default uniform strategy."""
        y_true, probas = sample_binary_data
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)

        prob_true, prob_pred, lower_ci, upper_ci = displayer._compute_reliability_data(
            y_true=y_true, probas=probas, n_bins=5, n_bootstraps=20
        )

        assert isinstance(prob_true, np.ndarray)
        assert isinstance(prob_pred, np.ndarray)
        assert lower_ci is not None
        assert upper_ci is not None
        assert len(lower_ci) == len(prob_pred)

    def test_compute_reliability_data_2d_probas(
        self, mock_orchestrator, sample_binary_data
    ) -> None:
        """Test that 2D probabilities are reduced to the positive-class column."""
        y_true, probas_1d = sample_binary_data
        probas_2d = np.column_stack((1 - probas_1d, probas_1d))
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)

        prob_true_2d, prob_pred_2d, _, _ = displayer._compute_reliability_data(
            y_true=y_true, probas=probas_2d, n_bins=5, n_bootstraps=0
        )
        prob_true_1d, prob_pred_1d, _, _ = displayer._compute_reliability_data(
            y_true=y_true, probas=probas_1d, n_bins=5, n_bootstraps=0
        )

        np.testing.assert_allclose(prob_true_2d, prob_true_1d)
        np.testing.assert_allclose(prob_pred_2d, prob_pred_1d)

    def test_compute_reliability_data_spline_2d_calibrate_output_is_sliced(
        self, mock_orchestrator
    ) -> None:
        """Test the defensive branch where SplineCalib.calibrate() returns a
        2D array: both the main curve and every bootstrap curve must be
        reduced to the positive-class column, exactly like the 2D `probas`
        handling elsewhere in this method.

        SplineCalib is mocked because, in practice, it never actually
        returns a 2D array here (probas is already sliced to 1D before
        fitting) — this exercises the defensive `.ndim == 2` checks as
        written, in case that guarantee ever changes.
        """
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        probas = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6])
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)

        fake_instance = MagicMock()
        two_d_output = np.column_stack(
            [1 - np.linspace(0.0, 1.0, 100), np.linspace(0.0, 1.0, 100)]
        )
        fake_instance.calibrate.return_value = two_d_output

        with patch("splinecalib.SplineCalib", return_value=fake_instance):
            prob_true, _prob_pred, lower_ci, upper_ci = (
                displayer._compute_reliability_data(
                    y_true=y_true, probas=probas, strategy="spline", n_bootstraps=5
                )
            )

        assert prob_true.ndim == 1
        np.testing.assert_allclose(prob_true, np.linspace(0.0, 1.0, 100))
        assert lower_ci is not None
        assert upper_ci is not None
        assert lower_ci.ndim == 1

    def test_compute_reliability_data_quantile_strategy(
        self, mock_orchestrator, sample_binary_data
    ) -> None:
        """Test calibration curve computation with the quantile strategy."""
        y_true, probas = sample_binary_data
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)

        prob_true, prob_pred, _lower_ci, _upper_ci = (
            displayer._compute_reliability_data(
                y_true=y_true,
                probas=probas,
                n_bins=5,
                strategy="quantile",
                n_bootstraps=10,
            )
        )

        assert isinstance(prob_true, np.ndarray)
        assert isinstance(prob_pred, np.ndarray)

    def test_compute_reliability_data_spline_strategy(
        self, mock_orchestrator, sample_binary_data
    ) -> None:
        """Test calibration curve computation with the spline strategy,
        including the bootstrap loop's spline-specific branch."""
        y_true, probas = sample_binary_data
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)

        prob_true, prob_pred, lower_ci, upper_ci = displayer._compute_reliability_data(
            y_true=y_true, probas=probas, strategy="spline", n_bootstraps=5
        )

        assert isinstance(prob_true, np.ndarray)
        assert isinstance(prob_pred, np.ndarray)
        assert len(prob_pred) == 100  # np.linspace(0, 1, 100)
        assert lower_ci is not None
        assert upper_ci is not None

    def test_compute_reliability_data_disabled_bootstraps(
        self, mock_orchestrator, sample_binary_data
    ) -> None:
        """Test that n_bootstraps <= 0 returns None for confidence interval arrays."""
        y_true, probas = sample_binary_data
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)

        _prob_true, _prob_pred, lower_ci, upper_ci = (
            displayer._compute_reliability_data(
                y_true=y_true, probas=probas, n_bootstraps=0
            )
        )

        assert lower_ci is None
        assert upper_ci is None

    def test_compute_reliability_data_empty_prob_pred_skips_bootstrapping(
        self, mock_orchestrator
    ) -> None:
        """Test that an empty calibration curve (e.g. from a degenerate
        input) short-circuits before attempting to bootstrap."""
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)

        with patch(
            "medpipe.visualisation.displayer.calibration_curve",
            return_value=(np.array([]), np.array([])),
        ):
            y_true = np.array([0, 1, 0, 1])
            probas = np.array([0.2, 0.8, 0.3, 0.7])

            _prob_true, prob_pred, lower_ci, upper_ci = (
                displayer._compute_reliability_data(
                    y_true=y_true, probas=probas, n_bootstraps=10
                )
            )

        assert len(prob_pred) == 0
        assert lower_ci is None
        assert upper_ci is None

    def test_compute_reliability_data_some_bootstraps_skipped(
        self, mock_orchestrator
    ) -> None:
        """Test that single-class bootstrap resamples are skipped but the
        method still succeeds when at least one resample has both classes."""
        y_true = np.array([0, 0, 0, 0, 1])
        probas = np.array([0.1, 0.2, 0.3, 0.4, 0.9])
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)

        _prob_true, _prob_pred, lower_ci, upper_ci = (
            displayer._compute_reliability_data(
                y_true=y_true,
                probas=probas,
                n_bins=3,
                n_bootstraps=30,
                random_state=0,
            )
        )

        assert lower_ci is not None
        assert upper_ci is not None

    def test_compute_reliability_data_all_bootstraps_skipped(
        self, mock_orchestrator
    ) -> None:
        """Test that if every bootstrap resample happens to be single-class,
        the method falls back to None CI bounds instead of raising. The RNG
        is mocked to always resample index 0 for determinism."""
        y_true = np.array([0, 1])
        probas = np.array([0.2, 0.8])
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)

        fake_rng = MagicMock()
        fake_rng.choice.return_value = np.array([0, 0])

        with patch("numpy.random.default_rng", return_value=fake_rng):
            _prob_true, _prob_pred, lower_ci, upper_ci = (
                displayer._compute_reliability_data(
                    y_true=y_true, probas=probas, n_bootstraps=5
                )
            )

        assert lower_ci is None
        assert upper_ci is None

    def test_compute_reliability_data_bootstrap_curve_collapses_to_one_point(
        self, mock_orchestrator
    ) -> None:
        """Test the uniform/quantile bootstrap branch that discards a
        resampled calibration curve with fewer than 2 points (too sparse to
        interpolate), by forcing calibration_curve to return a 1-point
        curve on every bootstrap call while the main call still succeeds."""
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        probas = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6])
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)

        from medpipe.visualisation.displayer import calibration_curve as real_curve

        call_count = {"n": 0}

        def flaky_calibration_curve(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                # First call is the main (non-bootstrap) computation.
                return real_curve(*args, **kwargs)
            # Every bootstrap call collapses to a single point.
            return np.array([0.5]), np.array([0.5])

        with patch(
            "medpipe.visualisation.displayer.calibration_curve",
            side_effect=flaky_calibration_curve,
        ):
            prob_true, _prob_pred, lower_ci, upper_ci = (
                displayer._compute_reliability_data(
                    y_true=y_true, probas=probas, n_bootstraps=5
                )
            )

        # All bootstrap iterations were discarded for being too sparse, so
        # no CI could be computed, but the main curve is still returned.
        assert len(prob_true) > 0
        assert lower_ci is None
        assert upper_ci is None


class TestComputeDcaData:
    """Tests for the internal `_compute_dca_data` helper method."""

    def test_compute_dca_data_default_thresholds(
        self, mock_orchestrator, sample_binary_data
    ) -> None:
        """Test DCA computation using the default threshold grid."""
        y_true, probas = sample_binary_data
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)

        thresholds, nb_model, nb_all = displayer._compute_dca_data(
            y_true=y_true, probas=probas
        )

        assert len(thresholds) == 99  # np.linspace(0.01, 0.99, 99)
        assert len(nb_model) == len(thresholds)
        assert len(nb_all) == len(thresholds)

    def test_compute_dca_data_custom_thresholds(
        self, mock_orchestrator, sample_binary_data
    ) -> None:
        """Test DCA computation using an explicitly provided threshold grid."""
        y_true, probas = sample_binary_data
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)
        custom_thresholds = np.linspace(0.1, 0.5, 10)

        thresholds, nb_model, nb_all = displayer._compute_dca_data(
            y_true=y_true, probas=probas, thresholds=custom_thresholds
        )

        assert thresholds is custom_thresholds
        assert len(nb_model) == 10
        assert len(nb_all) == 10

    def test_compute_dca_data_2d_probas(
        self, mock_orchestrator, sample_binary_data
    ) -> None:
        """Test DCA computation reduces 2D probabilities to the positive class."""
        y_true, probas_1d = sample_binary_data
        probas_2d = np.column_stack((1 - probas_1d, probas_1d))
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)

        _, nb_model_2d, _ = displayer._compute_dca_data(
            y_true=y_true, probas=probas_2d
        )
        _, nb_model_1d, _ = displayer._compute_dca_data(
            y_true=y_true, probas=probas_1d
        )

        np.testing.assert_allclose(nb_model_2d, nb_model_1d)


class TestSaveFigure:
    """Tests for the internal `_save_figure` method."""

    def test_save_figure_default_directory(
        self, mock_orchestrator, tmp_path: Path
    ) -> None:
        """Test saving a figure to the default run directory path."""
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)
        fig, _ = plt.subplots()

        saved_path = displayer._save_figure(fig=fig, filename="test_plot")

        expected_path = tmp_path / "plots" / "test_plot.png"
        assert saved_path == expected_path
        assert saved_path.exists()

    def test_save_figure_with_outcome_subdirectory(
        self, mock_orchestrator, tmp_path: Path
    ) -> None:
        """Test saving a figure inside an outcome-specific subdirectory."""
        displayer = MedpipeDisplayer(orchestrator=mock_orchestrator)
        fig, _ = plt.subplots()

        saved_path = displayer._save_figure(
            fig=fig, filename="roc_curve", outcome="mortality"
        )

        expected_path = tmp_path / "plots" / "mortality" / "roc_curve.png"
        assert saved_path == expected_path
        assert saved_path.exists()
