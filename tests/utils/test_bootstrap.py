"""Tests for the shared bootstrap resampling helpers."""

import numpy as np
import pytest

from medpipe.utils.bootstrap import draw_bootstrap_indices, run_bootstraps


def _sum_if_positive(idx: np.ndarray, values: np.ndarray) -> float | None:
    total = float(values[idx].sum())
    return total if total > 0 else None


class TestDrawBootstrapIndices:
    def test_shape_and_range(self) -> None:
        idx = draw_bootstrap_indices(np.random.default_rng(0), 7, 5)

        assert idx.shape == (5, 7)
        assert idx.min() >= 0
        assert idx.max() < 7

    def test_reproducible_for_same_seed(self) -> None:
        a = draw_bootstrap_indices(np.random.default_rng(3), 10, 4)
        b = draw_bootstrap_indices(np.random.default_rng(3), 10, 4)

        np.testing.assert_array_equal(a, b)

    def test_matches_sequential_draws(self) -> None:
        """Drawing all resamples at once consumes the rng like one draw
        per resample, so seeded results do not depend on the batching."""
        batched = draw_bootstrap_indices(np.random.default_rng(5), 9, 6)
        rng = np.random.default_rng(5)
        sequential = [rng.integers(0, 9, size=9) for _ in range(6)]

        np.testing.assert_array_equal(batched, np.stack(sequential))


class TestRunBootstraps:
    @pytest.mark.parametrize("backend", ["threading", "loky"])
    def test_drops_none_results_and_keeps_order(self, backend) -> None:
        values = np.array([1.0, -1.0])
        indices = np.array([[0, 0], [1, 1], [0, 1], [0, 0]])

        results = run_bootstraps(
            _sum_if_positive, indices, values, n_jobs=2, backend=backend
        )

        assert results == [2.0, 2.0]

    def test_identical_across_n_jobs(self) -> None:
        values = np.arange(10, dtype=float)
        indices = draw_bootstrap_indices(np.random.default_rng(1), 10, 12)

        serial = run_bootstraps(
            _sum_if_positive, indices, values, n_jobs=1, backend="threading"
        )
        parallel = run_bootstraps(
            _sum_if_positive, indices, values, n_jobs=2, backend="threading"
        )

        assert serial == parallel
