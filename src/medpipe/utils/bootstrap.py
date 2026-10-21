"""
Bootstrap resampling helpers shared by the metric and visualisation modules.

"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from joblib import Parallel, delayed


def draw_bootstrap_indices(
    rng: np.random.Generator, n_samples: int, n_bootstraps: int
) -> npt.NDArray:
    """Draw all bootstrap resample indices up front, sequentially, so the
    random draw sequence (and thus reproducibility for a given rng state) is
    independent of how the resamples are later computed or dispatched.

    Parameters
    ----------
    rng : np.random.Generator
        Random generator the resamples are drawn from.
    n_samples : int
        Number of samples in the data being resampled.
    n_bootstraps : int
        Number of resamples to draw.

    Returns
    -------
    npt.NDArray
        Integer array of shape (n_bootstraps, n_samples) whose rows are the
        resample indices, drawn with replacement.

    """
    return rng.integers(0, n_samples, size=(n_bootstraps, n_samples))


def run_bootstraps(
    worker: Callable[..., Any],
    bootstrap_indices: npt.NDArray,
    *args: Any,
    n_jobs: int | None,
    backend: Literal["threading", "loky"],
) -> list[Any]:
    """Apply `worker(idx, *args)` to every bootstrap resample, in parallel
    when `n_jobs` allows it, and drop resamples the worker rejected.

    Parameters
    ----------
    worker : callable
        Module-level function taking a resample's indices followed by `args`
        and returning its result, or `None` if the resample is unusable.
    bootstrap_indices : numpy.ndarray
        Resample indices, one row per resample, as returned by `draw_bootstrap_indices`.
    *args : Any
        Extra positional arguments forwarded unchanged to every `worker`
        call.
    n_jobs : int or None
        Number of parallel jobs (the display section's `n_jobs`).
    backend : {"threading", "loky"}
        Joblib backend. Benchmarks showed numpy-bound resamples (ROC/PR
        curves) gain from threads, since process start-up and pickling
        outweigh their millisecond cost, while resamples dominated by
        Python-level loops (predictive-distribution intervals, spline
        fitting) only gain from processes.

    Returns
    -------
    list
        The non-`None` results, in resample order.

    """
    results = Parallel(n_jobs=n_jobs, backend=backend)(
        delayed(worker)(idx, *args) for idx in bootstrap_indices
    )
    return [r for r in results if r is not None]
