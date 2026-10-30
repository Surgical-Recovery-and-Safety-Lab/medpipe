# Confidence intervals

The metrics of the evaluation are reported with a confidence interval, for the whole test set and for every subgroup. The intervals come from a bootstrap, which makes no assumption about the distribution of the metric. This page explains how they are computed and what to do for the metrics that use a predictive distribution.

## The bootstrap

For a set of `n` test samples:

1. `n_bootstraps` new sets of `n` samples are drawn with replacement. The metrics that are computed from the same kind of prediction (point predictions and probabilities, or a predictive distribution) share the same draws.
2. Every metric is computed on every new set.
3. The interval is made of the percentiles of these values that leave `(1 - ci_level) / 2` on each side. With the default `ci_level = 0.95`, these are the 2.5th and the 97.5th percentiles.

The point estimate is the metric on the original test set. The draws are seeded by `random_state` of `[workflow]`, so the same configuration gives the same intervals. `n_jobs` of `[workflow]` sets how many processes compute the resamples, for the metrics that need it.

The settings are in [`[workflow.evaluation.metrics]`](config/workflow.md#metrics): `n_bootstraps`, `ci_level` and the metrics themselves. A larger `n_bootstraps` gives steadier bounds at the cost of time. Setting `n_bootstraps = 0` skips the intervals, and the bounds are saved as `NaN`.

### When the bootstrap cannot be computed

* **A resample with a single class.** Some metrics, such as `roc_auc` and `ap`, are not defined when a resample has only one class, which happens with rare outcomes. Those resamples are skipped, and the interval uses the others. If every resample fails, or the test set itself has one class, the interval cannot be computed.
* **A failure in the whole computation.** The evaluation does not stop. A warning is logged, the point estimates are saved, and the bounds are `NaN`. If the point estimates cannot be computed either, they are `NaN` too.

## Metrics that use a predictive distribution

Metrics such as `crps` compare the observed value with the whole predictive distribution of a model like `OrdBoostRegressor`. A distribution object cannot be handed to the bootstrap like a column of numbers, so there are two ways to resample, and a metric uses one of them.

### Metrics with a `per_sample_func`

`crps` is the mean of one score per sample. Its `MetricSpec` has a `per_sample_func` that returns this score for every sample:

```python
per_sample = per_sample_func(y_true, dist)  # shape (n_samples,)
```

The scores are computed once. A resample is then a draw of these scores, and its metric is their mean. This is fast, and it works with any distribution object, since the distribution is never cut into pieces.

### Metrics without one

For any other metric, the pairs of observed value and distribution are resampled, and the function of the metric is run again on each resample. This is exact for any metric, but slower, and the resamples are computed in parallel with `n_jobs`.

It needs the distribution to support row indexing, `dist[indices]`, to build the resampled distribution. The distributions of OrdBoost 0.6.0 and later do. A distribution that does not, such as a frozen SciPy distribution, cannot be used with such a metric, and the interval fails with:

> ['name'] do not define a per_sample_func, so they are bootstrapped by resampling (y, dist) pairs, which requires a distribution supporting row indexing.

To avoid this, give the metric a `per_sample_func`. This is only valid for a metric that is the mean of a per-sample score.

### Registering a metric with a `per_sample_func`

```python
import numpy as np

from medpipe import MetricRegistry
from medpipe.metrics import MetricSpec


def absolute_pit_error(y, dist):
    """Distance of the PIT value of each sample from 0.5."""
    return np.abs(dist.cdf(y) - 0.5)


MetricRegistry.register_spec(
    MetricSpec(
        name="pit_error",
        func=lambda y, dist: float(np.mean(absolute_pit_error(y, dist))),
        response_method="predict_dist",
        display_name="PIT error",
        per_sample_func=absolute_pit_error,
    )
)
```

`func` returns the metric on the whole test set and `per_sample_func` its score for each sample, so the mean of the second is the first. A metric that needs the predictive distribution can only be used with models that predict one, as explained on the [regression](config/regression.md#metrics) page.

### Checks

The number of samples of the distribution has to match the number of observed values, otherwise an error is raised. For some distribution objects this check cannot be made, and the error then comes from the computation of the metric itself.
