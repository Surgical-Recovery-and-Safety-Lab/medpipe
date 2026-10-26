# Regression

A regression experiment uses the same configuration file layout as a classification experiment, and is run with `MedpipeRegressor` instead of `MedpipeClassifier`:

```python
from medpipe import MedpipeRegressor

pipe = MedpipeRegressor("config.toml")
pipe.run()
```

The file is validated against [`MedpipeRegressorConfig`][medpipe.utils.config.MedpipeRegressorConfig]. This page lists what is different from the classification configuration described in the other pages of this guide. Everything that is not mentioned here works in the same way.

## Summary

| Section | Difference |
| :--- | :--- |
| [`[meta]`](meta.md) | None. |
| [`[data]`](data.md) | None. The outcomes are continuous columns. |
| [`[default_model]`](default_model.md) and [`[outcome_overrides]`](outcome_overrides.md) | No `recalibration`. The algorithms are regressors. |
| [`[workflow.validation]`](workflow.md#workflowvalidation) | Different cross-validation splitters. The recalibration split is accepted but not used. |
| [`[workflow.evaluation]`](workflow.md#workflowevaluation) | Different metrics and defaults. No `cv_splines`. |
| [`[display]`](display.md) | Different defaults and figures. |

## Models

A regression model has an `algorithm` and its `hyperparameters`, and nothing else. A `recalibration` table is not part of a regression model, and it is rejected in `[default_model]` and in the overrides.

```toml
[default_model]
algorithm = "OrdBoostRegressor"

[default_model.hyperparameters]
n_bins = 10
mapper = "quantile"
learning_rate = 0.1
max_iter = 100
random_state = 42
```

The algorithm is looked up in the same places as for classification, so any regressor of scikit-learn, NGBoost or OrdBoost can be used, as well as the models you registered with `ModelRegistry`. Overrides work as described in [`[outcome_overrides]`](outcome_overrides.md), without the recalibration rules.

Two of the models can predict a full distribution of the target instead of a single value. This is what the `crps` metric and most of the figures need.

| Model | Predicts a distribution | Figures |
| :--- | :---: | :---: |
| `OrdBoostRegressor` | Yes | Yes |
| `NGBRegressor` | Yes | No |
| Other regressors, such as `LinearRegression` | No | No |

## Validation

`[workflow.validation]` has the same keys. The differences are:

* **Cross-validation:** the splitters of the `random` and `group` strategies are not stratified, since there are no classes to keep in proportion. They are `KFold` and `GroupKFold`. The `loso` strategy is the same.
* **Recalibration split:** `recalibration_split` is accepted, but regression models are not recalibrated, so nothing uses the rows. They are only removed from the training set. Leave the table out unless you need the rows to be set aside.

## Metrics

`[workflow.evaluation.metrics]` has the same keys, with these differences:

| Key | Default | Description |
| :--- | :--- | :--- |
| `metrics` | `["rmse", "mae"]` | Metrics to compute. |
| `cv_splines` | not available | Only used by the `ici` metric, which is for classifiers. The key is rejected. |

The metrics for regression are:

| Name | Description | Needs |
| :--- | :--- | :--- |
| `rmse` | Root mean squared error. | Any regressor |
| `mae` | Mean absolute error. | Any regressor |
| `crps` | Continuous ranked probability score. | A model that predicts a distribution |

The metrics are not checked against the type of the model when the configuration is validated. A classification metric such as `roc_auc` is accepted, but it cannot be computed on continuous values, so it is reported as `NaN` with a warning. Asking for `crps` with a model that does not predict a distribution stops the run during the evaluation.

The `[workflow.evaluation.fairness]` table works as for classification. The metrics are computed for each subgroup, and the heatmaps are drawn from them.

## Display

`[display]` has the same structure, and the same `overrides`, `outcome_overrides` and `theme` tables, with the following differences. The table is still required when `run_mode` is `"audit"` or `"eval"`.

### Defaults

```toml
[display.defaults]
n_bootstraps = 1000
save = true
show = false
dist_n_bins = 20
dist_yscale = "linear"
coverage_levels = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]
pit_n_bins = 20
```

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `n_bootstraps` | integer | `1000` | Number of bootstrap resamples for the 95% confidence bands of the coverage, sharpness and Winkler score curves. With `0`, the bands are not drawn. |
| `save` | boolean | `true` | Whether the figure is saved in the run folder. |
| `show` | boolean | `false` | Whether the figure is displayed in a window. |
| `n_jobs` | integer | `1` | Number of parallel jobs for the bootstrap of the figures. It is independent of `workflow.n_jobs`. |
| `dist_n_bins` | integer | `20` | Number of bins of the histogram of the target values. |
| `dist_yscale` | string | `"linear"` | Scale of the y-axis of that histogram: `"linear"`, `"log"`, `"symlog"` or `"logit"`. |
| `coverage_levels` | list of numbers | `5` to `95` in steps of 10 | Nominal coverage of the prediction intervals, in percent, evaluated by the coverage, sharpness and Winkler score curves. Each level must be strictly between 0 and 100. |
| `pit_n_bins` | integer | `20` | Number of bins of the PIT histogram. |

The keys of the classification defaults that only concern classifiers, `n_bins`, `strategy` and `cv_splines`, are rejected. Unlike for classification, other keys are rejected too.

### Figures

| Plot type | Aliases | Figure | File | Keys that have an effect |
| :--- | :--- | :--- | :--- | :--- |
| `coverage` | `coverage_curve` | Share of true values inside the prediction intervals | `<outcome>_coverage.png` | `n_bootstraps`, `n_jobs`, `coverage_levels`, `save`, `show` |
| `sharpness` | `sharpness_curve` | Mean width of the prediction intervals | `<outcome>_sharpness.png` | `n_bootstraps`, `n_jobs`, `coverage_levels`, `save`, `show` |
| `winkler` | `winkler_curve`, `winkler_score` | Winkler score of the intervals | `<outcome>_winkler.png` | `n_bootstraps`, `n_jobs`, `coverage_levels`, `save`, `show` |
| `marginal_calibration` | `marginal_calib` | Marginal calibration curve | `<outcome>_marginal_calibration.png` | `save`, `show` |
| `pit_histogram` | `pit` | Histogram of the probability integral transform | `<outcome>_pit_histogram.png` | `pit_n_bins`, `save`, `show` |
| `data_distribution` | `distribution`, `dist` | Histogram of the target values | `<outcome>_data_distribution.png` | `dist_n_bins`, `dist_yscale`, `save`, `show` |
| `strata_heatmap` | `heatmap` | Subgroup heatmap, one per metric | `<metric>_strata_heatmap.png` | `save`, `show` |

The classification plot types, such as `roc` and `calibration`, are rejected. The settings are combined in the same order as for classification: the defaults, then `overrides`, then `outcome_overrides`, then the arguments given in Python.

```toml
[display.overrides.coverage]
coverage_levels = [50, 80, 95]

[display.outcome_overrides.LOS_DAYS.pit]
pit_n_bins = 10
```

The first six figures are only drawn for models that predict a distribution through OrdBoost, as shown in the table of [models](#models). For any other model, a warning is logged for the outcome and its figures are skipped, but the run goes on and the evaluation results are saved as usual. The subgroup heatmaps only need the evaluation results, so they are drawn for every model when `[workflow.evaluation.fairness]` is defined. Errors in the units of the target (`rmse`, `mae` and `crps`) have a colour scale that follows the data, instead of the fixed scale used for metrics between 0 and 1.

## Using the pipeline

`MedpipeRegressor` has the same `run`, `fit`, `evaluate`, `plot_all` and `load` methods as `MedpipeClassifier`. The prediction methods are different:

| `MedpipeClassifier` | `MedpipeRegressor` |
| :--- | :--- |
| `predict`: class labels | `predict`: predicted values |
| `predict_proba`: class probabilities | `predict_dist`: predictive distribution, for the models that have one |
| `decision_function` | none |

## Complete example

```toml
[meta]
project_name = "length_of_stay"
run_mode = "audit"
verbose = 1

[data]
path = "data/cohort.csv"
predictors = ["AGE", "SEX", "OP_SEVERITY"]
outcomes = ["LOS_DAYS"]

[default_model]
algorithm = "OrdBoostRegressor"

[default_model.hyperparameters]
n_bins = 10
mapper = "quantile"
learning_rate = 0.1
max_iter = 100
random_state = 42

[workflow]
random_state = 42
n_jobs = 2

[workflow.validation.test_split]
strategy = "random"
test_size = 0.2

[workflow.validation.cross_validation]
strategy = "random"
n_splits = 3

[workflow.evaluation.metrics]
metrics = ["rmse", "mae", "crps"]
n_bootstraps = 200

[workflow.evaluation.fairness]
strata = ["SEX"]

[display.defaults]
n_bootstraps = 200
coverage_levels = [50, 80, 95]
```
