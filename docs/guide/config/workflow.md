# `[workflow]`

How the data is prepared, split and validated, and how the fitted models are evaluated. This is the largest section, so it is divided into sub-tables, each described below.

```toml
[workflow]
random_state = 42
n_jobs = 2

[workflow.preprocessing]
# ...

[workflow.validation.test_split]
# ...

[workflow.evaluation.metrics]
# ...
```

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `random_state` | integer | `42` | Seed for the random parts of the pipeline. Must be 0 or more. |
| `n_jobs` | integer | `1` | Number of parallel jobs. Use `-1` for all processors. |
| [`preprocessing`](#workflowpreprocessing) | table | none | Transformations applied to the predictors. |
| [`validation`](#workflowvalidation) | table | required | How the data is split and cross-validated. |
| [`evaluation`](#workflowevaluation) | table | required | Metrics, confidence intervals and subgroup analysis. |

`random_state` seeds the random train/test and recalibration splits, the shuffling in cross-validation, and the bootstrap resampling of the confidence intervals. Running the same configuration on the same data therefore gives the same splits and intervals. It is not passed to the models. To fix the randomness of a model, set the model's own `random_state` in its [hyperparameters](default_model.md#hyperparameters).

`n_jobs` sets the parallelism of cross-validation, grid search and the bootstrap resampling. It does not affect the plots.

## `[workflow.preprocessing]`

Transformations that are applied to the predictors before they reach the model.

```toml
[workflow.preprocessing]
preprocess = true
operations = [
  { name = "OrdinalEncoder", columns = ["SEX", "ADMISSION_ACUITY", "ADMISSION_SOURCE", "CATEGORY_LEVEL_1"] },
  { name = "StandardScaler", columns = ["CATEGORY_LEVEL_1", "AGE"], with_mean = true },
]
```

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `preprocess` | boolean | none | Whether the operations are applied. When it is `false` or the table is omitted, the predictors are given to the model unchanged. |
| `operations` | list of tables | none | The operations, applied in order. Required when `preprocess` is `true`. |

Each operation is a table with the following keys:

| Key | Type | Description |
| :--- | :--- | :--- |
| `name` | string | Class name of the transformer. See [available preprocessing operations](../../api/data.md#available-preprocessing-operations). |
| `columns` | list of strings | Predictors the operation is applied to. It cannot be empty. |
| any other key | any | Passed to the constructor of the transformer, such as `with_mean` above. |

The operations are applied one after the other, each one to its own columns. All the other columns are passed on unchanged. In the example, the four categorical predictors are first encoded as numbers, and `CATEGORY_LEVEL_1` and `AGE` are then scaled. A column can be used by several operations.

Some points to keep in mind:

* `columns` must be listed in `data.predictors`. An operation on any other column is rejected when the configuration is validated, with a message that names the operation and the columns.
* `columns` always refers to the original name of a predictor. A later operation cannot refer to a column that an earlier operation replaced with several columns, such as the output of a `OneHotEncoder`.
* The operations are fitted as part of the model, so during cross-validation they are fitted on the training folds only.
* The operation names and their arguments are only checked when the data is prepared. An unknown name or an argument the transformer does not accept stops the run at that point.

## `[workflow.validation]`

How the data is divided. The test set is set aside first. If a recalibration split is defined, the recalibration set is taken from what remains, and the rest is the training set. Cross-validation is then run on the training set only.

```text
all data ──► test set
         └─► remaining ──► recalibration set (optional)
                       └─► training set ──► cross-validation folds
```

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `test_split` | table | required | The held-out test set. |
| `recalibration_split` | table | none | The set used to recalibrate the models. Required when recalibration is used. |
| `cross_validation` | table | none | Cross-validation on the training set. Required when `run_mode` is `"cv"` or `"audit"`. |

### `test_split`

```toml
[workflow.validation.test_split]
strategy = "group"
group_column = "OP_YEAR"
values = [2024]
```

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `strategy` | string | `"random"` | `"random"` or `"group"`. |
| `test_size` | float | none | Fraction of the data used for testing, strictly between 0 and 1. Required for `"random"`. |
| `group_column` | string | none | Column that defines the groups. Required for `"group"`. |
| `values` | list of strings or integers | none | Values of `group_column` that go to the test set. Required for `"group"`. |

* With the `"random"` strategy, rows are assigned to the test set at random, using `random_state`. The split is not stratified.
* With the `"group"` strategy, every row whose `group_column` value is in `values` is in the test set, and all the other rows are not. This is useful for temporal or geographical validation, for example to test on the most recent year or on a hospital that was not used for training.

### `recalibration_split`

```toml
[workflow.validation.recalibration_split]
strategy = "group"
group_column = "OP_YEAR"
values = [2023]
```

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `strategy` | string | required | `"random"` or `"group"`. It must be the same as the test split. |
| `recalibration_size` | float | none | Fraction of the data left after the test split that is used for recalibration, strictly between 0 and 1. Required for `"random"`. |
| `group_column` | string | none | Required for `"group"`. It must be the same as in the test split. |
| `values` | list of strings or integers | none | Values of `group_column` that go to the recalibration set. Required for `"group"`. None of them can also be in the test split. |

The recalibration split follows the same logic as the test split, and has to be consistent with it. For example, with `test_size = 0.3` and `recalibration_size = 0.2`, the test set has 30% of the data, the recalibration set has 20% of the remaining 70%, and the training set has the rest. The recalibration split is required whenever [recalibration](default_model.md#recalibration) is enabled for a model.

### `cross_validation`

```toml
[workflow.validation.cross_validation]
strategy = "group"
grid_search = true
group_column = "DHB_NAME"
n_splits = 3
shuffle = true
```

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `strategy` | string | required | `"random"`, `"group"` or `"loso"`. |
| `n_splits` | integer | none | Number of folds, 2 or more. Required for `"random"` and `"group"`. Not used for `"loso"`. |
| `group_column` | string | none | Column that defines the groups. Required for `"group"` and `"loso"`. |
| `shuffle` | boolean | `true` | Whether to shuffle the data before it is divided into folds. The shuffling uses `random_state`. |
| `grid_search` | boolean | none | Whether to search over the list-valued hyperparameters. |

The strategies are:

| Strategy | Folds | Classifiers | Regressors |
| :--- | :--- | :--- | :--- |
| `"random"` | `n_splits` folds of random rows | `StratifiedKFold`, which keeps the share of positives in every fold | `KFold` |
| `"group"` | `n_splits` folds in which a group is never split between folds | `StratifiedGroupKFold` | `GroupKFold` |
| `"loso"` | One fold per group, each holding out a single group (leave one group out) | `LeaveOneGroupOut` | `LeaveOneGroupOut` |

The models are scored on every metric in `workflow.evaluation.metrics.metrics`. What happens next depends on `grid_search`:

* **`grid_search = true`:** every combination of the list-valued [hyperparameters](default_model.md#hyperparameters) is tried. The best combination is chosen using the **first** metric of the `metrics` list, and the model is refitted with it on the whole training set.
* **Otherwise:** the cross-validation only reports the scores. The model is then fitted on the whole training set, using the first value of every list-valued hyperparameter.

The fold scores are saved in the `CV/` folder of the run. Cross-validation is only run when `run_mode` is `"cv"` or `"audit"`.

## `[workflow.evaluation]`

How the fitted models are evaluated on the test set.

### `metrics`

```toml
[workflow.evaluation.metrics]
metrics = ["roc_auc", "log_loss"]
n_bootstraps = 50
ci_level = 0.95
```

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `metrics` | list of strings | `["roc_auc", "ici"]` | Metrics to compute. See the [list of available metrics](../../api/metrics.md#available-metrics). |
| `n_bootstraps` | integer | `200` | Number of bootstrap resamples used to compute the confidence intervals. Use `0` to skip the intervals. |
| `ci_level` | float | `0.95` | Confidence level of the intervals, between 0 and 1. |
| `cv_splines` | integer | `3` | Number of internal folds used by the spline fit of the `ici` metric, 2 or more. Lower values are faster. It has no effect on other metrics. |

Every metric is computed on the test set with a bootstrapped confidence interval. The names are checked when the configuration is validated, so a metric you registered with `MetricRegistry` is accepted if it is registered before the pipeline is created.

The results are saved for each outcome in `results/<outcome>_evaluation_results.json`, with the point estimate and the bounds of the interval for every metric:

```json
{
  "outcome": "MORTALITY_30D",
  "overall": {
    "roc_auc": {"point_estimate": 0.61, "ci_lower": 0.42, "ci_upper": 0.74}
  },
  "strata": {}
}
```

The `strata` entry has the same values for each subgroup of the [fairness analysis](#fairness).

With `n_bootstraps = 0` the bootstrap is skipped: only the point estimates are computed, and `ci_lower` and `ci_upper` are saved as `NaN`.

### `fairness`

Optional. Computes the metrics on subgroups of the test set, to check whether the model performs equally across patient groups. The results are also drawn as heatmaps when figures are produced.

```toml
[workflow.evaluation.fairness]
strata = ["SEX", "AGE"]
groups = { "AGE" = [[18, 50], [51, 120]] }
```

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `strata` | list of strings | required | Columns that define the subgroups. |
| `groups` | table | none | Ranges to use for some of the strata. |

For every column in `strata`:

* If the column is not in `groups`, there is one subgroup for each distinct value of the column. In the example, `SEX` gives one subgroup per sex.
* If the column is in `groups`, each entry is a `[minimum, maximum]` range, and both ends are included. In the example, `AGE` gives the subgroups `[18, 50]` and `[51, 120]`. Every entry must have exactly two values.

A key of `groups` must also be in `strata`, and its list cannot be empty. Ranges are for numeric columns. To make subgroups of a categorical column, leave it out of `groups`.

The strata columns have to exist in the data, but they do not have to be predictors. A column that is only used for the fairness analysis, for example a hospital, is read from the data but never given to the model. A column that is missing stops the run before any model is fitted.
