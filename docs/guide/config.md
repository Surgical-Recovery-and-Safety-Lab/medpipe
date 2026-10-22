# Configuration guide

A medpipe experiment is described by a single [TOML](https://toml.io) file. The file says which data to load, which models to fit, how to split and validate the data, which metrics and figures to produce, and how much to log. Because the whole experiment lives in one file, it is saved with every run and can be reused to reproduce the results.

The reference for everything that can be written in a configuration file is [`examples/default_classifier_config.toml`](https://github.com/Surgical-Recovery-and-Safety-Lab/medpipe/blob/main/examples/default_classifier_config.toml). This guide explains each section of that file and shows the matching snippet.

!!! note "Work in progress"
    The section on the display and the notes on the regression configuration are still being written.

## Overview

### Loading a configuration

Pass the path of the file to `MedpipeClassifier` (or `MedpipeRegressor`):

```python
from medpipe import MedpipeClassifier

pipe = MedpipeClassifier("config.toml")
pipe.run()
```

The file is read and validated when the pipeline is created, before any data is loaded. You can also pass an already built [`MedpipeClassifierConfig`][medpipe.utils.config.MedpipeClassifierConfig] object instead of a path.

### Sections

The file is organised in top-level tables, one per topic:

| Section | Purpose | Required |
| :--- | :--- | :--- |
| [`[meta]`](#meta) | Project name, run mode and console verbosity. | Yes |
| [`[data]`](#data) | Dataset location, predictor columns and outcome columns. | Yes |
| [`[default_model]`](#default_model) | Algorithm, hyperparameters and recalibration applied to every outcome. | Yes |
| [`[outcome_overrides.<outcome>]`](#outcome_overrides) | Per-outcome changes to the default model. | No |
| [`[workflow]`](#workflow) | Preprocessing, data splits, cross-validation, metrics and fairness analysis. | Yes |
| `[display]` | Figure settings. | When `run_mode` is `"audit"` or `"eval"` |

Each section is described in turn below. The classifier and regressor configurations share the same layout. The differences are summarised in a separate section at the end of this guide.

### Validation

A configuration is checked as a whole, so a mistake is reported with the full path of the offending key:

```text
1 validation error for MedpipeClassifierConfig
meta.run_mode
  Input should be 'fast', 'eval', 'cv' or 'audit' [type=literal_error, input_value='audti', input_type=str]
```

This is a Pydantic `ValidationError`. The checks that apply to every section are:

* **Unknown keys are rejected.** A misspelt key is an error rather than being silently ignored. The only exceptions are `[workflow.preprocessing]` operations, which accept the constructor arguments of the transformer, and `[display.defaults]`, which accepts extra plot arguments.
* **Required keys must be present.** A key listed as required in the tables below has no default.
* **Rules that involve several sections** (for example, a recalibration method requires a recalibration split) are reported with a message that names both.

!!! tip
    Catch `pydantic.ValidationError` around the pipeline creation, as in the [quickstart](../start/quickstart.md), to print configuration errors without a full traceback.

---

## `[meta]`

General information about the run.

```toml
[meta]
project_name = "medpipe_classifier_demo"
run_mode = "audit"
verbose = 1
```

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `project_name` | string | required | Name of the project. It cannot be empty. It is used to name the saved pipeline file, `<project_name>_fitted.joblib`. |
| `run_mode` | string | `"audit"` | Which stages of the pipeline are run. See [run modes](#run-modes). |
| `verbose` | string, integer or boolean | `"compact"` | How much is printed to the console. See [verbosity](#verbosity). |

### Run modes

The run mode controls whether the models are tuned with cross-validation and whether figures are produced. Models are always fitted on the training set, evaluated on the test set, and saved.

| `run_mode` | Cross-validation | Figures | Typical use |
| :--- | :---: | :---: | :--- |
| `"fast"` | No | No | Quick checks while setting up a configuration. |
| `"cv"` | Yes | No | Tuning and comparing models without producing figures. |
| `"eval"` | No | Yes | Producing figures for models whose settings are already chosen. |
| `"audit"` | Yes | Yes | The full run, for reporting. |

The run mode also decides which sections must be present:

* `[workflow.validation.cross_validation]` is required in `"cv"` and `"audit"`, the modes that run cross-validation.
* `[display]` is required in `"audit"` and `"eval"`.

!!! note
    A list-valued hyperparameter (for example `learning_rate = [0.1, 0.05]`) is a grid to search over when `grid_search = true` in a mode that runs cross-validation. In `"fast"` and `"eval"` there is no search, and the first value of the list is used.

Every run is written to a new, numbered folder (`artifacts/v1`, `artifacts/v2`, and so on). The folder always contains the models, the evaluation results, the resolved configuration, the environment description and the execution log. Cross-validation results (`CV/`) are added in `"cv"` and `"audit"`, and figures (`plots/`) in `"eval"` and `"audit"`.

### Verbosity

`verbose` only changes what is printed to the console. The execution log saved in the run folder always records everything, down to the debug level.

| Level | Accepted values | Console output |
| :--- | :--- | :--- |
| Quiet | `"quiet"`, `"warning"`, `0`, `false` | Warnings and errors only. |
| Compact | `"compact"`, `"progress"`, `1` | The main steps of the run (data loading, fitting, evaluation). |
| Info | `"info"`, `"detailed"`, `2`, `true` | Every informational message. |
| Debug | `"debug"`, `3` | Everything, including internal details. |

To change the verbosity for a single run without editing the file, pass `verbose_override` when creating the pipeline:

```python
pipe = MedpipeClassifier("config.toml", verbose_override="debug")
```

---

## `[data]`

Where the dataset is, which columns the models learn from, and which columns they predict.

```toml
[data]
path = "../tests/test_data.csv"
predictors = [
  "AGE",
  "SEX",
  "PRIOR_CANCER",
  "ADMISSION_ACUITY",
  "ADMISSION_SOURCE",
  "CATEGORY_LEVEL_1",
  "OP_SEVERITY",
]
outcomes = ["MORTALITY_30D", "ANY_COMP"]
```

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `path` | string | required | Path to the dataset file. It must have a file extension. |
| `predictors` | list of strings | required | Columns used as model inputs. |
| `outcomes` | list of strings | required | Columns to predict. One model is fitted for each outcome. |
| `kwargs` | table | `{}` | Extra keyword arguments passed to the file reader. See [reader arguments](#reader-arguments). |

### Dataset file

`path` is used as written, so a relative path is relative to the directory the script is run from, not to the configuration file. The example uses `../tests/test_data.csv`, which works when it is run from the `examples/` folder.

The file is read with the pandas reader that matches its extension. The extension is not case-sensitive.

| Extension | Reader |
| :--- | :--- |
| `.csv`, `.txt` | `pandas.read_csv` |
| `.tsv` | `pandas.read_csv` with a tab separator |
| `.parquet`, `.pq` | `pandas.read_parquet` |
| `.feather` | `pandas.read_feather` |
| `.xlsx`, `.xls` | `pandas.read_excel` |
| `.json` | `pandas.read_json` |
| `.jsonl` | `pandas.read_json` with `lines=True` |
| `.pkl`, `.pickle` | `pandas.read_pickle` |

!!! note
    Reading Excel files needs the `openpyxl` package, which is not installed with medpipe. Install it with `pip install openpyxl`. Only load pickle files that you created yourself or trust, as unpickling can run arbitrary code.

Other formats can be supported by registering a reader with [`DataLoaderRegistry`][medpipe.utils.io.DataLoaderRegistry].

The file is not opened when the configuration is validated. A missing file, a directory, or an unsupported extension is reported when the data is loaded at the start of the run.

### Predictors and outcomes

Only the columns that the configuration refers to are kept from the file. All other columns are dropped when the data is loaded. Besides `predictors` and `outcomes`, the configuration can refer to columns in the following places, and these columns must exist in the file too:

* the `group_column` of a split or of cross-validation (see `[workflow.validation]`);
* the `strata` of the fairness analysis (see `[workflow.evaluation.fairness]`).

These extra columns are used to split the data or to describe subgroups. They are never given to the model as inputs unless they are also listed in `predictors`. In the example, `OP_YEAR` and `DHB_NAME` are used to split the data, and they are not among the predictors.

The following rules are checked:

* A column cannot be both a predictor and an outcome. Otherwise the model would be given the answer, and the configuration is rejected with the list of overlapping columns.
* Every column the configuration refers to must be present in the file. A missing column stops the run, before any model is fitted, with the list of missing columns.

Each outcome gets its own model. The models can share the same settings or be configured separately, as described in [`[outcome_overrides]`](#outcome_overrides).

### Reader arguments

Anything under `[data.kwargs]` is passed to the pandas reader chosen for the file. This is useful for options such as the sheet of an Excel file or the separator of a text file:

```toml
[data]
path = "data/cohort.xlsx"
predictors = ["AGE", "SEX"]
outcomes = ["MORTALITY_30D"]

[data.kwargs]
sheet_name = "Cohort"
```

The arguments are not checked when the configuration is validated, so an argument the reader does not accept is reported when the data is loaded.

---

## `[default_model]`

The model fitted for every outcome, and the optional recalibration applied to it afterwards. Each outcome gets its own fitted copy, so outcomes share these settings but not the fitted model.

```toml
[default_model]
algorithm = "HistGradientBoostingClassifier"

[default_model.hyperparameters]
learning_rate = 0.1
loss = "log_loss"
max_features = 0.2

[default_model.recalibration]
recalibrate = true
method = "isotonic"

[default_model.recalibration.hyperparameters]
y_min = 0.0
y_max = 1.0
out_of_bounds = "clip"
```

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `algorithm` | string | required | Name of the model class. See [available models](../api/models.md#available-models). |
| `hyperparameters` | table | `{}` | Arguments passed to the model constructor. |
| `recalibration` | table | none | Recalibration applied after the model is fitted. Omit it to fit the model without recalibration. |

### Algorithm

`algorithm` is the exact class name of an estimator from scikit-learn (`sklearn.ensemble`, `sklearn.linear_model`, `sklearn.isotonic`), NGBoost or OrdBoost, or of a model you registered with `ModelRegistry`. The [models page](../api/models.md) lists where names are looked up and how to register your own.

The name is looked up when the configuration is validated, so a misspelt name is reported before any data is loaded. A model registered with `ModelRegistry` is accepted if it is registered before the pipeline is created.

### Hyperparameters

Every key under `[default_model.hyperparameters]` is passed to the constructor of the model. A key the model does not accept is reported when the model is created.

A hyperparameter can be a list of values to search over:

```toml
[default_model.hyperparameters]
learning_rate = [0.1, 0.05]
max_depth = 10
```

When `grid_search = true` in `[workflow.validation.cross_validation]` and the run mode runs cross-validation, every combination of the listed values is tried and the best model is kept. Scalar values are held fixed. Without a search, the first value of a list is used. See the [run modes](#run-modes).

### Recalibration

Recalibration adjusts the predicted probabilities of a fitted classifier so that they match the observed frequencies better. The model is first fitted on the training set, then a calibrator is fitted on a separate recalibration set.

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `recalibrate` | boolean | required | Whether to recalibrate the fitted model. |
| `method` | string | `"isotonic"` | `"isotonic"` (isotonic regression), `"sigmoid"` (Platt scaling) or `"temperature"` (temperature scaling). |
| `hyperparameters` | table | `{}` | Reserved for hyperparameters of the recalibration method. **Currently not used.** See the note below. |

!!! note
    The `hyperparameters` of the recalibration are accepted by the configuration but are not used yet: they are not passed to the recalibration method, and any values are ignored. A future version may pass them directly to the recalibrator, for example `y_min`, `y_max` and `out_of_bounds` for the isotonic method. Until then, only `recalibrate` and `method` have an effect.

The recalibration set is defined by `[workflow.validation.recalibration_split]`. A `recalibration` table in the default model requires that split, even when `recalibrate = false`. To disable recalibration, either omit the `recalibration` table or set `recalibrate = false` and keep the split.

---

## `[outcome_overrides]`

The default model is used for every outcome unless an outcome is overridden. An override is written as a table named after the outcome, for a column listed in `data.outcomes`:

```toml
[outcome_overrides.ANY_COMP]
algorithm = "GradientBoostingClassifier"

[outcome_overrides.ANY_COMP.hyperparameters]
learning_rate = [0.1, 0.05]
max_depth = 10

[outcome_overrides.ANY_COMP.recalibration]
recalibrate = false
```

An override has the same keys as `[default_model]`, and all of them are optional. Only the keys that are written change the default model, so an override can, for example, change a single hyperparameter or only switch the recalibration off:

```toml
[outcome_overrides.ANY_COMP.hyperparameters]
max_depth = 5

[outcome_overrides.MORTALITY_30D.recalibration]
method = "sigmoid"
```

The name of an outcome that is not in `data.outcomes` is rejected. Outcomes without an override use the default model unchanged.

### How an override is combined with the default

The settings of an override are applied on top of the default model, one group of settings at a time.

**Algorithm and hyperparameters**

* If the override uses the same algorithm as the default, or has no `algorithm`, its hyperparameters are merged into the default ones. Keys that are present in both take the override value, and the other default keys are kept.
* If the override uses a different algorithm, the default hyperparameters are discarded and only the hyperparameters written in the override are used. This avoids passing arguments that the new model does not accept.

For example, with the default model above:

| Override | Resulting hyperparameters |
| :--- | :--- |
| `algorithm = "HistGradientBoostingClassifier"`, `learning_rate = 0.5` | `learning_rate = 0.5`, `loss = "log_loss"`, `max_features = 0.2` |
| `algorithm = "RandomForestClassifier"`, `n_estimators = 50` | `n_estimators = 50` |
| `algorithm = "RandomForestClassifier"` | none, so the model defaults are used |

**Recalibration**

* If the override has no `recalibration` table, the default recalibration is kept, even when the algorithm changes.
* If the override has the same `method` as the default, or no `method`, the keys it sets replace the default ones. Its `hyperparameters` are merged into the default ones.
* If the override has a different `method`, the default recalibration hyperparameters are discarded and only those written in the override are used. `recalibrate` is kept from the default unless the override sets it.
* If the default model has no recalibration, an override that adds one must set `recalibrate`.
* Setting `recalibrate = false` in an override turns the recalibration off for that outcome.

In the example above, `ANY_COMP` uses a different algorithm and disables recalibration. `MORTALITY_30D` has no override and uses the default model with isotonic recalibration.

Whenever the default model has a `recalibration` table, or an override turns recalibration on for an outcome, `[workflow.validation.recalibration_split]` must be defined.

---

## `[workflow]`

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

`random_state` seeds the random train/test and recalibration splits, the shuffling in cross-validation, and the bootstrap resampling of the confidence intervals. Running the same configuration on the same data therefore gives the same splits and intervals. It is not passed to the models. To fix the randomness of a model, set the model's own `random_state` in its [hyperparameters](#hyperparameters).

`n_jobs` sets the parallelism of cross-validation, grid search and the bootstrap resampling. It does not affect the plots.

### `[workflow.preprocessing]`

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
| `name` | string | Class name of the transformer. See [available preprocessing operations](../api/data.md#available-preprocessing-operations). |
| `columns` | list of strings | Predictors the operation is applied to. It cannot be empty. |
| any other key | any | Passed to the constructor of the transformer, such as `with_mean` above. |

The operations are applied one after the other, each one to its own columns. All the other columns are passed on unchanged. In the example, the four categorical predictors are first encoded as numbers, and `CATEGORY_LEVEL_1` and `AGE` are then scaled. A column can be used by several operations.

Some points to keep in mind:

* `columns` must be listed in `data.predictors`. An operation on any other column is rejected when the configuration is validated, with a message that names the operation and the columns.
* `columns` always refers to the original name of a predictor. A later operation cannot refer to a column that an earlier operation replaced with several columns, such as the output of a `OneHotEncoder`.
* The operations are fitted as part of the model, so during cross-validation they are fitted on the training folds only.
* The operation names and their arguments are only checked when the data is prepared. An unknown name or an argument the transformer does not accept stops the run at that point.

### `[workflow.validation]`

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

#### `test_split`

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

#### `recalibration_split`

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

The recalibration split follows the same logic as the test split, and has to be consistent with it. For example, with `test_size = 0.3` and `recalibration_size = 0.2`, the test set has 30% of the data, the recalibration set has 20% of the remaining 70%, and the training set has the rest. The recalibration split is required whenever [recalibration](#recalibration) is enabled for a model.

#### `cross_validation`

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

* **`grid_search = true`:** every combination of the list-valued [hyperparameters](#hyperparameters) is tried. The best combination is chosen using the **first** metric of the `metrics` list, and the model is refitted with it on the whole training set.
* **Otherwise:** the cross-validation only reports the scores. The model is then fitted on the whole training set, using the first value of every list-valued hyperparameter.

The fold scores are saved in the `CV/` folder of the run. Cross-validation is only run when `run_mode` is `"cv"` or `"audit"`.

### `[workflow.evaluation]`

How the fitted models are evaluated on the test set.

#### `metrics`

```toml
[workflow.evaluation.metrics]
metrics = ["roc_auc", "log_loss"]
n_bootstraps = 50
ci_level = 0.95
```

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `metrics` | list of strings | `["roc_auc", "ici"]` | Metrics to compute. See the [list of available metrics](../api/metrics.md#available-metrics). |
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

#### `fairness`

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
