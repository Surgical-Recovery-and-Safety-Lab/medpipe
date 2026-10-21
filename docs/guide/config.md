# Configuration guide

A medpipe experiment is described by a single [TOML](https://toml.io) file. The file says which data to load, which models to fit, how to split and validate the data, which metrics and figures to produce, and how much to log. Because the whole experiment lives in one file, it is saved with every run and can be reused to reproduce the results.

The reference for everything that can be written in a configuration file is [`examples/default_classifier_config.toml`](https://github.com/Surgical-Recovery-and-Safety-Lab/medpipe/blob/main/examples/default_classifier_config.toml). This guide explains each section of that file and shows the matching snippet.

!!! note "Work in progress"
    The sections on models, the workflow and the display are still being written.

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
| `[default_model]` | Algorithm, hyperparameters and recalibration applied to every outcome. | Yes |
| `[outcome_overrides.<outcome>]` | Per-outcome changes to the default model. | No |
| `[workflow]` | Preprocessing, data splits, cross-validation, metrics and fairness analysis. | Yes |
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

Each outcome gets its own model. The models can share the same settings or be configured separately, as described in the model sections below.

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
