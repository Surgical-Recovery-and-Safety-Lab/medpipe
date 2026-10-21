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
| `[data]` | Dataset location, predictor columns and outcome columns. | Yes |
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
