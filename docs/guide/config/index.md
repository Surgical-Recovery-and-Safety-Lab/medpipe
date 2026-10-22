# Configuration guide

A medpipe experiment is described by a single [TOML](https://toml.io) file. The file says which data to load, which models to fit, how to split and validate the data, which metrics and figures to produce, and how much to log. Because the whole experiment lives in one file, it is saved with every run and can be reused to reproduce the results.

The reference for everything that can be written in a configuration file is [`examples/default_classifier_config.toml`](https://github.com/Surgical-Recovery-and-Safety-Lab/medpipe/blob/main/examples/default_classifier_config.toml). This guide explains each section of that file and shows the matching snippet.

!!! note "Work in progress"
    The notes on the regression configuration are still being written.

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
| [`[meta]`](meta.md) | Project name, run mode and console verbosity. | Yes |
| [`[data]`](data.md) | Dataset location, predictor columns and outcome columns. | Yes |
| [`[default_model]`](default_model.md) | Algorithm, hyperparameters and recalibration applied to every outcome. | Yes |
| [`[outcome_overrides.<outcome>]`](outcome_overrides.md) | Per-outcome changes to the default model. | No |
| [`[workflow]`](workflow.md) | Preprocessing, data splits, cross-validation, metrics and fairness analysis. | Yes |
| [`[display]`](display.md) | Figure settings. | When `run_mode` is `"audit"` or `"eval"` |

Each section has its own page. The classifier and regressor configurations share the same layout. The differences are summarised in a separate section at the end of this guide.

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
    Catch `pydantic.ValidationError` around the pipeline creation, as in the [quickstart](../../start/quickstart.md), to print configuration errors without a full traceback.
