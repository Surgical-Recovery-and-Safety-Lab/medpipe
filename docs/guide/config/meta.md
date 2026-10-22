# `[meta]`

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

## Run modes

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

## Verbosity

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
