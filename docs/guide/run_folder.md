# The run folder

Every pipeline writes what it produces in its own numbered folder. The folder holds the configuration, the environment, the fitted models, the results and the figures of the run, so a run can be inspected, restored and reproduced later.

## Where it is

The folder is created when the pipeline is created, not when it is run. It is named `v1` in the first free place of the base folder, `v2` for the next one, and so on, and a folder of an earlier run is never overwritten. The base folder is `artifacts` in the current directory, unless `base_artifact_dir` is given:

```python
from medpipe import MedpipeClassifier

pipe = MedpipeClassifier("config.toml", base_artifact_dir="runs")
print(pipe.run_dir)  # runs/v1
```

## Layout

This is the folder of a classification run in the `audit` mode with two outcomes, `MORTALITY_30D` and `ANY_COMP`:

```text
artifacts/v1/
├── execution.log
├── env/
│   ├── config.toml
│   ├── env_state.json
│   └── resolved_config.json
├── models/
│   ├── MORTALITY_30D_model.joblib
│   ├── ANY_COMP_model.joblib
│   └── medpipe_classifier_demo_fitted.joblib
├── CV/
│   ├── MORTALITY_30D_cv_results.csv
│   ├── MORTALITY_30D_cv_summary.json
│   ├── ANY_COMP_cv_results.csv
│   └── ANY_COMP_cv_summary.json
├── results/
│   ├── MORTALITY_30D_evaluation_results.json
│   └── ANY_COMP_evaluation_results.json
└── plots/
    ├── MORTALITY_30D/
    │   └── MORTALITY_30D_roc_curve.png, ...
    ├── ANY_COMP/
    ├── roc_auc_strata_heatmap.png
    └── log_loss_strata_heatmap.png
```

The `CV/` folder is only there when cross-validation was run, and `plots/` when figures were drawn. The table says which run modes write what.

| Content | `fast` | `cv` | `eval` | `audit` |
| :--- | :---: | :---: | :---: | :---: |
| `env/`, `execution.log`, `models/`, `results/` | Yes | Yes | Yes | Yes |
| `CV/` | | Yes | | Yes |
| `plots/` | | | Yes | Yes |

## Files

### `env/`

| File | Content |
| :--- | :--- |
| `resolved_config.json` | The validated configuration with all the defaults filled in, and `resolved_models`, the model of each outcome once the overrides are applied. This is the file `load()` restores a pipeline from. |
| `config.toml` | A copy of the TOML file the pipeline was created from. It is not written when the pipeline is created from a configuration object. |
| `env_state.json` | The description of the environment, listed below. |

`env_state.json` contains:

| Key | Content |
| :--- | :--- |
| `timestamp_utc` | When the pipeline was created. |
| `config_hash` | SHA-256 of the resolved configuration, which is identical for identical configurations. |
| `dataset_hash` | SHA-256 of the dataset file, or `FILE_NOT_FOUND`. A changed dataset gives a different hash. |
| `python_version`, `platform` | The Python version and the operating system. |
| `git_commit_hash` | The commit of the repository the code was run from, or `null` outside a repository. |

The versions of the installed packages are not recorded. If an exact environment matters, save the output of `pip freeze` alongside the run.

### `execution.log`

Every message of the run, down to the debug level, whatever the `verbose` setting of the console. Each line holds the time, the logger, the level, the place in the code and the message.

### `models/`

| File | Content |
| :--- | :--- |
| `<outcome>_model.joblib` | The fitted model of one outcome: the preprocessing and the estimator in a scikit-learn pipeline, wrapped by the recalibration if there is one. |
| `<project_name>_fitted.joblib` | A dictionary with the models of all outcomes. `load()` reads this file. |

The files are written with [joblib](https://joblib.readthedocs.io), which uses pickle. Only load files from runs you trust, since loading can run code. A model also needs compatible versions of scikit-learn, and of OrdBoost or NGBoost, to be loaded.

### `results/`

`<outcome>_evaluation_results.json` holds the metrics on the test set, with their confidence intervals. It is written by `run()` and by `evaluate()` when `save_artifacts` is true.

```json
{
  "outcome": "ANY_COMP",
  "overall": {
    "roc_auc": {"point_estimate": 0.57, "ci_lower": 0.41, "ci_upper": 0.72}
  },
  "strata": {
    "SEX": {
      "F": {"roc_auc": {"point_estimate": 0.61, "ci_lower": 0.42, "ci_upper": 0.74}},
      "M": {"roc_auc": {"point_estimate": 0.27, "ci_lower": 0.17, "ci_upper": 0.43}}
    },
    "AGE": {
      "[18, 50]": {"roc_auc": {"point_estimate": 0.39, "ci_lower": 0.19, "ci_upper": 0.62}}
    }
  }
}
```

* `overall` has one entry per metric of `[workflow.evaluation.metrics]`.
* `strata` has the same metrics for every subgroup of `[workflow.evaluation.fairness]`. A range is named by its two ends, `[18, 50]`. It is empty without a fairness configuration.
* The bounds come from the bootstrap, as explained in [confidence intervals](confidence_intervals.md). They are `NaN` (written as `NaN` in the file) when the intervals were not computed.

### `CV/`

Two files per outcome, written by the cross-validation:

| File | Content |
| :--- | :--- |
| `<outcome>_cv_results.csv` | One row per fit. Without a grid search, one row per fold with the fit and score times and a `test_<metric>` column. With a grid search, one row per combination of hyperparameters with scikit-learn's `cv_results_`: the parameters, the score of every fold (`split0_test_<metric>`, ...), their mean and standard deviation, and the rank. |
| `<outcome>_cv_summary.json` | The mean and standard deviation of each metric over the folds. For a grid search, those of the selected hyperparameters. |

The scores follow the scikit-learn convention that a larger value is better, so error and loss metrics are negative. A `log_loss` of 0.4 is saved as -0.4. The names of the parameters are prefixed with the step of the pipeline, `classifier__` or `regressor__`.

### `plots/`

The figures of each outcome are in a folder named after the outcome, and the subgroup heatmaps are directly in `plots/`. The names of the files are listed with the [classification](config/display.md) and [regression](config/regression.md#display) figures.

## Restoring a run

`MedpipeClassifier.load(run_dir)` and `MedpipeRegressor.load(run_dir)` rebuild a pipeline from `env/resolved_config.json` and the models of the run. What the restored pipeline saves from then on is written in an `eval` folder inside the folder of the run, so the original files are never changed. An example is shown in the [examples](../examples.md#reuse-a-fitted-pipeline).

## Reproducing a run

To run an experiment again, create a pipeline from `env/config.toml` (or from `resolved_config.json` with `MedpipeClassifierConfig.model_validate`). With the same data, the same `random_state` and the same package versions, the splits and the intervals are the same. The `config_hash` and the `dataset_hash` of the new run tell whether the configuration and the data are the ones of the original.

The `random_state` of `[workflow]` is not given to the models. To reproduce a model that has some randomness, set its own `random_state` in the hyperparameters.

The [TRIPOD+AI reporting](tripod_ai.md) page maps these files to the items of the reporting guideline.
