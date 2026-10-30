# Quickstart

## MedpipeClassifier simple example

Use `MedpipeClassifier` to create a pipeline for a classification task. The pipeline is defined by a TOML configuration file, which is validated when the pipeline is created.

``` python linenums="1"
{%
    include-markdown "../../examples/minimal_example.py"
%}
```

The configuration file used above is [`examples/default_classifier_config.toml`](https://github.com/Surgical-Recovery-and-Safety-Lab/medpipe/blob/main/examples/default_classifier_config.toml). It shows everything that can be written in a classification configuration, and each block is described in the [configuration guide](../guide/config/index.md). Run the example from the `examples/` folder, since the paths in the configuration are relative to where the script is run.

## MedpipeRegressor simple example

Use `MedpipeRegressor` for a continuous outcome. The pipeline is defined in the same way, by a TOML configuration file.

``` python linenums="1"
{%
    include-markdown "../../examples/minimal_regressor_example.py"
%}
```

The configuration file is [`examples/default_regressor_config.toml`](https://github.com/Surgical-Recovery-and-Safety-Lab/medpipe/blob/main/examples/default_regressor_config.toml). It predicts the length of stay and the blood loss with `OrdBoostRegressor`, which predicts a full distribution of each outcome. The [regression](../guide/config/regression.md) page of the guide lists how a regression configuration differs from a classification one.

## Results

Each run is saved in a new numbered folder, `artifacts/v1`, `artifacts/v2`, and so on. It contains the fitted models, the evaluation results with confidence intervals, the figures, and the configuration and environment of the run. The [run folder](../guide/run_folder.md) page describes each file.

