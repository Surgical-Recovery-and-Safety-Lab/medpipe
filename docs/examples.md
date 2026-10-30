# Examples

The scripts below are in the [`examples/`](https://github.com/Surgical-Recovery-and-Safety-Lab/medpipe/tree/main/examples) folder of the repository, next to the configuration files they use. Run them from that folder, since the paths in the configurations are relative to where a script is run:

```bash
cd examples
python manual_workflow.py
```

Every run saves its models, results and figures in a new `artifacts/vN` folder. The examples use the `fast` run mode so that they finish quickly. See the [run modes](guide/config/meta.md#run-modes) for the other modes.

## Run an experiment

The minimal examples create the pipeline from a TOML file and run it. They are described in the [quickstart](start/quickstart.md).

* [`minimal_example.py`](https://github.com/Surgical-Recovery-and-Safety-Lab/medpipe/blob/main/examples/minimal_example.py) runs a classification.
* [`minimal_regressor_example.py`](https://github.com/Surgical-Recovery-and-Safety-Lab/medpipe/blob/main/examples/minimal_regressor_example.py) runs a regression.

## Go through the steps yourself

`run()` prepares the data, fits the models, evaluates them and draws the figures. Each step can also be called on its own, for example to look at the predictions or to evaluate the models on other data. The configuration of the example is changed in Python before it is validated, to use the `fast` run mode.

``` python linenums="1" title="manual_workflow.py"
{%
    include-markdown "../examples/manual_workflow.py"
%}
```

* `prepare_data()` loads the dataset and splits it as described in `[workflow.validation]`. The recalibration set is `None` when no recalibration split is defined.
* `fit()` fits one model per outcome, and `models` holds them by outcome name.
* `evaluate()` returns the metrics with their confidence intervals, overall and for the subgroups it is given. The results are also saved in the `results/` folder of the run.
* `plot_all()` draws the figures of an outcome. The settings of `[display]` are used, and the arguments given here take precedence.

## Reuse a fitted pipeline

The folder of a run holds everything needed to restore the pipeline: the resolved configuration and the fitted models. `load()` returns a pipeline that is ready to predict and evaluate. What it saves from then on goes in an `eval` folder inside the folder of the run, so the original results are not overwritten.

``` python linenums="1" title="load_saved_run.py"
{%
    include-markdown "../examples/load_saved_run.py"
%}
```

## Write the configuration in Python

A configuration does not need a TOML file. A dictionary with the same structure is validated in the same way, and the same errors are raised. This is convenient to generate configurations, for example to compare several algorithms. `base_artifact_dir` chooses where the runs are saved, and `verbose_override` changes the console output without editing the configuration.

``` python linenums="1" title="config_in_python.py"
{%
    include-markdown "../examples/config_in_python.py"
%}
```

## Add your own model, preprocessing operation or metric

The registries make a component available by name in the configuration. They are described with the models, the preprocessing operations and the metrics of the API reference. This example registers one of each and uses them in a configuration.

``` python linenums="1" title="custom_components.py"
{%
    include-markdown "../examples/custom_components.py"
%}
```

* The components have to be registered before the pipeline is created. The model and the metric names are checked when the configuration is validated, and the preprocessing operation is looked up when the data is prepared.
* A preprocessing operation has to keep the names of its columns, which `OneToOneFeatureMixin` provides, as explained with the [preprocessing operations](api/data.md#adding-a-preprocessing-operation).

## Predict a distribution

For a regression, `OrdBoostRegressor` predicts a distribution of the outcome for each patient instead of a single value. The distribution is used by the `crps` metric and by the figures that check whether the prediction intervals are reliable. The prediction methods are described on the [regression](guide/config/regression.md#using-the-pipeline) page.

``` python linenums="1" title="regression_workflow.py"
{%
    include-markdown "../examples/regression_workflow.py"
%}
```
