# API overview

Everything you need for simple use is importable from the top-level `medpipe` package. The other classes documented in this reference are the building blocks behind those entry points.

## Entry points

| Object | Purpose |
| :--- | :--- |
| [`MedpipeClassifier`][medpipe.pipeline.pipeline.MedpipeClassifier] | Run, fit, predict, evaluate and plot binary classification pipelines. |
| [`MedpipeRegressor`][medpipe.pipeline.pipeline.MedpipeRegressor] | Same workflow for regression outcomes with predictive distributions. |
| [`MedpipeClassifierConfig`][medpipe.utils.config.MedpipeClassifierConfig] | Validated configuration for a classifier, built from a TOML file or in code. |
| [`MedpipeRegressorConfig`][medpipe.utils.config.MedpipeRegressorConfig] | Validated configuration for a regressor. |

```python
from medpipe import MedpipeClassifier

pipe = MedpipeClassifier("config.toml")
pipe.run()
```

## Extension points

| Object | Purpose |
| :--- | :--- |
| [`ModelRegistry`][medpipe.models.registry.ModelRegistry] | Look up or register the estimators available to the `algorithm` setting. |
| [`MetricRegistry`][medpipe.metrics.registry.MetricRegistry] | Look up or register the metrics available to the `metrics` setting. |

## Building blocks

Use these when you need more control than the entry points give you, for example to run only the evaluation step or to build your own figures.

| Object | Purpose |
| :--- | :--- |
| [`MedpipeOrchestrator`][medpipe.pipeline.orchestrator.MedpipeOrchestrator] | Reads the configuration, loads the data, builds the splits and manages run artifacts. |
| `MedpipeClassifierRunner`, `MedpipeRegressorRunner` | Cross-validation and model fitting. |
| `MedpipeClassifierEvaluator`, `MedpipeRegressorEvaluator` | Predictions, metrics and bootstrapped confidence intervals. |
| `MedpipeClassifierDisplayer`, `MedpipeRegressorDisplayer` | Figures built from evaluation results. |

## Reference pages

* [Pipeline](pipeline.md): entry points, orchestrator, runners and evaluators.
* [Data](data.md): preprocessing registry, label extraction and dataset splitting.
* [Models](models.md): the model registry.
* [Metrics](metrics.md): the available metrics and how to add your own.
* [Visualisation](visualisation.md): displayers, plotting functions and themes.
* [Utils](utils.md): configuration schemas, I/O, logging, reproducibility and validation.
