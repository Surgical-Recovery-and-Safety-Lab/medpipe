# Metrics

---

## Available metrics

Metrics are selected by name in the `metrics` setting of the configuration file. The tables list the names registered by default.

### Classification

| Name | Description | Uses |
| :--- | :--- | :--- |
| `accuracy` | Proportion of all classifications that were correct. | Class labels |
| `precision` | Proportion of positive classifications that are actually positive. | Class labels |
| `recall` | Proportion of actual positives that were classified correctly (true positive rate). | Class labels |
| `f1` | Harmonic mean of precision and recall. | Class labels |
| `roc_auc`, `auroc` | Area under the ROC curve. Both names are the same metric. | Decision function or probabilities |
| `ap` | Average precision, the area under the precision-recall curve. | Decision function or probabilities |
| `log_loss` | Logarithmic loss. | Probabilities |
| `brier_score` | Brier score loss. | Probabilities |
| `ici` | Integrated calibration index, the mean absolute difference between predicted probabilities and a spline-smoothed calibration curve. | Probabilities |

Metrics that use class labels convert predicted probabilities to labels before they are computed.

### Regression

| Name | Description | Uses |
| :--- | :--- | :--- |
| `rmse` | Root mean squared error. | Point predictions |
| `mae` | Mean absolute error. | Point predictions |
| `crps` | Continuous ranked probability score, averaged over samples (lower is better). | Predictive distribution |

## Adding a metric

Describe the metric with a [`MetricSpec`][medpipe.metrics.registry.MetricSpec] and register it with `MetricRegistry`. It can then be used by name in the configuration file.

```python
from functools import partial

from sklearn.metrics import fbeta_score

from medpipe import MetricRegistry
from medpipe.metrics import MetricSpec

MetricRegistry.register_spec(
    MetricSpec(
        name="f2",
        func=partial(fbeta_score, beta=2),
        response_method="predict",
        display_name="F2",
        needs_threshold=True,
    )
)
```

Register the metric before creating the pipeline, so that the configuration is validated against it.

---

::: medpipe.metrics.core

::: medpipe.metrics.registry
