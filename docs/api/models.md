# Models

---

## Available models

The `algorithm` setting of a model is the name of an estimator class. `ModelRegistry` resolves the name by first checking the models registered with it, then searching these modules in order:

| Module | Examples |
| :--- | :--- |
| [`sklearn.ensemble`](https://scikit-learn.org/stable/api/sklearn.ensemble.html) | `HistGradientBoostingClassifier`, `RandomForestClassifier` |
| [`sklearn.linear_model`](https://scikit-learn.org/stable/api/sklearn.linear_model.html) | `LogisticRegression` |
| [`sklearn.isotonic`](https://scikit-learn.org/stable/api/sklearn.isotonic.html) | `IsotonicRegression` |
| [`ngboost`](https://stanfordmlgroup.github.io/ngboost/intro) | `NGBRegressor` |
| [`ordboost`](https://surgical-recovery-and-safety-lab.github.io/ordboost/) | `OrdBoostClassifier`, `OrdBoostRegressor` |

The name must match the class name exactly. The `hyperparameters` setting is passed to the class constructor.

## Adding a model

Decorate an estimator class with `ModelRegistry.register`. The decorator takes the name to register the class under, which defaults to the class name. The model can then be used by name in the configuration file.

```python
from sklearn.ensemble import RandomForestClassifier

from medpipe import ModelRegistry


@ModelRegistry.register("ShallowForest")
class ShallowForest(RandomForestClassifier):
    def __init__(self, n_estimators=100, max_depth=3, random_state=None):
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )
```

```toml
[default_model]
algorithm = "ShallowForest"
```

Register the model before creating the pipeline, so that the configuration is validated against it.

A complete script that registers a model, a preprocessing operation and a metric is shown in the [examples](../examples.md#add-your-own-model-preprocessing-operation-or-metric).

---

::: medpipe.models.registry
