# Data

---

## Available preprocessing operations

The `name` of a preprocessing operation is the name of a transformer class. `PreprocessorRegistry` resolves the name by first checking the operations registered with it, then searching these modules in order:

| Module | Examples |
| :--- | :--- |
| [`sklearn.preprocessing`](https://scikit-learn.org/stable/api/sklearn.preprocessing.html) | `StandardScaler`, `OrdinalEncoder` |
| [`sklearn.impute`](https://scikit-learn.org/stable/api/sklearn.impute.html) | `SimpleImputer` |

The name must match the class name exactly. Any extra keys of an operation are passed to the class constructor.

```toml
[workflow.preprocessing]
preprocess = true
operations = [
  { name = "StandardScaler", columns = ["AGE"], with_mean = true },
]
```

## Adding a preprocessing operation

Decorate a transformer class with `PreprocessorRegistry.register`. The decorator takes the name to register the class under, which defaults to the class name. The operation can then be used by name in the configuration file.

```python
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.utils.validation import validate_data

from medpipe.data import PreprocessorRegistry


@PreprocessorRegistry.register()
class ClipOutliers(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    def __init__(self, lower=0.0, upper=1.0):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        validate_data(self, X)  # Records the names of the columns
        return self

    def transform(self, X):
        return X.clip(self.lower, self.upper)
```

The operations are chained as pandas data frames, so a transformer has to know the names of its output columns. `OneToOneFeatureMixin` provides them for a transformer that keeps one output column per input column, as long as `fit` records the column names with `validate_data`. Without them, the run stops with "Unable to configure output ... because `set_output` is not available".

```toml
[workflow.preprocessing]
preprocess = true
operations = [
  { name = "ClipOutliers", columns = ["AGE"], lower = 0, upper = 100 },
]
```

Register the operation before the pipeline is run. Its name is looked up when the preprocessing is built, not when the configuration is validated.

A complete script that registers a model, a preprocessing operation and a metric is shown in the [examples](../examples.md#add-your-own-model-preprocessing-operation-or-metric).

---

::: medpipe.data.registry

::: medpipe.data.utils
