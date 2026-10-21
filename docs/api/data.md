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
from sklearn.base import BaseEstimator, TransformerMixin

from medpipe.data import PreprocessorRegistry


@PreprocessorRegistry.register()
class ClipOutliers(BaseEstimator, TransformerMixin):
    def __init__(self, lower=0.0, upper=1.0):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.clip(self.lower, self.upper)
```

```toml
[workflow.preprocessing]
preprocess = true
operations = [
  { name = "ClipOutliers", columns = ["AGE"], lower = 0, upper = 100 },
]
```

Register the operation before creating the pipeline, so that the configuration is validated against it.

---

::: medpipe.data.registry

::: medpipe.data.utils
