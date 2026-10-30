from functools import partial

from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score
from sklearn.utils.validation import validate_data

from medpipe import (
    MedpipeClassifier,
    MedpipeClassifierConfig,
    MetricRegistry,
    ModelRegistry,
)
from medpipe.data import PreprocessorRegistry
from medpipe.metrics import MetricSpec


# A model, registered under the name used in the configuration
@ModelRegistry.register("ShallowForest")
class ShallowForest(RandomForestClassifier):
    def __init__(self, n_estimators=100, max_depth=3, random_state=None):
        super().__init__(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
        )


# A preprocessing operation. The name defaults to the name of the class.
# OneToOneFeatureMixin gives it the feature names that the pipeline needs to
# pass the data from one operation to the next.
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


# A metric that weights recall twice as much as precision
MetricRegistry.register_spec(
    MetricSpec(
        name="f2",
        func=partial(fbeta_score, beta=2),
        response_method="predict",
        display_name="F2",
        needs_threshold=True,
    )
)

# The components are registered before the configuration is validated
config = MedpipeClassifierConfig.model_validate(
    {
        "meta": {"project_name": "custom_components_demo", "run_mode": "fast"},
        "data": {
            "path": "../tests/test_data.csv",
            "predictors": ["AGE", "OP_SEVERITY"],
            "outcomes": ["ANY_COMP"],
        },
        "default_model": {
            "algorithm": "ShallowForest",
            "hyperparameters": {"max_depth": 4, "random_state": 42},
        },
        "workflow": {
            "preprocessing": {
                "preprocess": True,
                "operations": [
                    {
                        "name": "ClipOutliers",
                        "columns": ["AGE"],
                        "lower": 18,
                        "upper": 90,
                    },
                ],
            },
            "validation": {"test_split": {"strategy": "random", "test_size": 0.3}},
            "evaluation": {
                "metrics": {"metrics": ["roc_auc", "f2"], "n_bootstraps": 100}
            },
        },
    }
)

pipe = MedpipeClassifier(config)
results = pipe.run()
overall = results["evaluations"]["ANY_COMP"]["overall"]
auroc = overall["roc_auc"]["point_estimate"]
f2 = overall["f2"]["point_estimate"]
print(f"AUROC {auroc:.2f}, F2 {f2:.2f}")
