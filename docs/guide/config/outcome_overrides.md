# `[outcome_overrides]`

The default model is used for every outcome unless an outcome is overridden. An override is written as a table named after the outcome, for a column listed in `data.outcomes`:

```toml
[outcome_overrides.ANY_COMP]
algorithm = "GradientBoostingClassifier"

[outcome_overrides.ANY_COMP.hyperparameters]
learning_rate = [0.1, 0.05]
max_depth = 10

[outcome_overrides.ANY_COMP.recalibration]
recalibrate = false
```

An override has the same keys as `[default_model]`, and all of them are optional. Only the keys that are written change the default model, so an override can, for example, change a single hyperparameter or only switch the recalibration off:

```toml
[outcome_overrides.ANY_COMP.hyperparameters]
max_depth = 5

[outcome_overrides.MORTALITY_30D.recalibration]
method = "sigmoid"
```

The name of an outcome that is not in `data.outcomes` is rejected. Outcomes without an override use the default model unchanged.

## How an override is combined with the default

The settings of an override are applied on top of the default model, one group of settings at a time.

**Algorithm and hyperparameters**

* If the override uses the same algorithm as the default, or has no `algorithm`, its hyperparameters are merged into the default ones. Keys that are present in both take the override value, and the other default keys are kept.
* If the override uses a different algorithm, the default hyperparameters are discarded and only the hyperparameters written in the override are used. This avoids passing arguments that the new model does not accept.

For example, with the default model above:

| Override | Resulting hyperparameters |
| :--- | :--- |
| `algorithm = "HistGradientBoostingClassifier"`, `learning_rate = 0.5` | `learning_rate = 0.5`, `loss = "log_loss"`, `max_features = 0.2` |
| `algorithm = "RandomForestClassifier"`, `n_estimators = 50` | `n_estimators = 50` |
| `algorithm = "RandomForestClassifier"` | none, so the model defaults are used |

**Recalibration**

* If the override has no `recalibration` table, the default recalibration is kept, even when the algorithm changes.
* If the override has the same `method` as the default, or no `method`, the keys it sets replace the default ones. Its `hyperparameters` are merged into the default ones.
* If the override has a different `method`, the default recalibration hyperparameters are discarded and only those written in the override are used. `recalibrate` is kept from the default unless the override sets it.
* If the default model has no recalibration, an override that adds one must set `recalibrate`.
* Setting `recalibrate = false` in an override turns the recalibration off for that outcome.

In the example above, `ANY_COMP` uses a different algorithm and disables recalibration. `MORTALITY_30D` has no override and uses the default model with isotonic recalibration.

Whenever the default model has a `recalibration` table, or an override turns recalibration on for an outcome, `[workflow.validation.recalibration_split]` must be defined.
