# `[default_model]`

The model fitted for every outcome, and the optional recalibration applied to it afterwards. Each outcome gets its own fitted copy, so outcomes share these settings but not the fitted model.

```toml
[default_model]
algorithm = "HistGradientBoostingClassifier"

[default_model.hyperparameters]
learning_rate = 0.1
loss = "log_loss"
max_features = 0.2

[default_model.recalibration]
recalibrate = true
method = "isotonic"

[default_model.recalibration.hyperparameters]
y_min = 0.0
y_max = 1.0
out_of_bounds = "clip"
```

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `algorithm` | string | required | Name of the model class. See [available models](../../api/models.md#available-models). |
| `hyperparameters` | table | `{}` | Arguments passed to the model constructor. |
| `recalibration` | table | none | Recalibration applied after the model is fitted. Omit it to fit the model without recalibration. |

## Algorithm

`algorithm` is the exact class name of an estimator from scikit-learn (`sklearn.ensemble`, `sklearn.linear_model`, `sklearn.isotonic`), NGBoost or OrdBoost, or of a model you registered with `ModelRegistry`. The [models page](../../api/models.md) lists where names are looked up and how to register your own.

The name is looked up when the configuration is validated, so a misspelt name is reported before any data is loaded. A model registered with `ModelRegistry` is accepted if it is registered before the pipeline is created.

## Hyperparameters

Every key under `[default_model.hyperparameters]` is passed to the constructor of the model. A key the model does not accept is reported when the model is created.

A hyperparameter can be a list of values to search over:

```toml
[default_model.hyperparameters]
learning_rate = [0.1, 0.05]
max_depth = 10
```

When `grid_search = true` in `[workflow.validation.cross_validation]` and the run mode runs cross-validation, every combination of the listed values is tried and the best model is kept. Scalar values are held fixed. Without a search, the first value of a list is used. See the [run modes](meta.md#run-modes).

## Recalibration

Recalibration adjusts the predicted probabilities of a fitted classifier so that they match the observed frequencies better. The model is first fitted on the training set, then a calibrator is fitted on a separate recalibration set.

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `recalibrate` | boolean | required | Whether to recalibrate the fitted model. |
| `method` | string | `"isotonic"` | `"isotonic"` (isotonic regression), `"sigmoid"` (Platt scaling) or `"temperature"` (temperature scaling). |
| `hyperparameters` | table | `{}` | Reserved for hyperparameters of the recalibration method. **Currently not used.** See the note below. |

!!! note
    The `hyperparameters` of the recalibration are accepted by the configuration but are not used yet: they are not passed to the recalibration method, and any values are ignored. A future version may pass them directly to the recalibrator, for example `y_min`, `y_max` and `out_of_bounds` for the isotonic method. Until then, only `recalibrate` and `method` have an effect.

The recalibration set is defined by `[workflow.validation.recalibration_split]`. A `recalibration` table in the default model requires that split, even when `recalibrate = false`. To disable recalibration, either omit the `recalibration` table or set `recalibrate = false` and keep the split.
