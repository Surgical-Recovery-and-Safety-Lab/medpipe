## Examples

### Minimal example
Minimal example based on configuration files:
``` py linenums="1", title="Minimal example"
import medpipe as mp

config_path = "config-examples/HGBc_config.toml"
pipe = mp.MedpipePipeline(config_path, logger=None)
pipe.run()  # Run pipeline based on configuration
pipe.save()  # Save pipeline
```

### Predict and plot
Example allowing the user more control and shows how to make predictions and plots:
``` py linenums="1", title="Predict and plot"
import medpipe as mp

config_path = "config-examples/HGBc_config.toml"
pipe = mp.MedpipePipeline(config_path, logger=None)
data = load_data(pipe.medpipe_config.data.path)  # Load the data

# Get the different data sets
X_train, y_train, X_test, y_test, X_recal, y_recal, _ = pipe.get_data_sets(data)
pipe.fit(X_train, y_train, X_recal, y_recal)  # Fit the pipeline

# Make predictions for each outcome and plot some figures using predictor
for outcome in pipe.outcomes:
	y_preds = pipe.predict_proba(X_test, outcome, "predictor")[0]
	mp.plot_ROC_curve(y_test, y_preds, "ROC curve")
	mp.plot_reliability_diagram(y_test, y_preds, "Calibration", strategy="quantile")
	
```

### Load a fitted MedpipePipeline
Example showing how to load a saved MedpipePipeline and evaluates the models on the test set:
``` py linenums="1", title="Load a fitted MedpipePipeline"
import medpipe as mp

pipe = mp.load_pipeline("models/mp_pipeline_v0.0.1.joblib")

# Get the different data sets
data = load_data(pipe.medpipe_config.data.path)  # Load the data
_, _, X_test, y_test, _, _, _ = pipe.get_data_sets(data)

pipe.test_models(X_test, y_test)
```

### Compare predictor and recalibrator
Example showing how to evaluate the predictor and the recalibrator of a MedpipePipeline:
``` py linenums="1", title="Compare predictor and recalibrator"
import medpipe as mp

pipe = mp.load_pipeline("models/mp_pipeline_v0.0.1.joblib")

# Get the different data sets
data = load_data(pipe.medpipe_config.data.path)  # Load the data
_, _, X_test, y_test, _, _, _ = pipe.get_data_sets(data)

y_preds = pipe.predict_proba(X_test, pipe.outcomes, "predictor")
y_preds_recalibrated = pipe.predict(X_test, pipe.outcomes, "recalibrator")

for i, outcome in enumerate(pipe.outcomes):
	results = mp.compute_metrics(pipe.metrics, y_test[:, i], y_preds[i])
	results_recalibrated = mp.compute_metrics(pipe.metrics, y_test[:, i], y_preds_recalibrated[i])
	
	print(f"{outcome} results:")
	print(f"Raw: {results}")
	print(f"Recalibrated: {results_recalibrated}")
```
