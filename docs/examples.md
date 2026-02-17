Here is a short example that shows how to load data, train the models, and plot the calibration curves with the quantile strategy:

``` py linenums="1"
from pyrisk import (
	Pipeline
	read_toml_configuration,
	load_data_from_csv,
	get_positive_proba,
	extract_labels,
	plot_reliability_diagrams,
)

# Load configuration and data
config = read_toml_configuration("config_file.toml")
data = load_data_from_csv("data.csv")

# Create pipeline
pipeline = Pipeline(general_config)

# Split data into sets and train model
X_train, X_test = pipeline.get_test_data(data)
pipeline.run(X_train)

# Plot calibration curve
X_test, y_test = extract_labels(X_test, pipeline.label_list)
y_pred_proba = pipeline.predict_proba(X_test)
plot_reliability_diagrams(y_test, get_positive_proba(y_pred_proba, display_kwargs={"n_bins": 10, "strategy": "quantile"})

```


