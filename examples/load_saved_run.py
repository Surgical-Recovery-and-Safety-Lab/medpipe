import tomllib

from medpipe import MedpipeClassifier, MedpipeClassifierConfig

with open("default_classifier_config.toml", "rb") as file:
    raw_config = tomllib.load(file)
raw_config["meta"]["run_mode"] = "fast"

# Run the pipeline once. The models are saved in the folder of the run.
pipe = MedpipeClassifier(MedpipeClassifierConfig.model_validate(raw_config))
pipe.run()
run_dir = pipe.run_dir

# Later, restore the pipeline from the folder of the run
restored = MedpipeClassifier.load(run_dir)

# Use the fitted models on the test set
_, _, _, _, X_test, y_test, _ = restored.prepare_data()
for outcome in restored.models:
    predictions = restored.predict(X_test, outcome=outcome)
    print(f"{outcome}: {int(predictions.sum())} positive predictions")
