import tomllib

from medpipe import MedpipeRegressor, MedpipeRegressorConfig

with open("default_regressor_config.toml", "rb") as file:
    raw_config = tomllib.load(file)
raw_config["meta"]["run_mode"] = "fast"
pipe = MedpipeRegressor(MedpipeRegressorConfig.model_validate(raw_config))

X_train, y_train, X_recal, y_recal, X_test, y_test, groups_train = pipe.prepare_data()
pipe.fit(X_train, y_train, X_recal, y_recal, groups_train)

for outcome in pipe.models:
    y_true = y_test[outcome].to_numpy()

    # Point predictions, and the distribution of the outcome for each patient
    predictions = pipe.predict(X_test, outcome=outcome)
    distribution = pipe.predict_dist(X_test, outcome=outcome)

    # Evaluate the model with the metrics of the configuration
    results = pipe.evaluate(X_test, y_true, outcome=outcome)
    rmse = results["overall"]["rmse"]["point_estimate"]
    print(f"{outcome}: RMSE {rmse:.2f}")

    # Draw the coverage, sharpness, Winkler score, calibration and PIT figures
    pipe.plot_all(y_true, distribution, outcome=outcome, n_bootstraps=50)
