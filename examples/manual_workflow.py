import tomllib

from medpipe import MedpipeClassifier, MedpipeClassifierConfig

# Load the configuration of the example and only fit and evaluate the models
with open("default_classifier_config.toml", "rb") as file:
    raw_config = tomllib.load(file)
raw_config["meta"]["run_mode"] = "fast"
config = MedpipeClassifierConfig.model_validate(raw_config)

pipe = MedpipeClassifier(config)

# Load the data and split it as described in the configuration
X_train, y_train, X_recal, y_recal, X_test, y_test, groups_train = pipe.prepare_data()

# Fit one model per outcome
pipe.fit(X_train, y_train, X_recal, y_recal, groups_train)

for outcome in pipe.models:
    # Predict the probability of each class for the test set
    probas = pipe.predict_proba(X_test, outcome=outcome)

    # Evaluate the model, overall and for the subgroups of SEX
    results = pipe.evaluate(
        X_test, y_test[outcome], outcome=outcome, subgroup_specs={"SEX": "SEX"}
    )
    auroc = results["overall"]["roc_auc"]
    print(f"{outcome}: AUROC {auroc['point_estimate']:.2f}")

    # Draw the figures for the outcome
    pipe.plot_all(y_test[outcome].to_numpy(), probas, outcome=outcome, n_bootstraps=50)
