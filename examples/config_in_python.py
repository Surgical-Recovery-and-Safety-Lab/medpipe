from medpipe import MedpipeClassifier, MedpipeClassifierConfig

# A configuration can be written in Python instead of a TOML file
config = MedpipeClassifierConfig.model_validate(
    {
        "meta": {"project_name": "python_config_demo", "run_mode": "fast"},
        "data": {
            "path": "../tests/test_data.csv",
            "predictors": ["AGE", "SEX", "OP_SEVERITY"],
            "outcomes": ["MORTALITY_30D"],
        },
        "default_model": {
            "algorithm": "LogisticRegression",
            "hyperparameters": {"max_iter": 500},
        },
        "workflow": {
            "preprocessing": {
                "preprocess": True,
                "operations": [
                    {"name": "OrdinalEncoder", "columns": ["SEX"]},
                    {"name": "StandardScaler", "columns": ["AGE", "OP_SEVERITY"]},
                ],
            },
            "validation": {
                "test_split": {"strategy": "random", "test_size": 0.3},
            },
            "evaluation": {
                "metrics": {"metrics": ["roc_auc", "brier_score"], "n_bootstraps": 100},
            },
        },
    }
)

# The artifacts are saved in a custom folder and the console shows all messages
pipe = MedpipeClassifier(
    config, base_artifact_dir="python_config_runs", verbose_override="info"
)
results = pipe.run()
print(results["evaluations"]["MORTALITY_30D"]["overall"]["brier_score"])
