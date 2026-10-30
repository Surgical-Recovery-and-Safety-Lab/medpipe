# TRIPOD+AI reporting

[TRIPOD+AI](https://doi.org/10.1136/bmj-2023-078378) is the reporting guideline for studies that develop or evaluate prediction models built with regression or machine learning (Collins et al., *BMJ* 2024;385:e078378). It asks the authors to report how the data was prepared, how the model was built and validated, how it performs, and how the work can be reproduced.

MedPipe records the modelling decisions and the performance of a run, so that a large part of the checklist can be filled in from the [run folder](run_folder.md). It does not write the report, and using it does not make a study compliant with the guideline. Many items concern the study itself, such as the participants, the ethics approval or the funding, and only the authors can report them.

## What is covered

MedPipe is built around three parts of the checklist: the output and the performance of the model, the reproducibility of the work, and the correct handling of the data. The tables say what MedPipe provides for each topic, and where the information is. Match the topics with the items of the published checklist.

*Coverage:* **Provided** means that the information is recorded or produced by every run. **Partly** means that some of it is, and the rest has to be reported by the authors.

### Model output and performance

| Topic | What MedPipe provides | Coverage |
| :--- | :--- | :--- |
| Model output | Probabilities for classifiers, and the predictive distribution for the models that have one. Metrics that need class labels use a fixed threshold of 0.5. A threshold is not chosen or tuned. | Partly |
| Discrimination | The area under the ROC curve, the average precision, and the ROC and precision-recall curves. | Provided |
| Calibration | The Brier score, the log loss, the integrated calibration index, and the reliability diagram. A fitted model can be recalibrated on a separate set with isotonic, sigmoid or temperature scaling. | Provided |
| Clinical usefulness | The decision curve analysis, which gives the net benefit for a range of thresholds. | Provided |
| Regression performance | The RMSE, the MAE and the CRPS, and the coverage, sharpness, Winkler score, marginal calibration and PIT figures of the prediction intervals. | Provided |
| Uncertainty | A bootstrap confidence interval for every metric. The number of resamples and the level are in the configuration. See [confidence intervals](confidence_intervals.md). | Provided |
| Performance in subgroups | Every metric, with its interval, for each subgroup defined in `[workflow.evaluation.fairness]`, and a heatmap of the differences with the whole test set. The groups are chosen by the user, and no fairness criterion is tested. | Partly |

The metrics are computed on the test set, which is kept apart from the model building. The values, with their bounds, are saved in `results/<outcome>_evaluation_results.json`, and the figures in `plots/`.

### Reproducibility

| Topic | What MedPipe provides | Coverage |
| :--- | :--- | :--- |
| Specification of the experiment | The original and the resolved configuration, which hold the data, the preprocessing, the algorithm and the hyperparameters of each outcome, the splits, the metrics and the subgroups. | Provided |
| Hyperparameter tuning | Whether a grid search was used, the values searched, the cross-validation strategy, and the scores of every combination in `CV/<outcome>_cv_results.csv`. The combination is selected on the first metric of the list. | Provided |
| Identity of the data | The path of the dataset, and its SHA-256 hash in `env/env_state.json`, so that the exact file can be identified. | Provided |
| Identity of the configuration | A SHA-256 hash of the resolved configuration, in `env/env_state.json`. | Provided |
| Random seeds | `random_state` of the workflow, in the configuration. The seed of a model is a hyperparameter of that model. | Provided |
| Software | The Python version, the platform and the Git commit of the code, in `env_state.json`. The versions of the packages are not recorded. | Partly |
| The fitted models | The models are saved in `models/` with the configuration that built them. A run can be restored with `load()` and run again from its configuration. There is no export of the coefficients or of a readable equation. | Partly |
| Code | MedPipe is open source under the Apache 2.0 licence. The code of the analysis is the configuration file and the script that calls the pipeline. | Partly |

Every run is saved in its own folder and never overwritten, so the runs of a study can be kept and compared. See [the run folder](run_folder.md).

### Correct handling of the data

The pipeline is built so that the performance reported on the test set is not inflated by information from the model building.

| Topic | What MedPipe does | Coverage |
| :--- | :--- | :--- |
| Separate sets | The test set is set aside first, then the recalibration set, and the rest is the training set. The cross-validation and the grid search only use the training set. The recalibration set is separate from both. | Provided |
| Preprocessing | The operations are part of the model. They are fitted on the training data only, also inside each fold of the cross-validation, and only applied to the other sets. | Provided |
| Validation by group | The test and the recalibration sets can be defined by a column, for example the year or the hospital, which gives a temporal or geographical validation. The two sets cannot share a value, and the cross-validation can keep the groups apart in the folds. This is an internal validation, and an external validation needs a dataset from another source. | Provided |
| Leakage of the outcome | A column cannot be both a predictor and an outcome, which is checked before anything is run. The columns used to split the data or to describe subgroups are not given to the model unless they are listed as predictors. | Provided |
| Class balance | The cross-validation of classifiers keeps the share of positives in every fold. There is no resampling or reweighting, unless the model has a hyperparameter for it, such as `class_weight`. | Partly |
| Checks before the run | The configuration is validated as a whole before any data is read. The columns that are needed are checked before any model is fitted. | Provided |
| Size of the sets | The number of rows of each split, in `execution.log`. There is no calculation of the sample size needed. | Partly |

The other topics of the checklist, such as the participants, the definition of the outcomes and the predictors, the missing data, and the items about ethics and openness, are outside what MedPipe records. The ones to report yourself are listed at the end of this page. MedPipe does not provide an interpretation of the model, such as the importance of the predictors.

## Using the run folder for the report

To describe a study, the files to look at are:

1. `env/resolved_config.json` for the data, preprocessing, models, hyperparameters, splits, metrics and subgroups.
2. `env/env_state.json` for the dataset and configuration hashes, the Python version and the commit.
3. `CV/` for the internal validation and the tuning.
4. `results/` for the performance with the confidence intervals, overall and by subgroup.
5. `plots/` for the discrimination, calibration, decision curve and subgroup figures.
6. `execution.log` for the size of each split.

## What to report yourself

A run does not record these items, so they have to be written in the report or kept next to the run:

* the participants, the setting, the eligibility criteria and the dates of the data;
* the definition and the measurement of the outcomes and of the predictors;
* the amount of missing data and how it was handled;
* the sample size justification;
* the versions of the packages, for example with `pip freeze`;
* the choice of the threshold used to turn probabilities into decisions;
* the ethics approval, the funding, the conflicts of interest and the availability of the data and of the protocol.
