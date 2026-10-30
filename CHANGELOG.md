# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to Semantic Versioning ([SemVer](https://semver.org/spec/v2.0.0.html)).

## [Unreleased]

### Added
* A workflow that publishes the package to PyPI, with trusted publishing, when
a release of a commit on `main` is published and the tests pass.
* A workflow that deploys the documentation to the `gh-pages` branch when a
release of a commit on `main` is published.
* An `excel` extra (`pip install "medpipe[excel]"`) that installs `openpyxl` to
read Excel files.
* A workflow that builds the documentation with `mkdocs build --strict`, so that
broken links and anchors fail the checks.
* A `default_regressor_config.toml` example configuration and a
`minimal_regressor_example.py` script, with tests that read and run the example.
* The continuous outcomes `LOS_DAYS` and `BLOOD_LOSS_ML` in the test data.
* A regression section in the quickstart.
* A page describing the run folder (the `artifacts/vN` layout and the results),
and a page on the confidence intervals, including the bootstrapping of the
metrics that use a predictive distribution.
* A draft page mapping what medpipe records and produces to the topics of the
TRIPOD+AI reporting guideline.
* An examples page with five verified scripts in the `examples/` folder: going
through the steps of a pipeline, restoring a saved run, writing the
configuration in Python, registering custom components, and predicting a
distribution. A test checks that the scripts are valid, documented, and, for the
light ones, that they run.
* An `ideal` flag for the coverage plot to only plot the ideal coverage line once.
* The `prepare_data()` property to the pipelines.
* A `LeaveOneGroupOut` cross-validation option with the `loso` flag in the 
cross-validation configuration schema.
* The `validation` submodule to the API reference list in MkDocs.
* The `quickstart.md` docs files.
* The `decimals` parameter to the coverage, sharpness, and Winkler score 
calculations to round the intervals.
* Updated the dependencies list to include `mypy` and `pandas-stubs`.
* Bootstrap parallel computation for ROC, PR, coverage, sharpnes, and Winkler 
score. 
* Optimised the DCA computation loop to be more efficient.
* Workflows to check that `ruff` and `mypy` checks pass.
* A `bootstrap` module in `utils` to handle all boostrapping functions.
* A helper function in to check the length of a predictive distribution.
* New tests to cover the additional functions.
* A landing page for the API reference to explain what are the entry points
that can be used.
* A list of available metrics and how to register a new one in the metrics
API reference.
* A list of available models and how to register a new one in the models
API reference.
* A list of available preprocessing operations and how to register a new one
in the data API reference.
* A separate page for the configuration API reference.
* A configuration guide in the `docs/guide/config/` folder, with one page per
configuration block and a page on the regression configuration.

### Changed
* Using `pysplinecalib` instead of `splinecalib`.
* Updates in the marginal calibration and PIT functions to work with the new
version of ordboost.
* Using error bars instead of shaded areas to represent 95% CI in the coverage
and sharpness plots.
* The stratum heatmap labels were updated to simpler versions.
* Small changes to legend locations and labels.
* Made the ideal calibration line in the marginal calibration plot black.
* The `audit` and `eval` flags no longer raise a ValidationError if there is no
fairness data. The heatmaps are not plotted and a message is added to the log.
* The `default_config.toml` was renamed to `default_classifier_config.toml`
because there will be a `default_regressor_config.toml`.
* Updated the ordboost version to 0.6.0 or higher.
* The ROC, precision-recall, coverage, sharpness and Winkler score bootstraps now
use the `n_jobs` of the display configuration (threads for the ROC and
precision-recall curves, processes for the interval metrics).
* An `n_jobs` argument to the bootstrapped plot methods and `plot_all` to override
the `n_jobs` of the display configuration at runtime.
* Subgroup evaluation in the `MedpipeRegressor` no longer re-runs the model 
because `OrdBoost` supports row-indexing as of v0.6.0.
* Updated docstrings that mentionned that `OrdBoost` did not support row-indexing.
* Optimised the CRPS calculation for `OrdBoost` by using the stored CDF directly.
* The `per_sample_func` is no longer needed to compute confidence intervals and 
will no longer raise a warning if not specified.
* The `Quickstart example` now use the correct API.
* The order of the pipeline API reference.
* Remove the underlying functions in the API reference for `config.py`.
* The home page mentions regression, the Python requirement and the new guide
pages, and its acknowledgements match the README.
* The `docs` extra requires `mkdocs<2`.
* The documentation and the README refer to the project as MedPipe, and the
Python package is still called `medpipe`.
* The example of a custom preprocessing operation in the data API reference now
works in a pipeline: the transformer has to provide its feature names.
* Models are checked during the configuration validation and raise an 
error if an unknown model is given.
* Configuration overrides that need recalibration raise an error if the
recalibration split is not specified.
* The `algorithm`, `hyperparameters`, and `recalibration` keys of an outcome
override are now all optional and only the keys that are set are applied to the
default model.
* Cross-validation is only required in the `cv` and `audit` run modes.
* The number of splits (`n_splits`) is now required by the `random` and `group`
cross-validation strategies instead of failing with an `AssertionError` during
the run.
* Setting `n_bootstraps` to 0 in the metrics configuration now skips the
bootstrap and reports point estimates, instead of logging a warning for every
slice.
* Preprocessing operations on columns that are not predictors are now rejected
when the configuration is validated, and with a clear message when the
preprocessor is built.
* The `strategy` of the recalibration split is now required.
* `dist_n_bins` and `dist_yscale` are now fields of the display defaults. A
display configuration without them used to fail with a `KeyError` when the
reliability diagram was drawn.
* The `theme` of the display configuration is now used by the displayers. It
used to be validated but ignored, and unknown theme parameters are now rejected.
* Overrides under the `heatmap` plot type are now applied to the strata
heatmaps.
* `MedpipeRegressor.run()` no longer fails when a model cannot produce the
distributional figures (NGBoost and scikit-learn regressors). The figures of the
outcome are skipped with a warning.
* `MedpipeRegressor.run()` now draws the subgroup strata heatmaps when a
fairness configuration is set.
* The regressor display configuration has its own defaults
(`RegressorDisplayDefaultsConfig`): `coverage_levels` and `pit_n_bins` are now
configurable, the classifier-only keys are rejected, and `dist_n_bins` defaults
to 20. The `residuals` and `predicted_vs_actual` plot types, which had no plots,
are no longer accepted.
* The colour scale of the strata heatmaps of `rmse`, `mae`, and `crps` now
follows the data instead of a fixed scale.
* The metrics are checked against the type of outcome when the configuration is
validated: classification metrics are rejected for regression, `crps` is
rejected for classification and for models that cannot predict a distribution.
* The error for an unknown metric in the configuration listed the available
metrics twice.
* The `cv_summary.json` file saved with a grid search was empty. It now holds the
mean and standard deviation of each metric for the selected hyperparameters.
* The messages logged by `MedpipeRegressor` (start, load, end) are now shown in
the compact verbosity mode, and "Succesfully" was corrected to "Successfully".

### Fixed
* All tests now pass.
* Docstrings in the `metrics/core.py` file that were not in the Numpy style.
* The `shuffle` flag now makes it all the way to `_create_cv_splitter`.
* Formatted with `ruff`.
* Typo in the `MedpipeRegressor` docstring.
* Clone command in the `install.md` file to use ssh rather than https.
* The `BaseRegistry` class docstring.
* Inconsistencies in the documentation and typos in the docstrings.


### Removed
* Old references to modules that no longer exist in the MkDocs API section.
* Typing errors with `mypy`.

## [0.4.0.dev2] - 2026-10-05

### Added
* Fit time and total run time to debug log.
* Saving the raw toml configuration file to the `env/` artifact folder.
* Tests for new features.
* Splitting classes (`Medpipe`, `MedpipeEvaluator`, `MedpipeRunner`) into classifier and regressor. 
* Renamed `Medpipe` to `MedpipeClassifier`.
* Regression configuration schemas.
* Tests for the regression configuration schemas.
* New regression objects `MedpipeRegressionRunner` and `MedpipeRegressionEvaluator`.
* The CRPS as a score for cross-validation and evaluation for the regressor.
* A `MedpipeRegressor` class mirroring `MedpipeClassifier` for regression tasks.
* A `FairnessSplits` mirroring the `DataSplits` class that is used for the 
fairness evaluation.
* Renamed `read_toml_configuration` to `read_classifier_toml_configuration`.
* An `is_classifier` flag to `MedpipeOrchestrator` to load configuration correctly.
* Split the `MedipeDisplayer` into a base and a classifier version.
* The `MedipeRegressorDisplayer` and test cases.
* Parallel loop for stratum bootstrap evaluation.
* Parallel loop for spline bootstrap reliability diagram.
* `cv_splines` with a default value of 3 to ICI and reliability diagram 
`SplineCalib` computations.
* Moved `plot_probability_distribution` to `BaseDisplayer` as
`plot_data_distribution`, so `MedpipeRegressorDisplayer` can also plot the
target-value data distribution; included in both displayers' `plot_all`.

### Fixed
* Docstring in the `config.py` script.
* Bug in the default override values for the displayer.
* Bug in the `compute_metrics` function. 
* Bug in the `load` methods that did not load the fitted models to the evaluator.

### Removed
* Custom typings.

## [0.4.0.dev1] - 2026-09-12

### Added
* `ruff` as a dev dependency, with lint configuration (`select` rules,
line length, target Python version) in `pyproject.toml`.
* Separate files for the `config` tests with increased coverage.
* Separate files for the `logger` tests with increased coverage.
* Separate files for the `reproducibility` tests with increased coverage.
* Separate files for the `io` tests with increased coverage.
* Separate files for the `visualisation/displayer` tests with increased coverage.
* The `openpyxl` package as an optional dependency.
* Ignoring thrid-party warnings about `pyarrow.feather` in tests.
* Renamed `exceptions.py` to `validation.py` for clarity.
* Safety check in the `BaseRegistry` _fallback_module.
* Increased coverage in the `registry` tests.
* Test coverage for the `metrics/core.py` functions.
* Test coverage for the `metrics/registry.py` functions.
* Test coverage for the `data` functions.
* Test coverage for the `models` functions.
* Test coverage for the `visualisation` functions.
* Separate files for the `orchestrator` tests with increased coverage.
* Separate files for the `runner` tests with increased coverage.
* Test coverage for the `evaluator` functions.
* Test coverage for the `pipeline` functions.

### Removed
* Monolithic `test_config.py` file.
* Monolithic `test_logger.py` file.
* Monolithic `test_reproducibility.py` file.
* Monolithic `test_io.py` file.
* Monolithic `test_displayer.py` file.
* The `array_check` and `array_dim_check` functions, replaced with sklearn
functions instead.
* The `BoundedLogitTransformer` for regression.
* Monolithic `test_orchestrator.py` file.
* Monolithic `test_runner.py` file.

### Changed
* Reformatted the package and test suite to satisfy `ruff` lint rules:
import sorting, modernized type hints (PEP 604/585/695), collapsed
nested conditionals, and wrapped long lines.
* Replaced the ambiguous en dash with a hyphen in stratum range labels
(e.g. `"AGE: 18-50"`).

### Fixed
* Bug in `io.py` that was case-sensitive for the file extensions.
* Bug in data split which never had `random_state` reach it.
* Bug with the 'spline' strategy for the calibration curve that did not remove the
markers when plotting.
* Tightened several `pytest.raises(match=...)` patterns that relied on
unescaped regex metacharacters (e.g. `.`) to coincidentally match.
* Added explicit `strict=True` to a `zip()` call in the evaluator's
bootstrap-CI fallback path.

## [0.4.0.dev0] - 2026-09-09

### Added
* **BREAKING** MedpipeOrchestrator class that handles the loading configuration, data, and creates the ArtifactManager
* **BREAKING** MedpipeRunner class that handles the creation and fitting of the models
* **BREAKING** MedpipeEvaluator class that handles the evaluation of the fitted models
* A visualisation module to handle plotting functions
* MedpipeDisplayer class that handles plotting and saving graphs
* MedpipeTheme to manage graph parameters
* BaseRegistry to create model and preprocessing registries
* PreprocessingRegistry, a dynamic registry to add preprocessing operations
* ModelRegistry, a dynamic registry to add models
* MetricRegistry, a dynamic registry to add custom metrics
* Reproducibility module that contains functions to store configuration and environment information
* New configuration file called default_config.toml
* Test suites for the reproducibility module functions and classes
* Test suites for the logger module functions
* Test suites for the registries
* Test suites for the Medpipe classes
* Stress tests for the Medpipe class
* DataLoaderRegistry to accept unsupported types of data
* New data types supported (.tsv, .txt, .pq, .feather, .xls, .xlsx, .json, .jsonl, .pickle, .pkl)
* Additional plugins for MkDocs to include code snippets from examples and search
* A docs optional dependencies in the pyproject.toml file
* Visualisation API reference in MkDocs


### Changed
* **BREAKING** Refactored the logger module
* **BREAKING** Refactored configuration structure to be in a single file
* **BREAKING** Split the MedpipePipeline into five components Medpipe, MedpipeOrchestrator, MedpipeRunner, MedpipeEvaluator, and MedpipeDisplayer
* Updated the pyproject.toml according to modern PEP 517/518 and PEP 621 Python packaging standards
* Updated __init__.py for all modules
* Updated tests for the configuration schemas
* Updated read_toml_configuration function
* The compute_metrics function raises a ValueError if only one class is present for AUROC and AP calculations
* Renamed config-examples to examples
* Updated LICENSE with copyright year and owner name
* Renamed test folder to tests
* Updated README.md
* pyproject.toml project description
* MkDocs index page
* Location for the API references from docs/ to docs/api
* Site name and layout
* Moved install.md to docs/start folder

### Fixed
* Typo in the CATEGORY_LEVEL_1 column of test_data.csv

### Removed
* **BREAKING** plot functions in the metrics/ modules
* Functions to read subconfiguration files in utils/io.py
* The compute_stata_metrics and print_metrics functions in metrics/core.py
* The convert_dtypes function in data/utils.py
* requirements.txt moved to pyproject.toml
* Old configuration files

## [0.3.1] - 2026-07-27

### Added
* Changelog added. 
* Test for the test_models function to make sure it runs with a recalibrator.
### Fixed
* Fixed a bug when calling test_models with a recalibrator.
* Typo in the MedpipePipeline class methods docstring.

## [0.3.0] - 2026-07-23
### Added
* Added spline calibration to reliability diagram.
* Added fairness heatmap plot.

### Changed
* Removed custom classes for models and replaced by building models directly from sklearn or ngboost.
* Simplified preprocessing by using Pipelines and ColumnTransformers.
* Refactored configuration files to be much simpler and readable.
* Using pydantic to validate configuration files.
* Documentations updated.

### Removed
* Class imbalance mitigation methods (data sampling, cost-sensitive learning).

[Unreleased]: https://github.com/Surgical-Recovery-and-Safety-Lab/medpipe/compare/v0.4.0.dev2...HEAD
[v0.4.0.dev2]: https://github.com/Surgical-Recovery-and-Safety-Lab/medpipe/compare/v0.4.0.dev1...v0.4.0.dev2
[v0.4.0.dev1]: https://github.com/Surgical-Recovery-and-Safety-Lab/medpipe/compare/v0.4.0.dev0...v0.4.0.dev1
[v0.4.0.dev0]: https://github.com/Surgical-Recovery-and-Safety-Lab/medpipe/compare/v0.3.1...v0.4.0.dev0
[0.3.1]: https://github.com/Surgical-Recovery-and-Safety-Lab/medpipe/compare/v.0.3.0...v0.3.1
[0.3.0]: https://github.com/Surgical-Recovery-and-Safety-Lab/medpipe/compare/v.0.2.1...v0.3.0
