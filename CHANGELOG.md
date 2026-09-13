# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to Semantic Versioning ([SemVer](https://semver.org/spec/v2.0.0.html)).

## [Unreleased]

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

### Fixed
* Docstring in the `config.py` script.
* Bug in the default override values for the displayer.
* Bug in the `compute_metrics` function. 

### Removed
* Custom typings.

## [0.4.0.dev1] 2026-09-12

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

## [0.4.0.dev0] 2026-09-09

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

[Unreleased]: https://github.com/Surgical-Recovery-and-Safety-Lab/medpipe/compare/v0.4.0.dev1...HEAD
[v0.4.0.dev1]: https://github.com/Surgical-Recovery-and-Safety-Lab/medpipe/compare/v0.4.0.dev0...v0.4.0.dev1
[v0.4.0.dev0]: https://github.com/Surgical-Recovery-and-Safety-Lab/medpipe/compare/v0.3.1...v0.4.0.dev0
[0.3.1]: https://github.com/Surgical-Recovery-and-Safety-Lab/medpipe/compare/v.0.3.0...v0.3.1
[0.3.0]: https://github.com/Surgical-Recovery-and-Safety-Lab/medpipe/compare/v.0.2.1...v0.3.0
