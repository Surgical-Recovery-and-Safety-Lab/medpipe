# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to Semantic Versioning ([SemVer](https://semver.org/spec/v2.0.0.html)).

## [Unreleased]
### Added
* **BREAKING** MedpipeOrchestrator class that handles the loading configuration, data, and creates the ArtifactManager
* **BREAKING** MedpipeRunner class that handles the creation and fitting of the models
* **BREAKING** MedpipeEvaluator class that handles the evaluation of the fitted models
* BaseRegistry to create model and preprocessing registries
* PreprocessingRegistry, a dynamic registry to add preprocessing operations
* ModelRegistry, a dynamic registry to add models
* Reproducibility module that contains functions to store configuration and environment information
* Test suites for the reproducibility module functions and classes
* Test suites for the logger module functions
* Test suites for the registries
* Test suites for the MedpipeOrchestrator, MedpipeRunner, and MedpipeEvaluator classes
* New configuration file called default_config.toml

### Changed
* **BREAKING** Refactored the logger module
* **BREAKING** Refactored configuration structure to be in a single file
* **BREAKING** Split the MedpipePipeline into four components Medpipe, MedpipeOrchestrator, MedpipeRunner, and MedpipeEvaluator
* Updated the pyproject.toml according to modern PEP 517/518 and PEP 621 Python packaging standards
* Updated __init__.py for all modules
* Updated tests for the configuration schemas
* Updated read_toml_configuration function
* The compute_metrics function raises a ValueError if only one class is present for AUROC and AP calculations
* Renamed config-examples to examples

### Fixed
* Typo in plot_ROC_curve docstring

### Removed
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
