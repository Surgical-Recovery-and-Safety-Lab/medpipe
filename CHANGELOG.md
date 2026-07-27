# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to Semantic Versioning ([SemVer](https://semver.org/spec/v2.0.0.html)).

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