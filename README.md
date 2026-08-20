# Medpipe

[![PyPI Version](https://img.shields.io/pypi/v/medpipe.svg)](https://pypi.org/project/medpipe/)
[![PyPI Python Versions](https://img.shields.io/pypi/pyversions/medpipe.svg)](https://pypi.org/project/medpipe/)
[![License](https://img.shields.io/github/license/Surgical-Recovery-and-Safety-Lab/medpipe)](https://github.com/Surgical-Recovery-and-Safety-Lab/medpipe/blob/main/LICENSE)
[![tests](https://github.com/Surgical-Recovery-and-Safety-Lab/medpipe/actions/workflows/run_test.yml/badge.svg)](https://github.com/Surgical-Recovery-and-Safety-Lab/medpipe/actions/workflows/run_test.yml)
[![Docs](https://img.shields.io/badge/docs-mkdocs-526CFE?style=flat&logo=materialforgithub)](https://Surgical-Recovery-and-Safety-Lab.github.io/medpipe/)

## Table of content
1. [Overview](#overview)
2. [Installation](#installation)
3. [Key features](#key-features)
4. [Quickstart example](#quickstart-example)
5. [Acknowledgements](#acknowledgements)

## Overview
The **medpipe** package is a layer to help create AI models (classifiers and regressors) for clinical applications from tabular data. It covers data loading and preprocessing, model creation and training, recalibration, and visualisation. 
___

## Installation

To install **medpipe** use pip:
```
$ pip install medpipe
```
___

## Key features

* **TRIPOD+AI Guideline Compliant**: Designed from the ground up to follow rigorous clinical machine learning reporting standards.
* **Scikit-Learn Compatible**: Integrates seamlessly with standard Scikit-Learn estimators, transformers, and pipelines.
* **Declarative TOML Configuration**: Define entire experiments, from preprocessing to plotting aesthetics, in a single validated TOML configuration file.
* **Multi-Outcome Pipelines**: Model distinct clinical outcomes in a single run with independent hyperparameter and recalibration overrides.
* **Integrated Recalibration & Fairness**: Native probability recalibration and stratified subgroup analysis to evaluate algorithmic fairness across patient cohorts.
* **Clinical Metrics & Visualizations**: Automatic generation of Decision Curve Analysis (DCA), reliability diagrams, ROC/PR curves, and bootstrapped 95% confidence intervals.
* **Standardized Artifact Tracking**: Automatic, structured output management for models, metrics, environment metadata, and execution logs in a unified `artifacts/` hierarchy.
___

## Quickstart example

Minimal example based on the provided configuration file:

``` py linenums="1", title="Minimal example"
from pydantic import ValidationError

from medpipe import Medpipe

try:  # Catch any Pydantic configuration errors
    pipe = Medpipe("default_config.toml")
    pipe.run()
except ValidationError as err:
    print(err)
    exit()
```
___

## Acknowledgements

This package was developed using Gemini 3.6 Thinking. The code was reviewed and edited by humans.
