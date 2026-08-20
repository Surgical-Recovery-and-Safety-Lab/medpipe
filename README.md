# Medpipe

[![PyPI Version](https://img.shields.io/pypi/v/medpipe.svg)](https://pypi.org/project/medpipe/)
[![PyPI Python Versions](https://img.shields.io/pypi/pyversions/medpipe.svg)](https://pypi.org/project/medpipe/)
[![License](https://img.shields.io/github/license/Surgical-Recovery-and-Safety-Lab/medpipe)](https://github.com/Surgical-Recovery-and-Safety-Lab/medpipe/blob/main/LICENSE)
[![tests](https://github.com/Surgical-Recovery-and-Safety-Lab/medpipe/actions/workflows/run_test.yml/badge.svg)](https://github.com/Surgical-Recovery-and-Safety-Lab/medpipe/actions/workflows/run_test.yml)
[![Docs](https://img.shields.io/badge/docs-mkdocs-526CFE?style=flat&logo=materialforgithub)](https://Surgical-Recovery-and-Safety-Lab.github.io/medpipe/)

## Table of content
1. [Overview](#overview)
2. [Installation](#installation)
3. [Usage](#usage)
	1. [Preprocessing operations](#preprocess)
	2. [Models](#models)
	3. [Recalibration](#recalibration)
	4. [Metrics](#metrics)
	5. [Plots](#plots)
5. [Examples](#examples)

## Overview
The **medpipe** package is a layer to help create AI models (classifiers and regressors) for clinical applications from tabular data. It covers data loading and preprocessing, model creation and training, recalibration, and visualisation. 
___
## Installation

To install **medpipe** use pip:
```
$ pip install medpipe
```

or clone the GitHub repository and install the package with pip: 
```
$ git clone git@github.com:Surgical-Recovery-and-Safety-Lab/medpipe.git
$ cd medpipe
$ pip install .
```
**NOTE**: It is recommended to use a virtual environment (venv) to install this package. 

Ensure that the installation was successful and that all tests pass by running the following command in the medpipe directory:
```
$ pytest 
```
___

## Usage
This package was tested on a Linux distribution (Ubuntu 24.04) with Python v3.12.3. The [sckit-learn](https://scikit-learn.org/stable/index.html) was used as the base of most of the code. 

A MedpipePipeline contains the preprocessing operations, a model for each prediction outcome, and a recalibration model (if specified) for each outcome. Thus, with only a few lines of code, several models can be created and fitted using the same data. 

### Preprocessing operations
Preprocessing operations are chained and applied sequentially. Available operations are listed in the [sklearn.preprocessing](https://scikit-learn.org/stable/api/sklearn.preprocessing.html) and [sklearn.impute](https://scikit-learn.org/stable/api/sklearn.impute.html) modules.

### Models
Any model from the [sklearn.ensemble](https://scikit-learn.org/stable/api/sklearn.ensemble.html), [sklearn.linear_model](https://scikit-learn.org/stable/api/sklearn.linear_model.html), and [sklearn.isotonic](https://scikit-learn.org/stable/api/sklearn.isotonic.html) modules are available. Additionally, models from the [ngboost](https://stanfordmlgroup.github.io/ngboost/intro) package can be used.

**NOTE:** The package can handle binary classifiers and regressors. However, handling of regressors is not as developed yet.

### Recalibration
Two recalibration models are available: logistic regression (Platt scaling), and isotonic regression. 

### Metrics
The list of available metrics is the following:

| Metric | Description |
| :--- | :--- |
| Accuracy | Proportion of all classifications that were correct. |
| Recall | Proportion of all actual positives that were classified correctly (true positive rate). |
| Precision | Proportion of all the  positive classifications that are actually positive. |
| F1 score | Harmonic mean of precision and recall. |
| AUROC | Area under the ROC curve. |
| AP | Area under the precision-recall curve. |
| Log loss | Logarithmic loss. |
| Brier score | Brier score loss. |
| ICI | Integrated calibration index. |
| RMSE | Root mean squared error. |
| MAE| Mean absolute error. |

**NOTE:** Metrics can be added by editing the METRIC_MAPPING map in the metrics/core.py module.

### Plots
There are four different plots that can be generated (mainly for classifiers):
1. Probability distribution, histrogram of predicted probabilities;
2. ROC curve, receiver operating characteristic curve;
3. Reliability diagram, classifier calibration curve;
4. Fairness heatmap, heatmap of differences for different strata to assess model bias.

___
## Examples

Minimal example based on configuration files:
``` py linenums="1"
import medpipe as mp

config_path = "config-examples/HGBc_config.toml"
pipe = mp.MedpipePipeline(config_path, logger=None)
pipe.run()  # Run pipeline based on configuration
pipe.save()  # Save pipeline
```

Example allowing the user more control over the data and plots:
``` py linenums="1"
import medpipe as mp

config_path = "config-examples/HGBc_config.toml"
pipe = mp.MedpipePipeline(config_path, logger=None)
data = load_data(pipe.medpipe_config.data.path)  # Load the data

# Get the different data sets
X_train, y_train, X_test, y_test, X_recal, y_recal, _ = pipe.get_data_sets(data)
pipe.fit(X_train, y_train, X_recal, y_recal)  # Fit the pipeline

# Make predictions for each outcome and plot some figures
for outcome in pipe.outcomes:
	y_preds = pipe.predict_proba(X_test, outcome)[0]
	mp.plot_ROC_curve(y_test, y_preds, "ROC curve")
	mp.plot_reliability_diagram(y_test, y_preds, "Calibration", strategy="quantile")
	
```
