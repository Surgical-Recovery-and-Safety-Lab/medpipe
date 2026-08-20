# Medpipe documentation
**Configuration-driven, TRIPOD+AI compliant ML pipelines for tabular clinical data.**

Medpipe is a Python framework designed to streamline and standardize clinical machine learning workflows on tabular data.
Built around `scikit-learn` and TRIPOD+AI reporting guidelines, it transforms complex clinical modeling into reproducible, TOML-configured pipelines—covering everything from data preprocessing and multi-outcome model training to probability recalibration, subgroup fairness analysis, and clinical decision support visualizations.
By automatically logging execution details and persisting models, environment metadata, and evaluation metrics into a standardized artifact hierarchy, Medpipe ensures full transparency and reproducibility from initial experimentation to publication.

[![PyPI Version](https://img.shields.io/pypi/v/medpipe.svg)](https://pypi.org/project/medpipe/)
[![PyPI Python Versions](https://img.shields.io/pypi/pyversions/medpipe.svg)](https://pypi.org/project/medpipe/)
[![License](https://img.shields.io/github/license/Surgical-Recovery-and-Safety-Lab/medpipe)](https://github.com/Surgical-Recovery-and-Safety-Lab/medpipe/blob/main/LICENSE)
[![tests](https://github.com/Surgical-Recovery-and-Safety-Lab/medpipe/actions/workflows/run_test.yml/badge.svg)](https://github.com/Surgical-Recovery-and-Safety-Lab/medpipe/actions/workflows/run_test.yml)
[![Docs](https://img.shields.io/badge/docs-mkdocs-526CFE?style=flat&logo=materialforgithub)](https://Surgical-Recovery-and-Safety-Lab.github.io/medpipe/)

---

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
{%
    include "../examples/minimal_example.py"
%}
```
___

## Installation

### Standard Installation

Install the published package directly from PyPI:

```bash
pip install medpipe

```

### Installing from source (Developer setup)

To set up a local development environment and contribute to **medpipe**:

Clone the GitHub repository:

```bash
git clone [https://github.com/Surgical-Recovery-and-Safety-Lab/medpipe.git](https://github.com/Surgical-Recovery-and-Safety-Lab/medpipe.git)
cd medpipe

```

Install the package in editable mode with development dependencies:
```bash
pip install -e ".[dev]"

```

---

## Acknowledgements

This package was developed using Gemini 3.6 Thinking. The code was reviewed and edited by humans.
