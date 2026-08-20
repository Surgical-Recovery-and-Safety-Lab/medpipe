# Welcome to Medpipe
**Configuration-driven, TRIPOD+AI compliant ML pipelines for tabular clinical data.**

**medpipe** is a Python framework designed to streamline and standardize clinical machine learning workflows on tabular data.
Built around `scikit-learn` and TRIPOD+AI reporting guidelines, it transforms complex clinical modeling into reproducible, TOML-configured pipelines.
By automatically logging execution details and persisting models, environment metadata, and evaluation metrics into a standardized artifact hierarchy, **medpipe** ensures full transparency and reproducibility from initial experimentation to publication.

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

## Contributing

We welcome contributions from the community! Whether you are fixing bugs, improving documentation, or proposing new features:

1. Feel free to open an issue or start a discussion on our **[GitHub Repository](https://github.com/Surgical-Recovery-and-Safety-Lab/medpipe)**.

2. Submit Pull Requests targeting the `main` branch.
3. Ensure all unit tests pass before submitting (`pytest`).

---

## License

This project is licensed under the Apache-2.0 License. Developed and maintained by the **[Surgical Recovery and Safety Lab](https://github.com/Surgical-Recovery-and-Safety-Lab)**.

---

## Acknowledgements

This package was developed using Gemini 3.6 Thinking. The code was reviewed and edited by humans.
