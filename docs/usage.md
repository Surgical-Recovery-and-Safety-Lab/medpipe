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