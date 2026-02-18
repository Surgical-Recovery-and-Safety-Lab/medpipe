## Usage
This package was tested on a Linux distribution (Ubuntu 24.04) with Python v3.12.3. The [sckit-learn](https://scikit-learn.org/stable/index.html) was used as the base of most of the code.

### Preprocessing operations
Currently four preprocessing operations are available:
* **standarise**, this operation standardises the input features by removing the mean and scaling to unit variance;
* **ordinal encoding**, this operation converts non-numerical categorical input features into ordinal ones;
* **power transform**, this operation applies a power transform to make the data more Gaussian-like;
* **binning**, this operation converts a continuous input feature into bins and caps the value. 

### Models
There is only one classifier available at the moment: the histogram boosted gradient classifier. 

**NOTE:** Adding a new model only requires editing the **create_model** function in *models/core*. To work, it must have a fit and predict method. 

### Recalibration
Two recalibration models are available: logistic regression, and isotonic regression. 

### Metrics
The available metrics are divided into the score metrics and prediction metrics. The list of available metrics is the following:

| Metric | Type | Description |
| :--- | :--- | :--- |
| Accuracy | Prediction | Proportion of all classifications that were correct. |
| Recall | Prediction | Proportion of all actual positives that were classified correctly (true positive rate). |
| Precision | Prediction | Proportion of all the  positive classifications that are actually positive. |
| F1 score | Prediction | Harmonic mean of precision and recall. |
| AUROC | Score | Area under the ROC curve. |
| AP | Score | Area under the precision-recall curve. |
| Log loss | Score | Logarithmic loss. |

### Plots

Three types of plots are available: bar graphs for the metrics, predicted probability distributions, and calibration curves. 

The following graphs are from one pipeline with two models, one to predict complications and the other to predict 90-day mortality. The predictor and calibrator results are plotted on the same graphs to compare the effect of recalibration. 

Plots of the AUROC and log loss metric values with confidence intervals for each outcome:

| Any complication | 90-day mortality |
| :---: | :---: |
| ![AUROC_any_comp](assets/ai_risk_HGBc_v0.4.1.4-d.1.3.2_ANY_COMP_auroc.png) | ![AUROC_90d_mortality](assets/ai_risk_HGBc_v0.4.1.4-d.1.3.2_MORTALITY_90D_auroc.png) |
| ![log_loss_any_comp](assets/ai_risk_HGBc_v0.4.1.4-d.1.3.2_ANY_COMP_log_loss.png) | ![log_loss_90d_mortality](assets/ai_risk_HGBc_v0.4.1.4-d.1.3.2_MORTALITY_90D_log_loss.png) |


Predicted probability distributions for each outcome:

| Any complication | 90-day mortality |
| :---: | :---: |
| ![proba_dist_any_comp](assets/ai_risk_HGBc_v0.4.1.4-d.1.3.2_ANY_COMP_proba_dist.png) | ![proba_dist_90d_mortality](assets/ai_risk_HGBc_v0.4.1.4-d.1.3.2_MORTALITY_90D_proba_dist.png) |


Calibration curves for each outcome:

| Any complication | 90-day mortality |
| :---: | :---: |
| ![calibration_curve_any_comp](assets/ai_risk_HGBc_v0.4.1.4-d.1.3.2_ANY_COMP_reliability_diagram.png) | ![proba_dist_90d_mortality](assets/ai_risk_HGBc_v0.4.1.4-d.1.3.2_MORTALITY_90D_reliability_diagram.png) |


