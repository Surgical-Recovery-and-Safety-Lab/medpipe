# Configuration setup
This document provides details about the configuration structures for the **medpipe** package. See the [_config-examples_](https://github.com/Surgical-Recovery-and-Safety-Lab/medpipe/tree/main/config-examples) folder for examples. 

## Configuration file structure
Configuration files are required to provide variables and options. They are written using TOML. There are three main configuration folders: data, workflow, and hyperparameters. Examples for the configuration files can be found in the *config-examples* folder.

The configuration files are nested in folders and subfolders with the following structure:
```text
├── data/
│   └── data_v0.toml
├── hyperparameters/
│   ├── hyperparameters_v0.toml
│   └── hyperparameters_v1.toml
├── workflow/
│   ├── workflow_v0.toml
│   └── workflow_v1.toml
└── top_level_config.toml
```

## Naming conventions
The top-level configuration file does not follow any particular naming rules and can have any name. The path to this file is the one specified to the MedpipePipeline instance.

The folders data/, hyperparameters/, and workflow/ must be specified. They can be located in a different folder but must follow the same path.

The configuration files located within each subfolder (data/, hyperparameters/, or workflow/) all follow the same naming convention: **folder_name_vX.toml**, where folder_name is one of the three folders, and X is an integer. 

## Top-level configuration
This is the configuration file that is read when creating a MedpipePipeline object. There are four tables in a top-level configuration file. 

### Meta table

This table and its variables are all mandatory. 

| **Key** | **Type**  | **Description** |
| :--- | :---  | :--- |
| `version` | string  | Version number. See below for more information. |
| `project_name` | string | Name of the project which will be used for saving models and figures. |
| `run_mode` | {"audit", "cv", "eval", "fast"} | Mode to use when calling the run() function. See below for more details. |

**Version number info:** 

The version number is parsed and used to read the data/, workflow/, and hyperparameters/ configuration. Thus, version "v0.0.1" reads the data_v0.toml, workflow_v0.toml, and hyperparameters_v1.toml configurations.

**Run mode info:** 

The run mode has 4 different settings:
* "audit" runs a cross-validation scheme, tests the final models, and plots relevant figures for the classifiers; 
* "cv" only runs the cross-validation scheme and tests the final models; 
* "eval" trains and tests the models, and plots relevant figures for the classifiers; and,
* "fast" only trains and tests the final models.

### Paths table

This table and its variables are all mandatory. 

| **Key** | **Type**  | **Description** |
| :--- | :---  | :--- |
| `config_dir` | string  | Absolute or relative path to the directory that contains the configuration subfolders. |
| `model_dir` | string | Absolute or relative path to the directory where pipelines are saved or loaded from. |
| `figure_dir` | string | Absolute or relative path to the directory where figures are saved. |

**NOTE:** When using the "eval" or "audit" run mode, the figures are saved as figure_dir/project_name/version/saved_figure.png.

### Model table

This table and its variables are all mandatory. 

| **Key** | **Type**  | **Description** |
| :--- | :---  | :--- |
| `algorithm` | string  | Algorithm to use for the pipeline predictors. |

**NOTE:** The algorithm must be an exact match with the desired model to load in sklearn or ngboost. 

### Recalibration table

This table and its variables are optional. If not specified, no recalibration will be performed.

| **Key** | **Type**  | **Description** |
| :--- | :---  | :--- |
| `method` | string  | Method to use for the pipeline recalibrators. |

## Data configuration
The data_vX.toml file must contain the following parameters:

| **Key** | **Type**  | **Description** |
| :--- | :---  | :--- |
| `path` | string  | Aboslute or relative path to the data file. |
| `predictors` | list[string] | List of predictors to extract from the data. |
| `outcomes` | list[string] | List of outcomes to extract from the data. |

Errors will be raised if any of these variables are missing from the configuration file. Additionally, an error is raised if an outcome is present in the predictor list. 

**NOTE:** The data can be a .csv or a .parquet file.

## Workflow configuration
The workflow configuration file contains the parameter tables that configure data preprocessing, model validation, and evaluation. It is divided into three tables. 

### Preprocessing table

The preprocessing table contains the preprocessing operations. This table and its variables are optional. 

| **Key** | **Type**  | **Description** |
| :--- | :---  | :--- |
| `preprocess` | bool  | Flag for performing preprocessing operations. |
| `operations` | list[Operations] | List of preprocessing operations. See below for more details |

**operations subtable**

An Operation is dictionary with the following parameters:

| **Key** | **Type**  | **Description** |
| :--- | :---  | :--- |
| `name` | string  | Name of the preprocessing operation. |
| `columns` | list[string] | List of column on which to apply preprocessing operation. |
| `kwargs` | --- | Additional keyword arguments for the preprocessing operation. |

**NOTE:** The name of the Operation must match one of the available options. The keyword arguments can be provided as a dictionary key and value. 

### Validation table

The validation table contains subtables that describe the validation strategies. This table and the test_split subtable are mandatory. 

**test_split subtable**

This table is mandatory. 

| **Key** | **Type**  | **Description** |
| :--- | :---  | :--- |
| `strategy` | {"random", "group"}  | Strategy to split the data into train and test sets. |
| `test_size` | float | Only if strategy is random, size of the test set, between 0 and 1. |
| `group_column` | string | Only if strategy is group, column use to split the data. |
| `values` | list[string \| int] | Only if strategy is group, values of the group to use as the test set. |

**NOTE:** If the strategy is group, the data is split based on the values for the group_column. For example, if the group_colum is OP_YEAR and the values is [2024], the test set will be made up of all samples that have OP_YEAR = 2024. 

**recalibration_split subtable**

This table is only mandatory if a recalibration method is specified in the top-level config.

| **Key** | **Type**  | **Description** |
| :--- | :---  | :--- |
| `strategy` | {"random", "group"}  | Strategy to split the data into train and recalibration sets. |
| `recalibration_size` | float | Only if strategy is random, size of the recalibration set, between 0 and 1. |
| `group_column` | string | Only if strategy is group, column use to split the data. |
| `values` | list[string \| int] | Only if strategy is group, values of the group to use as the recalibration set. |

**NOTE:** If the strategy is group, the data is split based on the values for the group_column. For example, if the group_colum is OP_YEAR and the values is [2023], the recalibration set will be made up of all samples that have OP_YEAR = 2023. 

**IMPORTANT:** The recalibration split is performed after the test split, using the new train set (*i.e.* without the test samples).

**cross_validation subtable**

This table is only mandatory with the 'cv' or 'audit' run modes.

| **Key** | **Type**  | **Description** |
| :--- | :---  | :--- |
| `strategy` | {"random", "group"}  | Strategy to use for cross-validation. |
| `group_column` | string | Only if strategy is group, column use to cross-validate. |
| `n_splits` | int | Number of cross-validation splits, must be greater than 2. |
| `shuffle` | bool | Flag to shuffle data before splitting. |
| `random_state` | int | Affects the ordering of the shuffling if shuffle is True. |

### Evaluation table

The evaluation table contains the parameters for the pipeline evaluation. This table and the metrics subtable are mandatory. 

**metrics subtable**

This table is mandatory.

| **Key** | **Type**  | **Description** |
| :--- | :---  | :--- |
| `metrics` | list[string]  | List of metrics to evaluate the models on. |

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

**calibration subtable**

This table is only mandatory with the 'audit' or 'eval' run modes.

| **Key** | **Type**  | **Description** |
| :--- | :---  | :--- |
| `strategy` | {"uniform", "quantile", "spline"}  | Strategy to compute the calibration curve. |
| `n_boostraps` | int | Number of bootstrap iterations to compute confidence intervals. |

**NOTE:** The 'spline' calibration strategy fits a spline to the data, which is computationally intensive. The number of boostraps should be carefully chosen to avoid long computations. 

**fairness subtable**

This table is only mandatory with the 'audit' or 'eval' run modes.

| **Key** | **Type**  | **Description** |
| :--- | :---  | :--- |
| `strata` | list[string] | Strata to evaluate fairness on. |
| `groups` | dict[string, list[list[int \| float \| string]]] | Groups to evaluate fairness on. See below for more details. |

**IMPORTANT:** The strata are columns over which to evaluate the model to assess bias and fairness. For each column, all unique values are tested. For example, if sex is a strata, the models will be evaluated for all female and male samples. The groups parameters allows to set the groups on which to evaluate the model, mainly designed for a strata like age. In this case, the groups will be a list of lists [[18, 50], [51, 120]], which will evaluate the models for samples with ages between 18 and 50, and ages 51 and 120. 

## Hyperparameters configuration

### Hyperparameters table

The hyperparameters table contains the parameters used to tune the predictors and the recalibrators (if specified). The subtables allow for passing any keyword argument which will be passed on to the predictors or recalibrators when they are created. The hyperparameter table and the predictor subtable are mandatory. 

**predictor subtable**

This table is mandatory. 

| **Key** | **Type**  | **Description** |
| :--- | :---  | :--- |
| `learning_rate` | float  | Predictor learning rate, must be greater than 0. |
| `kwargs` | ---  | Additional arguments passed to the predictors. |

**recalibrator subtable**

This table is optional. 

| **Key** | **Type**  | **Description** |
| :--- | :---  | :--- |
| `kwargs` | ---  | Arguments passed to the recalibrators. |