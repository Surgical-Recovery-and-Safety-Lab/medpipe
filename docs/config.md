# Configuration setup
This document provides details about the configuration structures for the **medpipe** package. See the [_config-examples_](https://github.com/Surgical-Recovery-and-Safety-Lab/medpipe/tree/main/config-examples) folder for examples. 

## Configuration file structure
Configuration files are required to provide variables and options. They are written using TOML. The main configuration files are for the data (loading, preprocessing, etc.), the models (name, type, hyperparameters, etc.), and the logger (message, log location, etc.). Examples for the configuration files can be found in the *config-examples* folder.

The configuration files are nested in folders and subfolders with the following structure:
```text
├── data/
│   ├── features/
│   │   └── name_features-v1.toml
│   ├── general/
│   │   └── name_general-v0.toml
│   ├── preprocessing/
│   │   └── name_preprocessing-v0.toml
│   └── splitting/
│       └── name_splitting-v1.toml
├── model/
│   ├── calibrator/
│   │   └── name_calibrator-v1.toml
│   ├── hyperparameters/
│   │   └── name_hyperparameters-v1.toml
│   ├── imbalance/
│   │   └── name_imbalance-v0.toml
│   └── labels/
│       └── name_labels-va.toml
├── data_config.toml
├── log_config.toml
└── model_config.toml
```

## Naming conventions
The top-level configuration files (data_config.toml, log_config.toml, and model_config.toml) do not follow any particular naming rules and can have any name. 

The folders (data/ and model/) do not follow any particular naming rules. However, the subfolders must match with the names in tree, which are provided in the top level configuration files.

The subfolder .toml files must be structed as **name_subfoler-vX.toml**, where X is an integer or a letter. The *name* variable is provided in the top-level configuration files, the *subfolder* variable must match the name of the subfolder, and the *X* must match the version number provided in the top-level configuration. The **labels** version number is a letter to provide a separation between model parameters and data parameters. 

## Data configuration
The data_config.toml file contains main parameters and a data table with the following parameters:

| **Key** | **Type**  | **Description** |
| :--- | :---  | :--- |
| `version` | string  | Version number for the data configuration. |
| `base_dir` | string | Base directory location. |

**NOTE:** The version number is formatted as vX.Y where X and Y are integers. The version number contains as many numbers as there are subfolders. The numbers are parsed to fetch the correct file version in the subfolders. 

**Data table**

These parameters are used to extract, load, and save the data.  The table is named [data_parameters].

| **Key** | **Type**  | **Default** | **Description** |
| :--- | :---  | :--- | :--- |
| `dir` | string  | N/A | Location of data configuration files from base_dir. |
| `subfolders` | list[string] | See note | List of subfolders to search for data configuration files. |
| `name` | string | N/A | Prefix name of the data configuration files. |
| `extension` | string | '.toml' | Extension of the data configuration files. |

**NOTE:** The subfolders must be ['general', 'features'] for the data parameters.

### Data subfolders
#### Features
The features configuration files contain the list of features to extract from the database. The feature list is contained in a TOML table named [features].

| **Key** | **Type**  | **Description** |
| :--- | :---  | :--- |
| `feature_list` | list[string]  |  List of the features, including prediction labels, to extract from the data. |

#### General
The general configuration files contain main parameters and an I/O table, and a DB table with the following parameters:

| **Key** | **Type**  | **Description** |
| :--- | :---  | :--- |
| `base_dir` | string | Base directory location. |

**I/O table**

These paramaters are used for saving and loading data.  The table is named [io_parameters].

| **Key** | **Type**  | **Default** | **Description** |
| :--- | :---  | :--- | :--- |
| `dir` | string  | N/A | Location of the data from base_dir. |
| `name` | string | N/A | Prefix name of the data file. |
| `extension` | string | '.csv' | Extension of the data file. |

**DB table**

These paramaters are used for opening and extracting data from a database.  The table is named [db_parameters].

| **Key** | **Type**  | **Default** | **Description** |
| :--- | :---  | :--- | :--- |
| `dir` | string  | N/A | Location of the database from base_dir. |
| `name` | string | N/A | Prefix name of the database file. |
| `table_name` | string | 'main' | Name of the table to query. |
| `extension` | string | '.db' | Extension of the database file. |

#### Preprocessing
The preprocessing configuration files contain the preprocessing operations and the feature list on which to apply them. The variables are stored in tables with the following parameters:

**Preprocessing table**

| **Key** | **Type**  | **Description** |
| :--- | :---  | :--- |
| `preprocess` | bool | Flag to apply operations. |

**NOTE:** The v0 preprocessing file only contains the preprocessing table with the preprocess flag set to false.

The following tables can be removed if the preprocessing operations does not need to be performed.

**Ordinal encoder table**

These paramaters are used for the ordinal encoder preprocessing operation. The table is named [preprocessing.ordinal_encoder].

| **Key** | **Type**  | **Description** |
| :--- | :---  | :--- |
| `feature_list` | list[string]  |  List of the features encode. |

**Standarise table**

These paramaters are used for the standardise preprocessing operation. The table is named [preprocessing.standardise].

| **Key** | **Type**  | **Description** |
| :--- | :---  | :--- |
| `feature_list` | list[string]  |  List of the features to standardise. |

**Bin table**

These paramaters are used for the bin preprocessing operation. The table is named [preprocessing.bin].

| **Key** | **Type**  | **Description** |
| :--- | :---  | :--- |
| `feature_list` | list[string]  |  List of the features to bin. |

#### Splitting
The splitting configuration files contain the variables used for splitting the data into test and train splits and for cross validation. The variables are contained in two TOML tables named [split_variables] and [cv_variables].

**Split table**
These parameters are used for splitting test and train sets.

| **Key** | **Type**  | **Description** |
| :--- | :---  | :--- |
| `group_name` | string  | Feature to use to split the data. |
| `test_size` | float  | Size of the test set if group_name is "". |
| `drop` | bool  | Flag to drop the feature from the training process. |

**CV table**
These parameters are used for cross-validation scheme.

| **Key** | **Type**  | **Description** |
| :--- | :---  | :--- |
| `group_k_fold` | bool  | Flag to use group K-fold cross validation. |
| `group_name` | string  | Feature to use to create the groups. If "" then n_splits is used. |
| `drop` | bool  | Flag to drop the feature from the training process. |
| `n_splits` | int  | Number of splits to create for the cross validation, if group_k_fold is False. |
| `shuffle` | bool  | Flag to shuffle the groups. |
| `random_state` | int  | Random seed used for repeatability. |

## Model configuration
The model_config.toml file contains main parameters, a model table, a data table, an I/O table, and a fig table with the following parameters:

| **Key** | **Type**  | **Default** | **Description** |
| :--- | :---  | :--- | :--- |
| `version` | string  | N/A | Version number for the model configuration. |
| `predictor_type` | string | 'hgb-c' | Predictor type to use. |
| `base_dir` | string | N/A| Base directory location. |

**NOTE:** The version number is formatted as vA.B.C.D-W.X.Y.Z where W is a letter and A, B, C, D, X, Y, and Z are integers. The A.B.C.D portion represents the data version number and the W.X.Y.Z the model version number. The version number contains as many symbols as there are subfolders. The symbols are parsed to fetch the correct file version in the subfolders. 

**Model table**

These parameters are used to create the models. The table is named [model_parameters].

| **Key** | **Type**  | **Default** | **Description** |
| :--- | :---  | :--- | :--- |
| `dir` | string  | N/A | Location of model configuration files from base_dir. |
| `subfolders` | list[string] | See note | List of subfolders to search for model configuration files. |
| `name` | string | N/A | Prefix name of the model configuration files. |
| `extension` | string | '.toml' | Extension of the model configuration files. |

**NOTE:** The subfolders must be ['labels', 'imbalance', 'hyperparameters', 'calibrator'] for the model parameters. 

**Data table**

These parameters are used to load and preprocess the data. The table is named [data_parameters].

| **Key** | **Type**  | **Default** | **Description** |
| :--- | :---  | :--- | :--- |
| `dir` | string  | N/A | Location of data configuration files from base_dir. |
| `subfolders` | list[string] | See note | List of subfolders to search for data configuration files. |
| `name` | string | N/A | Prefix name of the data configuration files. |
| `extension` | string | '.toml' | Extension of the data configuration files. |

**NOTE:** The subfolders must be ['general', 'features', 'splitting', 'preprocessing'] for the data parameters.

**I/O table**

These paramaters are used for saving and loading models. The table is named [io_parameters].

| **Key** | **Type**  | **Default** | **Description** |
| :--- | :---  | :--- | :--- |
| `dir` | string  | N/A | Location of models to save or load from base_dir. |
| `name` | string | N/A | Prefix name of the I/O configuration files. |
| `extension` | string | '.pkl' | Extension of the models to load or save. |

**Fig table**

These paramaters are used for saving figures and results. The table is named [fig_parameters].

| **Key** | **Type**  | **Default** | **Description** |
| :--- | :---  | :--- | :--- |
| `dir` | string  | N/A | Location to save figures from base_dir. |
| `name` | string | N/A | Prefix name of the fig configuration files. |
| `extension` | string | '.png' | Extension of the figure to save. |

### Model subfolders
#### Calibrator
The calibrator configuration files contain the variables used to create the calibration layer. The calibrator model type is stored in the [calibrator] table.

| **Key** | **Type**  | **Description** |
| :--- | :---  | :--- |
| `calibrator_type` | string  | Type of calibrator model. |

Hyperparameters for the calibrator can be passed by specifying the keys in the [calibrator.hyperparameters] table. 

**NOTE:** The v1 file creates a logistic regression and the v2 a isotonic regression. 

#### Hyperparameters
The hyperparameters configuration files contain the hyperparameters for the predictor model that is used. The variables are contained in the [hyperparameters] table. The keys must match those for the model created. 

#### Imbalance
The imbalance configuration files contain the variables to address class imbalance in the data in a weighting and a sampler table with the following parameters:

**Weighting table**

These paramaters are used to apply weights for the examples or classes. The table is named [weighting].

| **Key** | **Type**  | **Description** |
| :--- | :---  | :--- |
| `weighting_fn` | string  | Weighting function to use. |

**Sampler table**

These paramaters are used to sample the examples. The table is named [sampler].

| **Key** | **Type**  | **Description** |
| :--- | :---  | :--- |
| `sampler_fn` | string  | Sampler function to use. |
| `target_ratio` | float  | Target ratio between majority and minority examples to achieve. |
| `hard_percent` | float  | Percent of examples deemed 'hard' to include. Only used with certain functions. |

#### Labels
The labels configuration files contain the list of prediction labels. The label list is contained in a TOML table named [labels].

| **Key** | **Type**  | **Description** |
| :--- | :---  | :--- |
| `label_list` | list[string]  |  List of the labels to predict |

**NOTE:** The label list version number is not an integer but a letter. 

## Logger configuration

The log_config.toml contains parameters for the logger.

| **Key** | **Type**  | **Default** | **Description** |
| :--- | :---  | :--- | :--- |
| `version` | string  | N/A | Version number for the logger configuration. |
| `base_dir` | string | N/A | Base directory location. |
| `log_dir` | string | N/A | Location to save log files in. |
| `log_message` | string | "Exception raised in " | Message used by logger when writing to log file. |
| `print_message` | string | "[ERROR] An exception was raised.\n\nCheck logs "| Message used by logger when printing to terminal. |
