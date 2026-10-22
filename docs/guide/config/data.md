# `[data]`

Where the dataset is, which columns the models learn from, and which columns they predict.

```toml
[data]
path = "../tests/test_data.csv"
predictors = [
  "AGE",
  "SEX",
  "PRIOR_CANCER",
  "ADMISSION_ACUITY",
  "ADMISSION_SOURCE",
  "CATEGORY_LEVEL_1",
  "OP_SEVERITY",
]
outcomes = ["MORTALITY_30D", "ANY_COMP"]
```

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `path` | string | required | Path to the dataset file. It must have a file extension. |
| `predictors` | list of strings | required | Columns used as model inputs. |
| `outcomes` | list of strings | required | Columns to predict. One model is fitted for each outcome. |
| `kwargs` | table | `{}` | Extra keyword arguments passed to the file reader. See [reader arguments](#reader-arguments). |

## Dataset file

`path` is used as written, so a relative path is relative to the directory the script is run from, not to the configuration file. The example uses `../tests/test_data.csv`, which works when it is run from the `examples/` folder.

The file is read with the pandas reader that matches its extension. The extension is not case-sensitive.

| Extension | Reader |
| :--- | :--- |
| `.csv`, `.txt` | `pandas.read_csv` |
| `.tsv` | `pandas.read_csv` with a tab separator |
| `.parquet`, `.pq` | `pandas.read_parquet` |
| `.feather` | `pandas.read_feather` |
| `.xlsx`, `.xls` | `pandas.read_excel` |
| `.json` | `pandas.read_json` |
| `.jsonl` | `pandas.read_json` with `lines=True` |
| `.pkl`, `.pickle` | `pandas.read_pickle` |

!!! note
    Reading Excel files needs the `openpyxl` package, which is not installed with medpipe. Install it with `pip install openpyxl`. Only load pickle files that you created yourself or trust, as unpickling can run arbitrary code.

Other formats can be supported by registering a reader with [`DataLoaderRegistry`][medpipe.utils.io.DataLoaderRegistry].

The file is not opened when the configuration is validated. A missing file, a directory, or an unsupported extension is reported when the data is loaded at the start of the run.

## Predictors and outcomes

Only the columns that the configuration refers to are kept from the file. All other columns are dropped when the data is loaded. Besides `predictors` and `outcomes`, the configuration can refer to columns in the following places, and these columns must exist in the file too:

* the `group_column` of a split or of cross-validation (see `[workflow.validation]`);
* the `strata` of the fairness analysis (see `[workflow.evaluation.fairness]`).

These extra columns are used to split the data or to describe subgroups. They are never given to the model as inputs unless they are also listed in `predictors`. In the example, `OP_YEAR` and `DHB_NAME` are used to split the data, and they are not among the predictors.

The following rules are checked:

* A column cannot be both a predictor and an outcome. Otherwise the model would be given the answer, and the configuration is rejected with the list of overlapping columns.
* Every column the configuration refers to must be present in the file. A missing column stops the run, before any model is fitted, with the list of missing columns.

Each outcome gets its own model. The models can share the same settings or be configured separately, as described in [`[outcome_overrides]`](outcome_overrides.md).

## Reader arguments

Anything under `[data.kwargs]` is passed to the pandas reader chosen for the file. This is useful for options such as the sheet of an Excel file or the separator of a text file:

```toml
[data]
path = "data/cohort.xlsx"
predictors = ["AGE", "SEX"]
outcomes = ["MORTALITY_30D"]

[data.kwargs]
sheet_name = "Cohort"
```

The arguments are not checked when the configuration is validated, so an argument the reader does not accept is reported when the data is loaded.
