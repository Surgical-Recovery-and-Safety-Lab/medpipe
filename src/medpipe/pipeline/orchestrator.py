from pathlib import Path
from typing import Any, Optional, Tuple, Union

import pandas as pd
from numpy.typing import NDArray
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from medpipe.data.registry import PreprocessorRegistry
from medpipe.data.utils import extract_labels, resolve_subgroup_mask, split_data
from medpipe.utils.config import MedpipeConfig
from medpipe.utils.io import load_data, read_toml_configuration
from medpipe.utils.logger import add_file_handler, get_console_logger
from medpipe.utils.reproducibility import ArtifactManager


class MedpipeOrchestrator:
    """
    Handles data ingress, transformation preparation, and reproducibility management.

    This orchestrator acts as the primary entry point for setting up the machine
    learning pipeline. It establishes the reproducibility environment, configures
    logging, ingests raw data, builds the scikit-learn preprocessing pipeline,
    and resolves configuration overrides and data splits.

    Parameters
    ----------
    config : Union[str, Path, MedpipeConfig]
        Path to the TOML configuration file or an instantiated MedpipeConfig object.
    base_artifact_dir : Union[str, Path], default="artifacts"
        Root directory where the versioned run artifacts and logs will be saved.

    Attributes
    ----------
    config : MedpipeConfig
        The resolved configuration object driving the pipeline.
    artifact_manager : ArtifactManager
        Manager handling the creation and population of reproducibility artifacts.
    run_dir : Path
        The specific, versioned directory path for the current execution run.
    logger : logging.Logger
        The configured logger instance for the orchestrator, routing to both
        console and the artifact directory.

    Methods
    -------
    ingest_data()
        Loads and validates the raw dataset specified in the configuration.
    prepare_data()
        Extracts labels and applies sequential splits for test and recalibration sets.
    extract_stratum_subgroup(X, column, group, y=None)
        Extract a stratified subgroup from features (and optional target labels).
    build_preprocessor()
        Constructs an sklearn Pipeline for data transformation.
    _save_reproducibility_artifacts()
        Persists the resolved configuration and environment state.
    _check_operation(op)
        Validates and retrieves a preprocessing operation.

    """

    def __init__(
        self,
        config: Union[str, Path, MedpipeConfig],
        base_artifact_dir: Union[str, Path] = "artifacts",
    ) -> None:
        if isinstance(config, (str, Path)):
            self.config = read_toml_configuration(config)
        elif isinstance(config, MedpipeConfig):
            self.config = config
        else:
            raise ValueError(
                "A configuration file or a MedpipeConfig must be specified."
            )

        self.artifact_manager = ArtifactManager(base_artifact_dir)
        self.run_dir = self.artifact_manager.create_run_directory()

        self.logger = get_console_logger("medpipe.orchestrator")
        add_file_handler(self.logger, log_dir=self.run_dir)

        self.logger.info(
            f"Initialised MedpipeOrchestrator. Run directory: {self.run_dir}"
        )

        self._save_reproducibility_artifacts()

    def _save_reproducibility_artifacts(self) -> None:
        """
        Saves the resolved configuration and environment state to the artifact directory.

        This method extracts the configuration state (handling different Pydantic
        versions) and writes it to disk alongside the runtime environment metadata.

        """
        config_dict = (
            self.config.model_dump()
            if hasattr(self.config, "model_dump")
            else dict(self.config)
        )

        dataset_path = self.config.data.path if hasattr(self.config, "data") else None
        self.artifact_manager.save_env_state(
            destination_dir=self.run_dir,
            config=config_dict,
            dataset_path=dataset_path,
        )
        self.logger.info("Reproducibility artifacts saved successfully.")

    def prepare_data(
        self,
    ) -> Tuple[
        pd.DataFrame,
        pd.DataFrame,
        Optional[pd.DataFrame],
        Optional[pd.DataFrame],
        pd.DataFrame,
        pd.DataFrame,
        Optional[NDArray],
    ]:
        """
        Ingests data, extracts labels, and performs configured train/recal/test splits.

        Returns
        -------
        X_train : pd.DataFrame
        y_train : pd.DataFrame
        X_recal : Optional[pd.DataFrame]
        y_recal : Optional[pd.DataFrame]
        X_test : pd.DataFrame
        y_test : pd.DataFrame
        groups_train : Optional[npt.NDArray]

        """
        data = self.ingest_data()
        outcomes = self.config.data.outcomes
        outcome_columns = pd.Index(outcomes)

        X, y_arr = extract_labels(data, outcomes)

        val_config = getattr(self.config.workflow, "validation", None)
        if not val_config:
            raise ValueError("Validation configuration is missing from workflow.")

        # 1. Apply Test Split
        test_cfg = val_config.test_split
        X_temp, y_temp_arr, X_test, y_test_arr = split_data(
            features=X,
            labels=y_arr,
            strategy=test_cfg.strategy,
            group_column=getattr(test_cfg, "group_column", None),
            values=getattr(test_cfg, "values", None),
            test_size=getattr(test_cfg, "test_size", None),
        )

        # Re-wrap multi-label arrays into DataFrames aligned with their X indices
        y_temp_df = pd.DataFrame(
            y_temp_arr, columns=outcome_columns, index=X_temp.index
        )
        y_test_df = pd.DataFrame(
            y_test_arr, columns=outcome_columns, index=X_test.index
        )

        # 2. Apply Recalibration Split (Optional)
        recal_cfg = getattr(val_config, "recalibration_split", None)

        if recal_cfg:
            X_train, y_train_arr, X_recal, y_recal_arr = split_data(
                features=X_temp,
                labels=y_temp_df.to_numpy(),
                strategy=recal_cfg.strategy,
                group_column=getattr(recal_cfg, "group_column", None),
                values=getattr(recal_cfg, "values", None),
                recalibration_size=getattr(recal_cfg, "recalibration_size", None),
            )
            y_train_df = pd.DataFrame(
                y_train_arr, columns=outcome_columns, index=X_train.index
            )
            y_recal_df = pd.DataFrame(
                y_recal_arr, columns=outcome_columns, index=X_recal.index
            )
            self.logger.info(
                f"Data successfully split into Train ({len(X_train):,}), "
                f"Test ({len(X_test):,}), and "
                f"Recalibration ({len(X_recal):,}) sets"
            )

        else:
            X_train = X_temp
            y_train_df = y_temp_df
            X_recal = None
            y_recal_df = None

            self.logger.info(
                f"Data successfully split into Train ({len(X_train):,}), and "
                f"Test ({len(X_test):,}) sets"
            )

        groups_train = None
        cv_cfg = getattr(val_config, "cross_validation", None)
        if cv_cfg and cv_cfg.group_column:
            if cv_cfg.group_column in data.columns:
                groups_train = data.loc[X_train.index, cv_cfg.group_column].to_numpy()
            else:
                raise KeyError(
                    f"Cross-validation group column '{cv_cfg.group_column}' "
                    f"was not found in dataset columns."
                )

            self.logger.info("Data successfully prepared with CV groups.")

        return X_train, y_train_df, X_recal, y_recal_df, X_test, y_test_df, groups_train

    def ingest_data(self) -> pd.DataFrame:
        """
        Loads the raw dataset specified in the configuration.

        Reads the dataset from the path defined in `config.data.path` and
        verifies that it is correctly loaded as a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            The loaded raw dataset.

        Raises
        ------
        TypeError
            If the loaded data is not a pandas DataFrame.

        """
        dataset_path = self.config.data.path
        self.logger.info(f"Ingesting data from {dataset_path}")

        data = load_data(dataset_path)
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"Input data should be a pd.DataFrame, but got {type(data)}"
            )

        return data

    def extract_stratum_subgroup(
        self,
        X: pd.DataFrame,
        column: str,
        group: Any,
        y: pd.DataFrame | None = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame | None]:
        """
        Extract a stratified subgroup from features (and optional target labels).

        Resolves continuous range bounds or discrete categorical identifiers.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix containing the stratification column.
        column : str
            Name of the column defining the stratum (e.g., 'AGE', 'SEX').
        group : Any
            The specific subgroup identifier or range (e.g., 'M', (18, 65), '[18, 65)').
        y : pd.DataFrame, optional
            Corresponding ground truth label DataFrame aligned with X. Default is None.

        Returns
        -------
        X_subgroup : pd.DataFrame
            Filtered feature DataFrame for the specified stratum group.
        y_subgroup : pd.DataFrame or None
            Filtered label DataFrame matching X_subgroup index, or None if y was not provided.

        """
        self.logger.info(
            f"Extracting subgroup for stratum '{column}' matching group: {group}"
        )

        mask = resolve_subgroup_mask(X, column=column, group=group)
        n_matched = mask.sum()

        if n_matched == 0:
            self.logger.warning(
                f"Subgroup extraction returned 0 samples for column '{column}' and group '{group}'."
            )

        X_subgroup = X.loc[mask].copy()
        y_subgroup = y.loc[mask].copy() if y is not None else None

        return X_subgroup, y_subgroup

    def build_preprocessor(self) -> Optional[Pipeline]:
        """
        Constructs an sklearn Pipeline for data transformation based on the configuration.

        Parses the workflow configuration to build a sequence of `ColumnTransformer`
        objects, mapping specific operations to requested columns.

        Returns
        -------
        Optional[Pipeline]
            A configured scikit-learn `Pipeline` containing the cascaded
            transformations, or `None` if preprocessing is disabled or omitted.

        """
        preprocessing_dict = self.config.workflow.preprocessing

        if preprocessing_dict and preprocessing_dict.preprocess:
            self.logger.info("Constructing preprocessing pipeline.")
            steps = []

            if preprocessing_dict.operations:
                ct_columns_dict = {pred: pred for pred in self.config.data.predictors}

                for i, operation in enumerate(preprocessing_dict.operations):
                    op_type = self._check_operation(operation.name)
                    op_extras = (
                        {} if not operation.model_extra else operation.model_extra
                    )

                    ct_columns = [
                        ct_columns_dict[column] for column in operation.columns
                    ]

                    ct = ColumnTransformer(
                        [(f"op_{i+1}", op_type(**op_extras), ct_columns)],
                        remainder="passthrough",
                    )
                    ct.set_output(transform="pandas")

                    if i == len(preprocessing_dict.operations) - 1:
                        ct.set_output(transform="default")

                    steps.append((f"transformer_{i+1}", ct))

                    ct_columns_dict = {
                        pred: (
                            f"remainder__{column}"
                            if column not in ct_columns
                            else f"op_{i+1}__{column}"
                        )
                        for (pred, column) in ct_columns_dict.items()
                    }

            return Pipeline(steps=steps)

        self.logger.info(
            "No preprocessing steps specified; skipping pipeline creation."
        )
        return None

    def _check_operation(self, op: str) -> type:
        """
        Validates whether a requested preprocessing operation exists.

        Delegates the lookup to the `PreprocessorRegistry`.

        Parameters
        ----------
        op : str
            The string name of the operation to validate and retrieve.

        Returns
        -------
        type
            The uninstantiated class for the preprocessing operation.

        Raises
        ------
        ValueError
            If the operation cannot be found in the registry or scikit-learn.

        """
        return PreprocessorRegistry.get(op)
