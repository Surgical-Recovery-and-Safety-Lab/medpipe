from pathlib import Path
from typing import Optional, Union

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from medpipe.data.preprocessing import PreprocessorRegistry
from medpipe.utils.config import MedpipeConfig
from medpipe.utils.io import load_data, read_toml_configuration
from medpipe.utils.logger import add_file_handler, get_console_logger
from medpipe.utils.reproducibility import ArtifactManager


class MedpipeOrchestrator:
    """
    Handles data ingress, transformation preparation, and reproducibility management.

    This orchestrator acts as the primary entry point for setting up the machine
    learning pipeline. It establishes the reproducibility environment, configures
    logging, ingests raw data, and builds the scikit-learn preprocessing pipeline.

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

        self.logger = get_console_logger("medpipe")
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

        self.artifact_manager.save_resolved_config(config_dict, self.run_dir)

        dataset_path = self.config.data.path if hasattr(self.config, "data") else None
        self.artifact_manager.save_env_state(
            destination_dir=self.run_dir,
            config=config_dict,
            dataset_path=dataset_path,
        )
        self.logger.info("Reproducibility artifacts saved successfully.")

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
