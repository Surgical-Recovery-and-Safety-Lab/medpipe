### Simple example
Here is a short example that shows how to load data, train the models, and plot the calibration curves with the quantile strategy:

``` py linenums="1", title="Simple example"
from medpipe import (
	Pipeline
	read_toml_configuration,
	load_data_from_csv,
	get_positive_proba,
	extract_labels,
	plot_reliability_diagrams,
)

# Load configuration and data
config = read_toml_configuration("config_file.toml")
data = load_data_from_csv("data.csv")

# Create pipeline
pipeline = Pipeline(general_config)

# Split data into sets and train model
X_train, X_test = pipeline.get_test_data(data)
pipeline.run(X_train)

# Plot calibration curve
X_test, y_test = extract_labels(X_test, pipeline.label_list)
y_pred_proba = pipeline.predict_proba(X_test)
plot_reliability_diagrams(y_test, get_positive_proba(y_pred_proba, display_kwargs={"n_bins": 10, "strategy": "quantile"})

```

### Setting up a logger
Here is an example that shows how to create a logger and change the level of printing to the terminal:

``` py title="Setting up a logger", linenums="1"
from medpipe import (
    print_message,
    read_toml_configuration,
    setup_logger,
)

if __name__ == "__main__":
    print_message("Loading parameters from configuration file")
       
    # Read log and general configuration file
    log_config = read_toml_configuration("log_config.toml")

    print_message("Setting up logger")
        
    # Create logger
    script_name = "setup_logger.py"
    log_dir = log_config["base_dir"] + log_config["log_dir"]
    log_dir = log_config["log_dir"]
    logger = setup_logger(script_name, log_dir)

	# This message prints to both log file and terminal.
	print_message(
		"Printing a message in log file and on the terminal.",
		logger,
		script_name
	)
    
    logger.setLevel(-1)  # Turn of terminal printing
    
    # This message prints only to the log file.
	print_message(
		"Printing a message only to log file.",
		logger,
		script_name
	)
        

```

### Loading and preprocessing data
This examples loads the data from the information in the configuration files and fits the preprocessing operations specified in the model configuration. The pipeline is then saved, with the preprocessing operations fitted.

``` py linenums="1", title="Loading and preprocessing data"
from medpipe import (
    Pipeline,
    exception_handler,
    load_data_from_csv,
    print_message,
    read_toml_configuration,
    setup_logger,
    save_pipeline,
)
from medpipe.utils.config import get_configuration, get_file_path, split_version_number

if __name__ == "__main__":
	print_message("Loading parameters from configuration file")
	
    # Read log and general configuration file
    log_config = read_toml_configuration("log_config.toml")
    general_config = read_toml_configuration("config_file.toml")

    print_message("Setting up logger")
    
    # Create logger
    script_name = "load_data.py"
    log_dir = log_config["base_dir"] + log_config["log_dir"]
    log_dir = log_config["log_dir"]
    logger = setup_logger(script_name, log_dir)
     
    print_message(
       f"Version number: {general_config["version"]}", logger, script_name
    )
     
    try:
        data_version, _ = split_version_number(general_config["version"])

        # Get data configuration parameters and load the data
        data_config = get_configuration(
            general_config["data_parameters"],
            data_version,
        )
        data = load_data_from_csv(
            get_file_path(  # Get data path based on config parameters
                data_config, v_number=data_version[:4]  # Use only first 2 numbers
            )
        )
        
        # Create a Pipeline and fit the preprocessing operations
        pipeline = Pipeline(general_config, logger)
        pipeline.preprocessor.fit(data)
        
        # Save pipeline
        print_message("Saving pipeline", logger, script_name)
        save_file = get_file_path(  # Get save file based on config parameters
            general_config,
            v_number=general_config["version"],
            exists=False,
        )
        save_pipeline(pipeline, save_file)

    except Exception:
    		# Catch exceptions and log them
        exception_handler(logger, log_dir, log_config, script_name)
        exit(1)

```

### Load a fitted Pipeline
This final example loads an already fitted Pipeline and plots the score metrics with confidence intervals (CIs). The predicted probabilities from each training fold are loaded and used to calculate the score metrics. From these values, the CIs are calculated and the mean and CIs are plotted. 

``` py linenums="1", title="Load a fitted Pipeline"
from medpipe import (
    Pipeline,
    compute_all_CI,
    compute_score_metrics,
    exception_handler,
    extract_labels,
    get_full_proba,
    get_positive_proba,
    load_data_from_csv,
    load_pipeline,
    plot_metrics_CI,
    print_message,
    read_toml_configuration,
    setup_logger,
)
from medpipe.utils.config import get_configuration, get_file_path, split_version_number

if __name__ == "__main__":
    try:
        print_message("Loading parameters from configuration file")
        
        # Read log and general configuration file
    		log_config = read_toml_configuration("log_config.toml")
    		general_config = read_toml_configuration("config_file.toml")

        print_message("Setting up logger")
        
        # Create logger
        script_name = "load_pipeline.py"
        log_dir = log_config["base_dir"] + log_config["log_dir"]
        log_dir = log_config["log_dir"]
        logger = setup_logger(script_name, log_dir)

        print_message(
            f"Version number: {general_config["version"]}", logger, script_name
        )

    except (TypeError, ValueError, FileNotFoundError, IsADirectoryError) as err:
        sys.stderr.write("An error occured when trying to create the logger\n")
        sys.stderr.write(repr(err))
        exit(1)

    try:
         data_version, _ = split_version_number(general_config["version"])

        # Get data configuration parameters and load the data
        data_config = get_configuration(
            general_config["data_parameters"],
            data_version,
        )
        data = load_data_from_csv(
            get_file_path(  # Get data path based on config parameters
                data_config, v_number=data_version[:4]  # Use only first 2 numbers
            )
        )
        
        # Load model
        print_message("Loading model", logger, script_name)
        load_file = get_file_path(
            general_config,
            v_number=general_config["version"],
        )
        pipeline = load_pipeline(load_file)
        
        # Transform the loaded data based on fitted operations
        data = pipeline.preprocessor.transform(data)
        
        # Get the test set from the data
        print_message("Preparing test set", logger, script_name)
        X_train, X_test = pipeline.get_test_data(data)
        X_test, y_test = extract_labels(X_test, pipeline.label_list)

    		# Compute statistics and plots
        print_message("Computing model statistics", logger, script_name)
        group_name = data_config["split_variables"]["group_name"]

        extension = general_config["fig_parameters"]["extension"]
        label_list = ["Unadjusted", "Recalibrated"]

        for i, label in enumerate(pipeline.label_list):
            # Plot for each outcome individually
            metric_dict = {}  # Store unadjusted values
            metric_dict_cal = {}  # Store recalibrated values

            for key in pipeline.predictor_probabilities[label]:
                # Compute metric values for both unadjusted and recalibrated
                y_true = y_train[X_train[group_name] == key]
                metric_dict[key] = compute_score_metrics(
                    ["auroc", "ap", "log_loss"],
                    y_true[:, i],
                    get_full_proba(pipeline.predictor_probabilities[label][key]),
                )
                metric_dict_cal[key] = compute_score_metrics(
                    ["auroc", "ap", "log_loss"],
                    y_true[:, i],
                    get_full_proba(pipeline.calibrator_probabilities[label][key]),
                )

                for k in metric_dict[key].keys():
                    metric_dict[key][k] += metric_dict_cal[key][k]

            # Create one CI dict for both outcomes and plot results
            ci_dict = compute_all_CI(metric_dict)
            plot_metrics_CI(
                ci_dict,
                label_list=label_list,
                dpi=300,
                figsize=(5, 5),
                extension=extension,
            )

    except Exception:
        exception_handler(logger, log_dir, log_config, script_name)
        exit(1)
```
