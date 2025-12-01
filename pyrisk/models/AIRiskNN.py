"""
AIRiskNN class.

This class creates an AI Risk neural network.

"""

import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score
from torchsummary import summary

from pyrisk.utils.logger import print_message


class AIRiskNN(nn.Module):
    """
    Class that creates a Pytorch neural network.

    Attributes
    ----------
    model : nn.Sequential
        Pytorch model.
    logger : logging.Logger or None, default: None
        Logger object to log prints. If None print to terminal.

    Methods
    -------
    forward(X)
        Forward pass method.
    fit(X,
        y,
        X_test=[],
        y_test=[],
        epochs=50,
        batch_size=64,
        lr=0.001,
        loss_name="BECWithLogitsLoss",
        optim_name="Adam"
        )
        Train the model on the provided dataset.
    predict_proba(X)
        Predicts probabilities from input data.
    predict(X)
        Predicts labels from input data.
    parse_architecture(architecture, n_features)
        Parse the architecture of the model.
    compute_precision(y_true, y_pred)
        Computes precision based on true and predicted labels.
    """

    def __init__(self, n_features, logger=None, **architecture):
        """
        Initialise an AIRiskNN class instance.

        Parameters
        ----------
        n_features : int
            Number of features in the data.
        logger : logging.Logger or None, default: None
            Logger object to log prints. If None print to terminal.
        architecture
            Model architecture dictionary.

        Returns
        -------
        None
            Nothing is returned.

        """
        super().__init__()
        layers_dict = self.parse_architecture(architecture, n_features)
        self.model = nn.Sequential(layers_dict)
        self.logger = logger
        stats = summary(self.model, (n_features,), verbose=0)
        print_message(str(stats), logger, "models/AIRiskNN")

    def forward(self, X):
        """
        Forward pass method.

        Parameters
        ----------
        X : torch.tensor
            Data to pass through the model.

        Returns
        -------
        logits : array-like
            Output of the model.

        """
        return self.model(X)

    def fit(
        self,
        X,
        y,
        X_test=[],
        y_test=[],
        class_weights=[],
        epochs=50,
        batch_size=64,
        lr=0.001,
        loss_name="BCEWithLogitsLoss",
        optim_name="Adam",
    ):
        """
        Train the model on the provided dataset.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples, n_classes)
            Prediction labels.
        X_test : array-like, default: []
            Test data.
        y_test : array-like, default: []
            Test labels.
        class_weights : array-like, default: []
            Class weights for loss function.
        epochs : int, default: 50
            Number of training epochs.
        batch_size : int, default: 64
            Size of the mini-batches for training.
        lr : float, default: 0.001
            Learning rate for the optimiser.
        loss_name : str, default: "BCEWithLogitsLoss"
            Name of the loss function to use.
        optim_name : str, default: "Adam"
            Name of the optimiser to use.

        Returns
        -------
        None
            Nothing is returned.

        """
        self.train()  # Set model to training model

        # Convert data to tensors
        X_train = torch.tensor(X.to_numpy(dtype=float), dtype=torch.float32)
        y_train = torch.tensor(y.squeeze(), dtype=torch.float32)
        X_test = torch.tensor(X_test.to_numpy(dtype=float), dtype=torch.float32)
        y_test = torch.tensor(y_test.squeeze(), dtype=torch.float32)

        if len(class_weights) == 0:
            # No weigths provided, default to 1
            class_weights = torch.ones(y_test.shape[1])

        # Define the loss function and optimiser
        criterion = getattr(nn, loss_name)(
            pos_weight=torch.tensor(class_weights, dtype=torch.float32)
        )
        optimiser = getattr(optim, optim_name)(self.parameters(), lr=lr)

        # Create data loaders for batching
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

        # Training loop
        for epoch in range(epochs):
            start_time = time.time()
            running_loss = 0.0
            train_predictions = []
            train_labels = []
            message = f"    Epoch {epoch+1}/{epochs} | "

            for inputs, labels in train_loader:
                optimiser.zero_grad()  # Zero the gradients

                outputs = self(inputs)  # Forward pass
                loss = criterion(outputs.squeeze(), labels)  # Compute loss
                loss.backward()  # Backpropagation
                optimiser.step()  # Update model weights

                running_loss += loss.item()  # Track loss

                # Calculate accuracy
                predicted = torch.round(torch.sigmoid(outputs))  # Round
                train_predictions.append(predicted.squeeze())
                train_labels.append(labels.squeeze())

            # Create epoch-level predictions and labels
            train_predictions = torch.cat(train_predictions).detach().numpy()
            train_labels = torch.cat(train_labels).numpy()

            # Compute train statistics
            avg_loss = running_loss / len(train_loader)
            train_precision = self.compute_precision(train_labels, train_predictions)
            message += f"Loss {avg_loss:.4f} | Train precision {train_precision:.4f} | "

            # Compute test statistics
            if len(X_test) > 0 and len(y_test) > 0:
                self.eval()  # Set to eval mode for testing
                with torch.no_grad():  # Disable gradient calculation
                    outputs = self(X_test)
                    test_predictions = torch.round(torch.sigmoid(outputs))
                    test_precision = self.compute_precision(y_test, test_predictions)
                    message += f"Test precision {test_precision:.4f} | "

                self.train()  # Reset to train

            epoch_time = time.time() - start_time

            message += f"Compute time {epoch_time:.3f} s"

            if self.logger:
                print_message(message, self.logger, "models/AIRiskNN")
            else:
                print(message)

    def predict_proba(self, X):
        """
        Predicts probabilities from input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        probabilities : np.array (n_classes,) of arrays (n_samples, 2)
            Predicted probabilities.

        """
        # Set to evaluate mode
        self.eval()

        probabilities = []

        # Convert data to be correct
        X_pred = torch.tensor(X.to_numpy(dtype=float), dtype=torch.float32)

        with torch.no_grad():
            # Get model output (logits)
            logits = self(X_pred)

            # Apply sigmoid to convert logits to probabilities
            outputs = torch.sigmoid(logits)

        for i in range(outputs.shape[1]):
            probabilities.append(np.array([1 - outputs[:, i], outputs[:, i]]).T)

        if outputs.shape[1] == 1:
            return probabilities[0]
        else:
            return probabilities

    def predict(self, X):
        """
        Predicts labels from input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        predictions : array-like of shape (n_samples, n_classes)
            Predicted labels.

        """
        probabilities = self.predict_proba(X)  # Get probabilities

        # Convert probabilities to binary predictions (0 or 1)
        if type(probabilities) is not type(list()):
            # Single label
            predictions = np.argmax(probabilities, axis=1)
        else:
            # Multilabel
            predictions = np.argmax(probabilities, axis=0)

        return predictions

    def parse_architecture(self, architecture, n_features):
        """
        Parse the architecture of the model.

        Parameters
        ----------
        architecture : dict
            Architecture of the model.
        n_features : int
            Number of features for first layer.

        Returns
        -------
        layers_dict : OrderedDict
            Ordered dictionary to create Sequential model.

        """
        layers = []
        nb_feats = n_features

        for i, layer_config in enumerate(architecture["layers"]):
            layer_type = layer_config.pop("type")  # Layer type
            layer_fn = getattr(nn, layer_type)  # Create an layer function

            match layer_type:
                case "Linear":
                    # If Linear layer add in_features
                    layer_config["in_features"] = nb_feats
                    nb_feats = layer_config["out_features"]  # Reset number of features

                case "BatchNorm1d":
                    # If BatchNorm1D add num_features
                    layer_config["num_features"] = nb_feats

            layer = layer_fn(**layer_config)

            layers.append((layer_type + f"_{i}", layer))  # Appends to layer list

        return OrderedDict(layers)

    def compute_precision(self, y_true, y_pred):
        """
        Computes the precision from true and predicted labels.

        Parameters
        ----------
        y_true : array-like of shape (n_samples, n_classes)
            Ground truth labels.
        y_pred : array-like of shape (n_samples, n_classes)
            Prediction labels.

        Returns
        -------
        precision : float
            Precision.

        """
        # Support both binary and multi-label cases
        if y_true.ndim == 1:
            precision = precision_score(y_true, y_pred, zero_division=0.0)
        else:
            precision = precision_score(
                y_true, y_pred, average="macro", zero_division=0.0
            )
        return precision
