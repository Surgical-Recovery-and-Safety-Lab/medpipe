"""
AIRiskNN class.

This class creates an AI Risk neural network.

"""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim


class AIRiskNN(nn.Module):
    """
    Class that creates a Pytorch neural network.

    Attributes
    ----------
    model : nn.Sequential
        Pytorch model.

    Methods
    -------
    forward(X)
        Forward pass method.
    fit(X, y, epochs=50, batch_size=64, lr=0.001)
        Train the model on the provided dataset.
    predict(X)
        Predicts labels from input data.
    save_model(save_file)
        Saves model weights to a file.
    load_model(load_file)
        Loads model weights from a file.

    """

    def __init__(self, n_features, **architecture):
        """
        Initialise an AIRiskNN class instance.

        Parameters
        ----------
        n_features : int
            Number of features in the data.
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
        y_train = torch.tensor(y, dtype=torch.float32)

        # Define the loss function and optimiser
        criterion = getattr(nn, loss_name)()
        optimiser = getattr(optim, optim_name)(self.parameters(), lr=lr)

        # Create data loaders for batching
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        # Training loop
        for epoch in range(epochs):
            running_loss = 0.0
            correct_preds = 0
            total_preds = 0

            for inputs, labels in train_loader:
                optimiser.zero_grad()  # Zero the gradients

                outputs = self(inputs)  # Forward pass
                loss = criterion(outputs.squeeze(), labels)  # Compute loss
                loss.backward()  # Backpropagation
                optimiser.step()  # Update model weights

                running_loss += loss.item()  # Track loss

                # Calculate accuracy
                predicted = torch.round(torch.sigmoid(outputs))  # Round
                correct_preds += (predicted.squeeze() == labels).sum().item()
                total_preds += labels.size(0)

            # Print statistics after each epoch
            avg_loss = running_loss / len(train_loader)
            accuracy = 100 * correct_preds / total_preds
            print(
                f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%"
            )

    def predict_proba(self, X):
        """
        Predicts probabilities from input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        probabilities : array-like of shape (n_samples, n_classes)
            Predicted probabilities.

        """
        # Set to evaluate mode
        self.eval()

        # Convert data to be correct
        X_pred = torch.tensor(X.to_numpy(dtype=float), dtype=torch.float32)

        with torch.no_grad():
            # Get model output (logits)
            logits = self(X_pred)

            # Apply sigmoid to convert logits to probabilities
            probabilities = torch.sigmoid(logits)
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
        predictions = torch.round(
            probabilities
        ).squeeze()  # Remove unnecessary dimensions

        return predictions

    def save_model(self, save_file):
        """
        Saves model weights to file.

        Parameters
        ----------
        save_file : str
            Path to the file to save the model.

        Returns
        -------
        None
            Nothing is returned.

        """
        torch.save(self.model.state_dict(), save_file)

    def load_model(self, load_file: str):
        """
        Loads model weights from a file.

        Parameters
        ----------
        load_file
            File to the model weights.

        Returns
        -------
        None
            Nothing is returned.

        """
        self.model.load_state_dict(torch.load(load_file, weights_only=True))

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
