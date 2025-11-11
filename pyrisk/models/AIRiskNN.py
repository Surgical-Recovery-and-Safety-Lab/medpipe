"""
AIRiskNN class.

This class creates an AI Risk neural network.

"""

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

    def __init__(self, n_features):
        """
        Initialise an AIRiskNN class instance.

        Parameters
        ----------
        n_features : int
            Number of features in the data.

        Returns
        -------
        None
            Nothing is returned.

        """
        super().__init__()
        self.save_file = "/home/mroe734/Documents/srs/AI-risk-score/models/NN_v0.1.1.pt"
        self.model = nn.Sequential(
            nn.Linear(n_features, n_features),
            nn.BatchNorm1d(n_features),
            nn.ReLU(),
            nn.Linear(n_features, 30),
            nn.BatchNorm1d(30),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(30, 60),
            nn.BatchNorm1d(60),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(60, 30),
            nn.BatchNorm1d(30),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(30, n_features),
            nn.Dropout(0.5),
            nn.BatchNorm1d(n_features),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(n_features, 1),
        )

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

    def fit(self, X, y, epochs=50, batch_size=64, lr=0.001):
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
            Learning rate for the optimizer.

        Returns
        -------
        None
            Nothing is returned.

        """
        self.train()  # Set model to training model

        # Convert data to tensors
        X_train = torch.tensor(X.to_numpy(dtype=float), dtype=torch.float32)
        y_train = torch.tensor(y, dtype=torch.float32)

        # Define the loss function and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

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
                optimizer.zero_grad()  # Zero the gradients

                outputs = self(inputs)  # Forward pass

                loss = criterion(outputs.squeeze(), labels)  # Compute loss
                loss.backward()  # Backpropagation
                optimizer.step()  # Update model weights

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

    def predict(self, X):
        """
        Predicts labels from input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        predictions array-like of shape (n_samples, n_classes)
            Predicted labels.

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
