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

    Methods
    -------
    forward(X)
        Forward pass method.

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
        self.model = nn.Sequential(
            nn.Linear(n_features, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid(),
        )

    def forward(self, X):
        """
        Forward pass method.

        Parameters
        ----------
        X : array-like
            Data to pass through the model.

        Returns
        -------
        logits : array-like
            Output of the model.

        """
        # Add a check that the data has the correct dimension
        logits = self.model(X)
        return logits

    def fit(self, X, y, epochs=20, batch_size=32, lr=0.001):
        """
        Train the model on the provided dataset.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples, n_classes)
            Prediction labels.
        epochs : int, default: 20
            Number of training epochs.
        batch_size : int, default: 32
            Size of the mini-batches for training.
        lr : float, default: 0.001
            Learning rate for the optimizer.

        Returns
        -------
        None
            Nothing is returned.

        """
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
                self.train()

                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self(inputs)

                # Calculate the loss
                loss = criterion(outputs.squeeze(), labels)

                # Backward pass (compute gradients)
                loss.backward()

                # Update the model weights
                optimizer.step()

                # Track loss and accuracy
                running_loss += loss.item()

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
