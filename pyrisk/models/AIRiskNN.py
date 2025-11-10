"""
AIRiskNN class.

This class creates an AI Risk neural network.

"""

import torch.nn as nn


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
            nn.Linear(10, 2),
            nn.Softmax(),
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
