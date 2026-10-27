"""
model.py - Artificial Neural Network (ANN) Architectures for Binary Classification
on the make_moons dataset.
"""

from typing import List, Optional
import torch
import torch.nn as nn


class LinearClassifier(nn.Module):
    """
    A single-layer Perceptron (equivalent to Logistic Regression).
    Demonstrates the fundamental limitation of linear models on non-linearly
    separable datasets like make_moons.
    """
    def __init__(self, in_features: int = 2):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Returns raw logits for BCEWithLogitsLoss
        return self.linear(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        probs = self.predict_proba(x)
        return (probs >= threshold).long()


class DeepMoonANN(nn.Module):
    """
    Multi-Layer Artificial Neural Network (ANN) for Non-Linear Binary Classification.

    Architecture:
      Input (2)
        -> Linear(2 -> hidden[0]) -> BatchNorm1d -> Activation (ReLU/LeakyReLU) -> Dropout
        -> Linear(hidden[0] -> hidden[1]) -> BatchNorm1d -> Activation -> Dropout
        ...
        -> Linear(hidden[-1] -> 1) (Raw Logits)

    Using raw logits alongside nn.BCEWithLogitsLoss combines Sigmoid and
    Binary Cross-Entropy into one layer for maximum numerical stability
    (log-sum-exp trick prevents vanishing/exploding gradients).
    """
    def __init__(
        self,
        in_features: int = 2,
        hidden_dims: Optional[List[int]] = None,
        dropout_rate: float = 0.1,
        activation: str = "leaky_relu"
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 16, 8]

        self.in_features = in_features
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate

        # Select activation function
        if activation == "relu":
            act_fn = nn.ReLU()
        elif activation == "leaky_relu":
            act_fn = nn.LeakyReLU(negative_slope=0.1)
        elif activation == "tanh":
            act_fn = nn.Tanh()
        elif activation == "elu":
            act_fn = nn.ELU()
        else:
            act_fn = nn.ReLU()

        layers = []
        current_dim = in_features

        # Construct hidden layers with Batch Normalization, Activation, and Dropout
        for idx, h_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(act_fn)
            if dropout_rate > 0.0 and idx < len(hidden_dims) - 1:
                layers.append(nn.Dropout(p=dropout_rate))
            current_dim = h_dim

        # Final output projection to 1 logit
        layers.append(nn.Linear(current_dim, 1))

        self.network = nn.Sequential(*layers)

        # Apply Kaiming / He Weight Initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass outputting raw logits."""
        return self.network(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate class 1 probabilities using Sigmoid."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Output discrete binary predictions (0 or 1) based on decision threshold."""
        probs = self.predict_proba(x)
        return (probs >= threshold).long()

    def count_parameters(self) -> int:
        """Count total trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    linear_model = LinearClassifier()
    ann_model = DeepMoonANN(in_features=2, hidden_dims=[32, 16, 8], dropout_rate=0.1)

    dummy_input = torch.randn(5, 2)
    print("Linear model output shape:", linear_model(dummy_input).shape)
    print("Deep ANN output shape:", ann_model(dummy_input).shape)
    print(f"Deep ANN trainable parameters: {ann_model.count_parameters()}")
    print("Deep ANN predicted probabilities:\n", ann_model.predict_proba(dummy_input).numpy())
