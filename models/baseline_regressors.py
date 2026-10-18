"""
Baseline Regressor Wrappers for Comparative Benchmarking on C-MAPSS.

Provides:
- MLP Baseline Regressor (Feed-forward PyTorch / Scikit-Learn wrapper)
- Random Forest Regressor (Ensemble non-linear bagging baseline)
"""

from __future__ import annotations

import sys
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


class BaselineMLP(nn.Module):
    """
    Feed-Forward baseline network operating on flattened temporal windows.
    """
    def __init__(self, sequence_length: int = 30, in_features: int = 14, dropout: float = 0.20):
        super().__init__()
        in_dim = sequence_length * in_features
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        x_flat = x.view(b, -1)
        return self.net(x_flat)


class BaselineRandomForest:
    """
    Scikit-Learn Random Forest wrapper on flattened sequence windows.
    """
    def __init__(self, n_estimators: int = 100, max_depth: int = 12, random_state: int = 42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=-1,
            random_state=random_state
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X_flat = X.reshape(len(X), -1)
        self.model.fit(X_flat, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_flat = X.reshape(len(X), -1)
        return self.model.predict(X_flat)
