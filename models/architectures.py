"""
Deep Learning Architectures for Turbomachinery RUL Prognostics.

Implements a systematic progression of time-series neural architectures in PyTorch:
1. Baseline MLP (Flattened Window)
2. 1D Temporal CNN (Convolutional Feature Extractor)
3. Bidirectional LSTM (Bi-LSTM with Temporal Memory)
4. CNN-LSTM Hybrid (Spatial-Temporal State-of-the-Art)
"""

from __future__ import annotations

import sys
import torch
import torch.nn as nn

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


class MLPBaseline(nn.Module):
    """
    Standard Feed-Forward Neural Network on flattened temporal window.
    """
    def __init__(self, sequence_length: int = 30, n_features: int = 14):
        super().__init__()
        in_dim = sequence_length * n_features
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len, n_features]
        b = x.size(0)
        x_flat = x.view(b, -1)
        return self.net(x_flat)


class CNN1DModel(nn.Module):
    """
    1D Temporal Convolutional Neural Network.
    Filters learn local wear signatures across sensor time windows.
    """
    def __init__(self, sequence_length: int = 30, n_features: int = 14):
        super().__init__()
        # Conv1D expects: [batch_size, in_channels, seq_len]
        self.conv1 = nn.Conv1d(in_channels=n_features, out_channels=32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len, n_features] -> transpose to [batch_size, n_features, seq_len]
        x = x.transpose(1, 2)
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.gap(x).squeeze(-1)
        return self.fc(x)


class BidirectionalLSTM(nn.Module):
    """
    Deep Bidirectional Long Short-Term Memory Network.
    Models long-term thermodynamic component degradation and wear memory.
    """
    def __init__(self, n_features: int = 14, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.20):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len, n_features]
        out, _ = self.lstm(x)
        # Pool across sequence length (mean pooling)
        pooled = torch.mean(out, dim=1)
        return self.fc(pooled)


class CNNLSTMHybrid(nn.Module):
    """
    Hybrid CNN-LSTM Architecture.
    1D-CNN extracts local sensor cross-correlation features,
    followed by LSTM modeling temporal progression towards component failure.
    """
    def __init__(self, n_features: int = 14, hidden_dim: int = 64, dropout: float = 0.20):
        super().__init__()
        # Conv block: [batch, n_features, seq_len]
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=n_features, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        # LSTM input: [batch, seq_len, 32]
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=False,
            dropout=dropout
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, n_features] -> [batch, n_features, seq_len]
        x_conv = self.conv(x.transpose(1, 2))
        # -> [batch, seq_len, 32]
        x_lstm_in = x_conv.transpose(1, 2)
        out, (hn, _) = self.lstm(x_lstm_in)
        # Take last time step
        last_step = out[:, -1, :]
        return self.head(last_step)


def build_model(model_name: str, sequence_length: int = 30, n_features: int = 14) -> nn.Module:
    """
    Factory function for prognostic neural architectures.
    """
    name = model_name.lower()
    if "mlp" in name:
        return MLPBaseline(sequence_length=sequence_length, n_features=n_features)
    elif "cnn-lstm" in name or "hybrid" in name:
        return CNNLSTMHybrid(n_features=n_features)
    elif "cnn" in name:
        return CNN1DModel(sequence_length=sequence_length, n_features=n_features)
    elif "lstm" in name:
        return BidirectionalLSTM(n_features=n_features)
    else:
        raise ValueError(f"Unknown architecture: {model_name}")
