"""
Bidirectional LSTM Augmented with Temporal Self-Attention Mechanism for Turbomachinery RUL Estimation.

Architecture:
1. Multi-Layer Bidirectional LSTM Backbone (capturing forward wear accumulation and backward context).
2. Temporal Additive Self-Attention Layer:
   Computes normalized weights alpha_t across the sliding time window [1, ..., T]
   e_t = v^T * tanh(W * h_t + b)
   alpha_t = exp(e_t) / sum(exp(e_t'))
   context = sum(alpha_t * h_t)
3. Fully-Connected Regression Head with Dropout(0.20) for Remaining Useful Life (RUL) prediction.
"""

from __future__ import annotations

import sys
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


class TemporalAdditiveAttention(nn.Module):
    """
    Computes temporal attention distribution over time-series sequence states.
    Allows inspecting which historical operational cycles contributed most to failure prediction.
    """
    def __init__(self, hidden_dim: int, attention_dim: int = 64):
        super().__init__()
        self.W = nn.Linear(hidden_dim, attention_dim, bias=True)
        self.v = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, lstm_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lstm_outputs: [batch_size, seq_len, hidden_dim]
        Returns:
            context_vector: [batch_size, hidden_dim]
            attention_weights: [batch_size, seq_len]
        """
        # score = v^T * tanh(W * h_t) -> [batch_size, seq_len, 1]
        energy = torch.tanh(self.W(lstm_outputs))
        scores = self.v(energy).squeeze(-1)  # [batch_size, seq_len]

        # Normalized attention weights
        attention_weights = F.softmax(scores, dim=-1)  # [batch_size, seq_len]

        # Context vector: sum(alpha_t * h_t)
        # [batch_size, 1, seq_len] x [batch_size, seq_len, hidden_dim] -> [batch_size, 1, hidden_dim]
        context = torch.bmm(attention_weights.unsqueeze(1), lstm_outputs).squeeze(1)

        return context, attention_weights


class BiLSTMAttention(nn.Module):
    """
    Production-grade Bidirectional LSTM with Temporal Attention for Turbofan RUL Prognostics.
    """
    def __init__(
        self,
        in_features: int = 14,
        hidden_dim: int = 64,
        num_layers: int = 2,
        attention_dim: int = 64,
        dropout: float = 0.20
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Bidirectional LSTM backbone
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # 2 * hidden_dim because bidirectional
        bidirectional_dim = hidden_dim * 2
        self.attention = TemporalAdditiveAttention(
            hidden_dim=bidirectional_dim,
            attention_dim=attention_dim
        )

        # Regression projection head
        self.head = nn.Sequential(
            nn.Linear(bidirectional_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input sliding windows of shape [batch_size, seq_len, in_features]
            return_attention: If True, returns (pred_rul, attention_weights)
        """
        # lstm_out: [batch_size, seq_len, 2 * hidden_dim]
        lstm_out, _ = self.lstm(x)

        # Context vector from attention
        context, attn_weights = self.attention(lstm_out)

        # Predict RUL
        rul_pred = self.head(context)

        if return_attention:
            return rul_pred, attn_weights
        return rul_pred


if __name__ == "__main__":
    net = BiLSTMAttention(in_features=14, hidden_dim=64, num_layers=2)
    dummy_x = torch.randn(8, 30, 14)
    pred, attn = net(dummy_x, return_attention=True)
    print("Pred RUL shape:", pred.shape)
    print("Attention weights shape:", attn.shape)
    print("Attention sums:", attn.sum(dim=-1).detach().numpy())
