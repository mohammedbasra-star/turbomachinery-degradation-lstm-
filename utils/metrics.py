"""
NASA C-MAPSS Performance Metrics and Asymmetric Loss Evaluation.

Standard metrics used across turbomachinery prognostic literature:
1. RMSE (Root Mean Squared Error, cycles)
2. NASA Asymmetric Scoring Function:
   Penalizes late predictions (catastrophic engine failure) exponentially more heavily
   than early predictions (premature maintenance inspection).
   d_i = y_pred - y_true
   s_i = exp(-d_i / 13) - 1  for d_i < 0 (Late prediction)
   s_i = exp( d_i / 10) - 1  for d_i >= 0 (Early prediction)
   Total Score S = sum(s_i)
3. MAE (Mean Absolute Error, cycles)
"""

from __future__ import annotations

import sys
from typing import Dict, Any, List
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def nasa_scoring_function(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the official NASA C-MAPSS asymmetric evaluation score.
    Late predictions (y_pred > y_true, d > 0 in some conventions, here d = y_pred - y_true):
    When predicted RUL is greater than actual RUL (late maintenance), penalty is higher.
    """
    d = np.asarray(y_pred).ravel() - np.asarray(y_true).ravel()
    # If d < 0 (y_pred < y_true -> early maintenance): exp(-d/13) - 1
    # If d > 0 (y_pred > y_true -> late maintenance): exp(d/10) - 1
    scores = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(scores))


def calculate_rul_metrics(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model") -> Dict[str, Any]:
    """
    Computes full prognostic evaluation metrics on turbomachinery test engines.
    """
    y_t = np.asarray(y_true, dtype=np.float64).ravel()
    y_p = np.asarray(y_pred, dtype=np.float64).ravel()

    diff = y_p - y_t
    mse = np.mean(diff ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(diff))
    max_err = np.max(np.abs(diff))
    nasa_score = nasa_scoring_function(y_t, y_p)

    return {
        "model_name": model_name,
        "rmse": float(rmse),
        "nasa_score": float(nasa_score),
        "mae": float(mae),
        "max_err": float(max_err),
        "n_engines": len(y_t)
    }


def format_rul_metrics_table(metrics_list: List[Dict[str, Any]]) -> str:
    """
    Formats a markdown benchmark comparison table.
    """
    header = "| Model Architecture             | Test RMSE (Cycles) | NASA Score (S) | MAE (Cycles) | Max Error (Cycles) |"
    divider = "|:-------------------------------|:------------------:|:--------------:|:------------:|:------------------:|"
    rows = []
    for m in metrics_list:
        rows.append(
            f"| {m['model_name']:<30} | {m['rmse']:<18.2f} | {m['nasa_score']:<14.1f} | {m['mae']:<12.2f} | {m['max_err']:<18.2f} |"
        )
    return "\n".join([header, divider] + rows)


if __name__ == "__main__":
    y_t = np.array([30.0, 50.0, 10.0])
    y_p = np.array([28.0, 53.0, 9.0])
    print(format_rul_metrics_table([calculate_rul_metrics(y_t, y_p, "Test LSTM")]))
