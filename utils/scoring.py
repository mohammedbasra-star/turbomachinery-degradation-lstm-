"""
NASA Asymmetric Prognostic Scoring Function and Standard Evaluation Metrics.

NASA Score Formulation:
d = y_pred - y_true
s_i = exp(-d / 13) - 1   for d < 0 (Early prediction / Safe inspection)
s_i = exp( d / 10) - 1   for d >= 0 (Late prediction / Critical safety hazard)
Total Score S = sum(s_i)
"""

from __future__ import annotations

import sys
from typing import Dict, Any, List
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def nasa_scoring_function(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes official NASA asymmetric penalty score.
    """
    d = np.asarray(y_pred, dtype=np.float64).ravel() - np.asarray(y_true, dtype=np.float64).ravel()
    scores = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(scores))


def compute_evaluation_metrics(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model") -> Dict[str, Any]:
    """
    Computes RMSE, NASA Score S, MAE, and Max Absolute Error.
    """
    y_t = np.asarray(y_true, dtype=np.float64).ravel()
    y_p = np.asarray(y_pred, dtype=np.float64).ravel()

    diff = y_p - y_t
    rmse = np.sqrt(np.mean(diff ** 2))
    mae = np.mean(np.abs(diff))
    max_err = np.max(np.abs(diff))
    score = nasa_scoring_function(y_t, y_p)

    return {
        "model_name": model_name,
        "rmse": float(rmse),
        "nasa_score": float(score),
        "mae": float(mae),
        "max_err": float(max_err),
        "n_samples": len(y_t)
    }


def format_benchmark_table(metrics_list: List[Dict[str, Any]]) -> str:
    """
    Renders formatted Markdown comparison table.
    """
    header = "| Model Architecture             | Test RMSE (Cycles) | NASA Score (S) | MAE (Cycles) | Max Error (Cycles) |"
    divider = "|:-------------------------------|:------------------:|:--------------:|:------------:|:------------------:|"
    rows = []
    for m in metrics_list:
        rows.append(
            f"| {m['model_name']:<30} | {m['rmse']:<18.2f} | {m['nasa_score']:<14.1f} | {m['mae']:<12.2f} | {m['max_err']:<18.2f} |"
        )
    return "\n".join([header, divider] + rows)
