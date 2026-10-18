"""
Publication-grade visualization suite for Turbomachinery Degradation & RUL Prognostics.

Generates 300 DPI figures:
1. degradation_trajectories.png: Multi-sensor operational wear trajectories across cycles.
2. rul_predictions_vs_actual.png: Predicted RUL vs True RUL for 100 test engines with NASA risk zones.
3. single_engine_degradation_profile.png: Lifetime trajectory of a sample engine from healthy state to failure.
4. training_loss_convergence.png: Epoch loss convergence comparison.
5. model_benchmark_comparison.png: RMSE and NASA score comparison across architectures.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Configure publication-grade styling
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.5,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def plot_sensor_degradation_trajectories(
    train_df: pd.DataFrame,
    sample_units: List[int] = [1, 2, 5, 8],
    save_path: Optional[str | Path] = None
) -> plt.Figure:
    """
    Plots degradation wear trajectories for key turbomachinery sensors across cycles.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.5), sharex=True)
    axes = axes.ravel()

    sensors_to_plot = [
        ("s_3", "HPC Outlet Temperature T30 (°R)", "#D32F2F"),
        ("s_4", "LPT Outlet Temperature T50 (°R)", "#E64A19"),
        ("s_7", "HPC Outlet Total Pressure P30 (psia)", "#1976D2"),
        ("s_11", "HPC Static Pressure Ps30 (psia)", "#7B1FA2")
    ]

    colors = ["#1E88E5", "#43A047", "#FB8C00", "#E53935"]

    for idx, (s_col, title, _) in enumerate(sensors_to_plot):
        ax = axes[idx]
        for u_i, unit in enumerate(sample_units):
            sub = train_df[train_df["unit_nr"] == unit]
            ax.plot(
                sub["time_cycles"], sub[s_col],
                color=colors[u_i % len(colors)],
                alpha=0.85, linewidth=1.6,
                label=f"Engine Unit #{unit}" if idx == 0 else ""
            )

        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.set_ylabel("Sensor Reading")
        ax.grid(True, linestyle=":", alpha=0.6)
        if idx >= 2:
            ax.set_xlabel("Operational Flight Cycles")

    axes[0].legend(loc="upper left", framealpha=0.9)
    fig.suptitle("NASA C-MAPSS Turbomachinery Sensor Degradation Trajectories", fontweight="bold", y=0.99)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        print(f"[Visualization] Saved Sensor Trajectories -> {save_path}")

    return fig


def plot_rul_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "CNN-LSTM Hybrid",
    metrics: Optional[Dict[str, Any]] = None,
    save_path: Optional[str | Path] = None
) -> plt.Figure:
    """
    Plots sorted 100 test engines with NASA asymmetric risk envelopes and parity scatter.
    """
    sort_idx = np.argsort(y_true)
    y_t_sorted = y_true[sort_idx]
    y_p_sorted = y_pred[sort_idx]
    engine_ranks = np.arange(1, len(y_true) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.2), gridspec_kw={"width_ratios": [1.4, 1.0]})

    # 1. Sorted Comparison Plot
    ax1.plot(engine_ranks, y_t_sorted, "k-", linewidth=2.5, label="Ground Truth RUL")
    ax1.scatter(engine_ranks, y_p_sorted, color="#00897B", s=28, alpha=0.9, label=f"Predicted ({model_name})", zorder=3)

    # Shading error bands
    ax1.fill_between(engine_ranks, y_t_sorted - 10, y_t_sorted + 10, color="#C8E6C9", alpha=0.45, label="±10 Cycles Tolerance")

    ax1.set_xlabel("Test Engines (Ranked by Actual RUL)", fontweight="bold")
    ax1.set_ylabel("Remaining Useful Life (Cycles)", fontweight="bold")
    ax1.set_title(f"RUL Forecasting Across 100 Test Fleet Engines", fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left", framealpha=0.92)

    # 2. Parity Scatter Plot
    min_v, max_v = 0, max(y_true.max(), y_pred.max()) + 10
    id_line = np.linspace(min_v, max_v, 100)
    ax2.scatter(y_true, y_pred, color="#1E88E5", alpha=0.65, s=24, edgecolors="none")
    ax2.plot(id_line, id_line, "k-", linewidth=1.5, label="Ideal Parity (y=x)")
    ax2.plot(id_line, id_line + 15, "r--", linewidth=1.0, alpha=0.7, label="Late Prediction Risk (+15)")
    ax2.plot(id_line, np.maximum(0, id_line - 15), "b--", linewidth=1.0, alpha=0.7, label="Early Maintenance (-15)")

    stats_str = ""
    if metrics:
        stats_str = f"RMSE = {metrics.get('rmse', 0):.2f} cycles\nNASA Score = {metrics.get('nasa_score', 0):.1f}\nMAE = {metrics.get('mae', 0):.2f} cycles"
        ax2.text(
            0.05, 0.92, stats_str,
            transform=ax2.transAxes, verticalalignment="top",
            fontsize=9.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.92)
        )

    ax2.set_xlabel("Actual RUL (Cycles)", fontweight="bold")
    ax2.set_ylabel("Predicted RUL (Cycles)", fontweight="bold")
    ax2.set_title("Parity & Asymmetric Risk Map", fontweight="bold")
    ax2.set_xlim(min_v, max_v)
    ax2.set_ylim(min_v, max_v)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower right", fontsize=8.5)

    fig.suptitle(f"Prognostic Evaluation on Unseen Turbofan Fleet ({model_name})", fontweight="bold", y=1.00)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        print(f"[Visualization] Saved RUL Predictions -> {save_path}")

    return fig


def plot_single_engine_lifecycle(
    cycles: np.ndarray,
    actual_rul: np.ndarray,
    predicted_rul: np.ndarray,
    engine_id: int = 1,
    save_path: Optional[str | Path] = None
) -> plt.Figure:
    """
    Plots the full lifetime degradation trajectory for an individual engine to failure.
    """
    fig, ax = plt.subplots(figsize=(8.5, 4.8))

    ax.plot(cycles, actual_rul, "k--", linewidth=2.2, label="Actual Piecewise RUL Target")
    ax.plot(cycles, predicted_rul, color="#E53935", linewidth=2.5, label="Deep LSTM Predicted RUL")

    # Failure alert threshold
    ax.axhline(0, color="black", linestyle="-", linewidth=1.2)
    ax.axhline(20, color="#FF9800", linestyle=":", linewidth=1.5, label="Maintenance Alert Threshold (RUL ≤ 20)")
    ax.fill_between(cycles, 0, 20, color="#FFE0B2", alpha=0.35)

    ax.set_xlabel("Operational Flight Cycles", fontweight="bold")
    ax.set_ylabel("Remaining Useful Life (Cycles)", fontweight="bold")
    ax.set_title(f"Engine Unit #{engine_id} Continuous Life Cycle Degradation Tracking", fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", framealpha=0.92)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        print(f"[Visualization] Saved Lifecycle Plot -> {save_path}")

    return fig


def plot_loss_convergence(
    history_dict: Dict[str, List[float]],
    save_path: Optional[str | Path] = None
) -> plt.Figure:
    """
    Plots training loss convergence across candidate architectures.
    """
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    colors = ["#757575", "#1E88E5", "#7B1FA2", "#2E7D32"]

    for idx, (m_name, losses) in enumerate(history_dict.items()):
        c = colors[idx % len(colors)]
        epochs = np.arange(1, len(losses) + 1)
        ax.plot(epochs, losses, color=c, linewidth=2.0, label=m_name)

    ax.set_xlabel("Training Epochs", fontweight="bold")
    ax.set_ylabel("MSE Loss (Cycles²)", fontweight="bold")
    ax.set_title("Training Loss Convergence Across Prognostic Architectures", fontweight="bold")
    ax.set_yscale("log")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", framealpha=0.92)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        print(f"[Visualization] Saved Loss Convergence -> {save_path}")

    return fig


def plot_architecture_benchmark_comparison(
    metrics_list: List[Dict[str, Any]],
    save_path: Optional[str | Path] = None
) -> plt.Figure:
    """
    Plots bar comparison of RMSE and NASA Asymmetric Score across models.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.5))

    names = [m["model_name"] for m in metrics_list]
    rmses = [m["rmse"] for m in metrics_list]
    scores = [m["nasa_score"] for m in metrics_list]

    colors = ["#90A4AE", "#42A5F5", "#AB47BC", "#26A69A"]

    # 1. RMSE Bar Chart (Lower is better)
    b1 = ax1.bar(names, rmses, color=colors, edgecolor="black", width=0.55)
    ax1.set_ylabel("Test RMSE (Cycles)", fontweight="bold")
    ax1.set_title("Prognostic Accuracy (RMSE)", fontweight="bold")
    ax1.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax1.set_xticklabels(names, rotation=15, ha="right")
    for bar in b1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, h + 0.3, f"{h:.1f}", ha="center", va="bottom", fontweight="bold", fontsize=10)

    # 2. NASA Score Bar Chart (Lower is better)
    b2 = ax2.bar(names, scores, color=colors, edgecolor="black", width=0.55)
    ax2.set_ylabel("NASA Asymmetric Loss S", fontweight="bold")
    ax2.set_title("Industrial Safety Risk Score (NASA Metric)", fontweight="bold")
    ax2.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax2.set_xticklabels(names, rotation=15, ha="right")
    for bar in b2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, h + 15, f"{h:.0f}", ha="center", va="bottom", fontweight="bold", fontsize=10)

    fig.suptitle("NASA C-MAPSS FD001 Prognostic Architecture Benchmark", fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        print(f"[Visualization] Saved Benchmark Comparison -> {save_path}")

    return fig


def plot_attention_heatmap(
    attention_weights: np.ndarray,
    engine_ids: List[int],
    save_path: Optional[str | Path] = None
) -> plt.Figure:
    """
    Plots temporal attention weights across the sliding window (T=30 cycles)
    for sample test engines, showing which historical cycles influenced the failure alert most.
    """
    fig, ax = plt.subplots(figsize=(10.0, 4.5))

    # attention_weights shape: [n_sample_engines, seq_len]
    seq_len = attention_weights.shape[1]
    time_steps = np.arange(1, seq_len + 1)

    im = ax.imshow(attention_weights, cmap="YlOrRd", aspect="auto", interpolation="nearest")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Normalized Attention Weight α_t", rotation=270, labelpad=15)

    ax.set_yticks(np.arange(len(engine_ids)))
    ax.set_yticklabels([f"Engine #{eid}" for eid in engine_ids])
    ax.set_xticks(np.arange(0, seq_len, 5))
    ax.set_xticklabels([f"t - {seq_len - t}" for t in np.arange(0, seq_len, 5)])

    ax.set_xlabel("Relative Time Step in Sliding Window (Cycles)", fontweight="bold")
    ax.set_ylabel("Inspected Turbofan Unit", fontweight="bold")
    ax.set_title("Temporal Self-Attention Weight Heatmap (Wear Inception Tracking)", fontweight="bold", pad=12)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        print(f"[Visualization] Saved Attention Heatmap -> {save_path}")

    return fig

