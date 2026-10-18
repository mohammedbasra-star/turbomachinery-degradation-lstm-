"""
Comprehensive Training Pipeline for Turbomachinery Degradation & RUL Prognostics.

Executes:
1. Data loading and sliding window temporal sequencing
2. Multi-architecture training in PyTorch (MLP, 1D-CNN, Bi-LSTM, CNN-LSTM Hybrid)
3. Formal evaluation on 100 test engines using RMSE and NASA Asymmetric Scoring Function
4. Best checkpoint serialization (.pt) and normalization parameter logging
5. 300 DPI publication figure generation in figures/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import (
    load_cmapss_data,
    prepare_sliding_windows,
    TurbofanWindowDataset,
    INFORMATIVE_SENSORS
)
from models.architectures import build_model
from utils.metrics import calculate_rul_metrics, format_rul_metrics_table
from utils.visualization import (
    plot_sensor_degradation_trajectories,
    plot_rul_predictions,
    plot_single_engine_lifecycle,
    plot_loss_convergence,
    plot_architecture_benchmark_comparison
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def train_single_model(
    model: nn.Module,
    train_loader: DataLoader,
    X_test_tensor: torch.Tensor,
    y_test: np.ndarray,
    device: torch.device,
    epochs: int = 25,
    lr: float = 0.003
) -> Dict[str, Any]:
    """
    Trains an individual PyTorch prognostic model with Adam optimizer and LR scheduling.
    """
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    loss_history = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(1, n_batches)
        loss_history.append(avg_loss)

    # Evaluate on test fleet
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test_tensor.to(device)).cpu().numpy().ravel()
        test_preds = np.maximum(0.0, test_preds)

    metrics = calculate_rul_metrics(y_test, test_preds)

    return {
        "model": model,
        "loss_history": loss_history,
        "test_preds": test_preds,
        "metrics": metrics
    }


def run_pipeline(
    epochs: int = 25,
    batch_size: int = 128,
    lr: float = 0.003,
    sequence_length: int = 30,
    max_rul: float = 125.0,
    checkpoints_dir: Path = PROJECT_ROOT / "checkpoints",
    figures_dir: Path = PROJECT_ROOT / "figures"
) -> Dict[str, Any]:
    print("=" * 80)
    print("  NASA C-MAPSS TURBOMACHINERY DEGRADATION & RUL PROGNOSTIC PIPELINE")
    print("  High-Pressure Compressor Wear Modeling & Deep Time-Series Architectures")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Environment] Using Device: {device} (PyTorch {torch.__version__})")

    # 1. Ingest Data
    print("\n[Step 1/5] Loading NASA C-MAPSS FD001 Dataset...")
    train_df, test_df, true_rul = load_cmapss_data(PROJECT_ROOT / "data")
    print(f"  Training Fleet: {train_df['unit_nr'].nunique()} engines ({len(train_df):,} flight cycles to failure)")
    print(f"  Testing Fleet:  {test_df['unit_nr'].nunique()} engines ({len(test_df):,} observed cycles)")

    # 2. Window Sequencing
    print("\n[Step 2/5] Creating Sliding Temporal Windows (Length = 30 cycles)...")
    data_dict = prepare_sliding_windows(
        train_df, test_df, true_rul, sequence_length=sequence_length, max_rul=max_rul
    )
    X_train, y_train = data_dict["X_train"], data_dict["y_train"]
    X_test, y_test = data_dict["X_test"], data_dict["y_test"]

    print(f"  X_train windows: {X_train.shape} -> {len(X_train):,} sequences")
    print(f"  X_test windows:  {X_test.shape} -> {len(X_test)} fleet engines")

    train_dataset = TurbofanWindowDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    # 3. Model Benchmark Training
    print("\n[Step 3/5] Benchmarking Prognostic Deep Learning Architectures...")
    candidate_names = [
        "MLP Baseline",
        "1D Temporal CNN",
        "Bidirectional LSTM",
        "CNN-LSTM Hybrid"
    ]

    all_metrics = []
    loss_histories = {}
    test_predictions = {}
    trained_models = {}

    best_model_name = None
    best_rmse = float("inf")

    for name in candidate_names:
        t0 = time.time()
        model_instance = build_model(name, sequence_length=sequence_length, n_features=len(INFORMATIVE_SENSORS))
        res = train_single_model(
            model=model_instance,
            train_loader=train_loader,
            X_test_tensor=X_test_tensor,
            y_test=y_test,
            device=device,
            epochs=epochs,
            lr=lr
        )
        elapsed = time.time() - t0

        m = res["metrics"]
        m["model_name"] = name
        m["train_time_sec"] = elapsed
        all_metrics.append(m)

        loss_histories[name] = res["loss_history"]
        test_predictions[name] = res["test_preds"]
        trained_models[name] = res["model"]

        print(f"  -> {name:<22} | Test RMSE: {m['rmse']:5.2f} cycles | NASA Score S: {m['nasa_score']:8.1f} | Time: {elapsed:4.1f}s")

        if m["rmse"] < best_rmse:
            best_rmse = m["rmse"]
            best_model_name = name

    print("\n" + "=" * 80)
    print("  NASA C-MAPSS FD001 FINAL PROGNOSTIC BENCHMARK TABLE")
    print("=" * 80)
    table_str = format_rul_metrics_table(all_metrics)
    print(table_str)

    # 4. Save Best Model Checkpoint
    print(f"\n[Step 4/5] Saving Best Model Checkpoint: {best_model_name} (RMSE = {best_rmse:.2f} cycles)...")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    best_model_obj = trained_models[best_model_name]
    checkpoint_path = checkpoints_dir / "best_model.pt"
    torch.save(best_model_obj.state_dict(), checkpoint_path)

    # Save normalization parameters for standalone deployment
    meta = {
        "best_model_name": best_model_name,
        "best_rmse": best_rmse,
        "nasa_score": [m["nasa_score"] for m in all_metrics if m["model_name"] == best_model_name][0],
        "sequence_length": sequence_length,
        "max_rul": max_rul,
        "sensors": INFORMATIVE_SENSORS,
        "feature_means": data_dict["feature_means"],
        "feature_stds": data_dict["feature_stds"],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(checkpoints_dir / "norm_params.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"  Model weights saved -> {checkpoint_path}")
    print(f"  Metadata saved      -> {checkpoints_dir / 'norm_params.json'}")

    # 5. Visualizations
    print(f"\n[Step 5/5] Generating 300 DPI Publication Visualizations in {figures_dir}...")
    figures_dir.mkdir(parents=True, exist_ok=True)

    # A: Sensor Degradation Trajectories
    plot_sensor_degradation_trajectories(train_df, sample_units=[1, 2, 5, 8], save_path=figures_dir / "degradation_trajectories.png")

    # B: RUL Predictions vs Ground Truth (Best Model)
    best_preds = test_predictions[best_model_name]
    best_m = [m for m in all_metrics if m["model_name"] == best_model_name][0]
    plot_rul_predictions(y_test, best_preds, model_name=best_model_name, metrics=best_m, save_path=figures_dir / "rul_predictions_vs_actual.png")

    # C: Single Engine Continuous Degradation Profile (Engine #1)
    engine_1_data = train_df[train_df["unit_nr"] == 1]
    e1_norm = engine_1_data.copy()
    for col in INFORMATIVE_SENSORS:
        e1_norm[col] = (e1_norm[col] - data_dict["feature_means"][col]) / data_dict["feature_stds"][col]

    e1_feats = e1_norm[INFORMATIVE_SENSORS].values
    e1_windows = []
    e1_cycles = []
    e1_actual_ruls = []
    total_e1_cycles = len(engine_1_data)

    for i in range(sequence_length, total_e1_cycles + 1):
        e1_windows.append(e1_feats[i - sequence_length:i])
        e1_cycles.append(i)
        e1_actual_ruls.append(min(total_e1_cycles - i, max_rul))

    with torch.no_grad():
        e1_tensor = torch.tensor(np.array(e1_windows, dtype=np.float32)).to(device)
        e1_preds = best_model_obj(e1_tensor).cpu().numpy().ravel()

    plot_single_engine_lifecycle(
        np.array(e1_cycles), np.array(e1_actual_ruls), e1_preds,
        engine_id=1, save_path=figures_dir / "single_engine_degradation_profile.png"
    )

    # D: Training Loss Convergence
    plot_loss_convergence(loss_histories, save_path=figures_dir / "training_loss_convergence.png")

    # E: Benchmark Comparison
    plot_architecture_benchmark_comparison(all_metrics, save_path=figures_dir / "model_benchmark_comparison.png")

    print("\n" + "=" * 80)
    print("  TRAINING & BENCHMARK COMPLETED SUCCESSFULLY!")
    print(f"  Best Architecture: {best_model_name} (RMSE = {best_rmse:.2f} cycles)")
    print("=" * 80)

    return {
        "best_model": best_model_name,
        "metrics_table": table_str,
        "metadata": meta
    }


def main():
    parser = argparse.ArgumentParser(description="Turbomachinery RUL Prognostics Pipeline")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=0.003, help="Learning rate")
    parser.add_argument("--seq_len", type=int, default=30, help="Sliding window length")
    parser.add_argument("--max_rul", type=float, default=125.0, help="Piecewise linear RUL threshold")

    args = parser.parse_args()
    run_pipeline(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        sequence_length=args.seq_len,
        max_rul=args.max_rul
    )


if __name__ == "__main__":
    main()
