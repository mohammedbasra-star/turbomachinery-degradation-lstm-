"""
Comprehensive Training Pipeline for Turbomachinery Degradation & RUL Prognostics.
Features:
- Bidirectional LSTM Augmented with Temporal Self-Attention Mechanism
- Piecewise Linear RUL Target Clipping (Heimes, 2008)
- Smooth L1 / Huber Loss for Robust Error Minimization
- NASA Asymmetric Prognostics Scoring & Test Fleet Benchmarking
- Multi-Model Comparison (Random Forest, MLP Baseline, BiLSTM, BiLSTM+Attention)
- Attention Weight Heatmap Export
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
from models.lstm_attention import BiLSTMAttention
from models.baseline_regressors import BaselineMLP, BaselineRandomForest
from models.architectures import BidirectionalLSTM
from utils.scoring import nasa_scoring_function, compute_evaluation_metrics, format_benchmark_table
from utils.visualization import (
    plot_sensor_degradation_trajectories,
    plot_rul_predictions,
    plot_single_engine_lifecycle,
    plot_loss_convergence,
    plot_architecture_benchmark_comparison,
    plot_attention_heatmap
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def train_pytorch_model(
    model: nn.Module,
    train_loader: DataLoader,
    X_test_tensor: torch.Tensor,
    y_test: np.ndarray,
    device: torch.device,
    epochs: int = 25,
    lr: float = 0.003
) -> Dict[str, Any]:
    model = model.to(device)
    criterion = nn.SmoothL1Loss(beta=1.0)  # Robust Huber loss
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

    # Evaluate
    model.eval()
    with torch.no_grad():
        if isinstance(model, BiLSTMAttention):
            test_preds, attn_weights = model(X_test_tensor.to(device), return_attention=True)
            test_preds = test_preds.cpu().numpy().ravel()
            attn_weights = attn_weights.cpu().numpy()
        else:
            test_preds = model(X_test_tensor.to(device)).cpu().numpy().ravel()
            attn_weights = None

        test_preds = np.maximum(0.0, test_preds)

    metrics = compute_evaluation_metrics(y_test, test_preds)

    return {
        "model": model,
        "loss_history": loss_history,
        "test_preds": test_preds,
        "attn_weights": attn_weights,
        "metrics": metrics
    }


def run_pipeline(
    epochs: int = 25,
    batch_size: int = 128,
    lr: float = 0.003,
    sequence_length: int = 30,
    max_rul: float = 125.0,
    results_dir: Path = PROJECT_ROOT / "results",
    figures_dir: Path = PROJECT_ROOT / "figures"
) -> Dict[str, Any]:
    print("=" * 80)
    print("  NASA C-MAPSS TURBOMACHINERY PROGNOSTICS WITH TEMPORAL ATTENTION")
    print("  BiLSTM + Self-Attention Architecture for High-Pressure Compressor Degradation")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Environment] Device: {device} | PyTorch: {torch.__version__}")

    # 1. Ingest Data
    print("\n[Step 1/5] Loading NASA C-MAPSS FD001 Dataset...")
    train_df, test_df, true_rul = load_cmapss_data(PROJECT_ROOT / "data")
    print(f"  Training Fleet: {train_df['unit_nr'].nunique()} engines ({len(train_df):,} cycles)")
    print(f"  Testing Fleet:  {test_df['unit_nr'].nunique()} engines ({len(test_df):,} cycles)")

    # 2. Window Sequencing
    print(f"\n[Step 2/5] Creating Sliding Temporal Windows (Length = {sequence_length} cycles)...")
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
    print("\n[Step 3/5] Benchmarking Models on NASA Asymmetric Metric...")
    all_metrics = []
    loss_histories = {}
    test_predictions = {}
    trained_models = {}
    saved_attention = None

    # A: Random Forest Baseline
    print("  -> Training Baseline Random Forest...")
    rf = BaselineRandomForest(n_estimators=100, max_depth=12)
    rf.fit(X_train, y_train)
    rf_preds = np.maximum(0.0, rf.predict(X_test))
    rf_metrics = compute_evaluation_metrics(y_test, rf_preds, model_name="Random Forest Baseline")
    all_metrics.append(rf_metrics)
    test_predictions["Random Forest Baseline"] = rf_preds
    print(f"     RMSE: {rf_metrics['rmse']:5.2f} cy | NASA Score S: {rf_metrics['nasa_score']:8.1f}")

    # B: PyTorch Neural Architectures
    neural_architectures = {
        "MLP Baseline": BaselineMLP(sequence_length=sequence_length, in_features=len(INFORMATIVE_SENSORS)),
        "Bidirectional LSTM": BidirectionalLSTM(n_features=len(INFORMATIVE_SENSORS), hidden_dim=64),
        "BiLSTM + Temporal Attention": BiLSTMAttention(in_features=len(INFORMATIVE_SENSORS), hidden_dim=64)
    }

    for name, net in neural_architectures.items():
        t0 = time.time()
        res = train_pytorch_model(
            model=net,
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

        if res["attn_weights"] is not None:
            saved_attention = res["attn_weights"]

        print(f"  -> {name:<27} | RMSE: {m['rmse']:5.2f} cy | NASA Score S: {m['nasa_score']:8.1f} | Time: {elapsed:4.1f}s")

    print("\n" + "=" * 80)
    print("  NASA C-MAPSS FD001 PROGNOSTIC BENCHMARK TABLE")
    print("=" * 80)
    table_str = format_benchmark_table(all_metrics)
    print(table_str)

    # 4. Save Flagship Attention Checkpoint
    best_arch_name = "BiLSTM + Temporal Attention"
    best_model_obj = trained_models[best_arch_name]
    best_m = [m for m in all_metrics if m["model_name"] == best_arch_name][0]

    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = PROJECT_ROOT / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    torch.save(best_model_obj.state_dict(), results_dir / "best_model.pt")
    torch.save(best_model_obj.state_dict(), checkpoints_dir / "best_model.pt")

    meta = {
        "best_model_name": best_arch_name,
        "best_rmse": best_m["rmse"],
        "nasa_score": best_m["nasa_score"],
        "sequence_length": sequence_length,
        "max_rul": max_rul,
        "sensors": INFORMATIVE_SENSORS,
        "feature_means": data_dict["feature_means"],
        "feature_stds": data_dict["feature_stds"],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(results_dir / "norm_params.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    with open(checkpoints_dir / "norm_params.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[Step 4/5] Checkpoint saved -> {results_dir / 'best_model.pt'}")

    # 5. Visualizations
    print(f"\n[Step 5/5] Exporting 300 DPI Publication Visualizations in {figures_dir}...")
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_sensor_degradation_trajectories(train_df, sample_units=[1, 2, 5, 8], save_path=figures_dir / "degradation_trajectories.png")
    plot_rul_predictions(y_test, test_predictions[best_arch_name], model_name=best_arch_name, metrics=best_m, save_path=figures_dir / "rul_predictions_vs_actual.png")

    # Single engine continuous lifecycle
    engine_1_data = train_df[train_df["unit_nr"] == 1]
    e1_norm = engine_1_data.copy()
    for col in INFORMATIVE_SENSORS:
        e1_norm[col] = (e1_norm[col] - data_dict["feature_means"][col]) / data_dict["feature_stds"][col]
    e1_feats = e1_norm[INFORMATIVE_SENSORS].values
    e1_windows = []
    e1_cycles = []
    e1_actual_ruls = []
    tot_e1 = len(engine_1_data)
    for i in range(sequence_length, tot_e1 + 1):
        e1_windows.append(e1_feats[i - sequence_length:i])
        e1_cycles.append(i)
        e1_actual_ruls.append(min(tot_e1 - i, max_rul))

    with torch.no_grad():
        e1_t = torch.tensor(np.array(e1_windows, dtype=np.float32)).to(device)
        e1_preds = best_model_obj(e1_t).cpu().numpy().ravel()

    plot_single_engine_lifecycle(np.array(e1_cycles), np.array(e1_actual_ruls), e1_preds, engine_id=1, save_path=figures_dir / "single_engine_degradation_profile.png")
    plot_loss_convergence(loss_histories, save_path=figures_dir / "training_loss_convergence.png")
    plot_architecture_benchmark_comparison([m for m in all_metrics if "Random Forest" not in m["model_name"]], save_path=figures_dir / "model_benchmark_comparison.png")

    # Attention Heatmap
    if saved_attention is not None:
        plot_attention_heatmap(saved_attention[:8], engine_ids=list(range(1, 9)), save_path=figures_dir / "attention_weights_heatmap.png")

    print("\n" + "=" * 80)
    print(f"  TRAINING COMPLETED: {best_arch_name} (RMSE = {best_m['rmse']:.2f} cycles, Score S = {best_m['nasa_score']:.1f})")
    print("=" * 80)

    return meta


def main():
    parser = argparse.ArgumentParser(description="Turbomachinery Prognostics with BiLSTM + Attention")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--window-size", "--seq_len", dest="window_size", type=int, default=30, help="Sliding window size")
    parser.add_argument("--lr", type=float, default=0.003, help="Learning rate")
    parser.add_argument("--max-rul", "--max_rul", dest="max_rul", type=float, default=125.0, help="Piecewise RUL clipping")

    args = parser.parse_args()
    run_pipeline(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        sequence_length=args.window_size,
        max_rul=args.max_rul
    )


if __name__ == "__main__":
    main()
