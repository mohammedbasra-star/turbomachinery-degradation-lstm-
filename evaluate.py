"""
Inference and Prognostic Evaluation CLI for Turbomachinery Fleet.

Allows maintenance engineers and operators to:
1. Evaluate the saved model on the complete 100-engine test fleet.
2. Query the predicted Remaining Useful Life (RUL) for a specific test engine unit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import load_cmapss_data, prepare_sliding_windows, INFORMATIVE_SENSORS
from models.architectures import build_model
from utils.metrics import calculate_rul_metrics

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def evaluate_fleet(checkpoints_dir: Path = PROJECT_ROOT / "checkpoints") -> None:
    ckpt_path = checkpoints_dir / "best_model.pt"
    meta_path = checkpoints_dir / "norm_params.json"

    if not ckpt_path.exists() or not meta_path.exists():
        print(f"[Error] Checkpoint not found at {ckpt_path}. Run `python train.py` first.")
        sys.exit(1)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    model_name = meta["best_model_name"]
    seq_len = meta["sequence_length"]
    max_rul = meta["max_rul"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name, sequence_length=seq_len, n_features=len(INFORMATIVE_SENSORS))
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()

    train_df, test_df, true_rul = load_cmapss_data(PROJECT_ROOT / "data")
    data_dict = prepare_sliding_windows(train_df, test_df, true_rul, sequence_length=seq_len, max_rul=max_rul)

    X_test_tensor = torch.tensor(data_dict["X_test"], dtype=torch.float32).to(device)
    y_test = data_dict["y_test"]

    with torch.no_grad():
        preds = model(X_test_tensor).cpu().numpy().ravel()
        preds = np.maximum(0.0, preds)

    metrics = calculate_rul_metrics(y_test, preds, model_name=model_name)

    print("\n" + "=" * 65)
    print("  NASA C-MAPSS FD001 TEST FLEET PROGNOSTIC VERIFICATION")
    print("=" * 65)
    print(f"  Model Architecture : {model_name}")
    print(f"  Trained Timestamp  : {meta['timestamp']}")
    print(f"  Evaluated Engines  : {metrics['n_engines']}")
    print("-" * 65)
    print(f"  Test RMSE          : {metrics['rmse']:.2f} Flight Cycles")
    print(f"  NASA Score S       : {metrics['nasa_score']:.1f}")
    print(f"  Mean Absolute Error: {metrics['mae']:.2f} Flight Cycles")
    print(f"  Maximum Error      : {metrics['max_err']:.2f} Flight Cycles")
    print("=" * 65)


def evaluate_single_unit(unit_id: int, checkpoints_dir: Path = PROJECT_ROOT / "checkpoints") -> None:
    ckpt_path = checkpoints_dir / "best_model.pt"
    meta_path = checkpoints_dir / "norm_params.json"

    if not ckpt_path.exists() or not meta_path.exists():
        print(f"[Error] Checkpoint not found at {ckpt_path}. Run `python train.py` first.")
        sys.exit(1)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    model_name = meta["best_model_name"]
    seq_len = meta["sequence_length"]
    max_rul = meta["max_rul"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name, sequence_length=seq_len, n_features=len(INFORMATIVE_SENSORS))
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()

    train_df, test_df, true_rul = load_cmapss_data(PROJECT_ROOT / "data")
    unit_data = test_df[test_df["unit_nr"] == unit_id]

    if len(unit_data) == 0:
        print(f"[Error] Unit #{unit_id} not found in test fleet (1 to 100).")
        return

    # Standardize
    unit_norm = unit_data.copy()
    for col in INFORMATIVE_SENSORS:
        unit_norm[col] = (unit_norm[col] - meta["feature_means"][col]) / meta["feature_stds"][col]

    feats = unit_norm[INFORMATIVE_SENSORS].values
    if len(feats) >= seq_len:
        window = feats[-seq_len:]
    else:
        pad = np.repeat(feats[:1], seq_len - len(feats), axis=0)
        window = np.vstack([pad, feats])

    x_tensor = torch.tensor(window[np.newaxis, ...], dtype=torch.float32).to(device)
    with torch.no_grad():
        pred_rul = float(model(x_tensor).cpu().numpy()[0, 0])
        pred_rul = max(0.0, pred_rul)

    actual_rul = float(min(true_rul[unit_id - 1], max_rul))
    err = pred_rul - actual_rul

    print("=" * 65)
    print(f"  TURBOMACHINERY PROGNOSTIC HEALTH INSPECTION: UNIT #{unit_id}")
    print("=" * 65)
    print(f"  Observed Cycles to Date : {len(unit_data)} Flight Cycles")
    print(f"  Ground Truth Target RUL : {actual_rul:.1f} Cycles")
    print(f"  Deep Model Predicted RUL: {pred_rul:.1f} Cycles")
    print(f"  Prognostic Error        : {err:+.1f} Cycles")
    print("-" * 65)
    if pred_rul <= 20:
        print("  [ALERT: CRITICAL] Engine near failure! Immediate overhaul required.")
    elif pred_rul <= 45:
        print("  [WARNING: MEDIUM] Moderate wear detected. Schedule inspection.")
    else:
        print("  [HEALTHY] Component wear within safe operational limits.")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Turbomachinery Prognostic Model")
    parser.add_argument("--unit", type=int, default=None, help="Specific engine unit ID to inspect (1 to 100)")
    parser.add_argument("--fleet", action="store_true", help="Run full fleet evaluation benchmark")

    args = parser.parse_args()

    if args.unit is not None:
        evaluate_single_unit(args.unit)
    else:
        evaluate_fleet()


if __name__ == "__main__":
    main()
