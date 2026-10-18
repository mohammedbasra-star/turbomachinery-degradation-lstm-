"""
Prognostic Evaluation and Single-Engine Health Inspection CLI.
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
from models.lstm_attention import BiLSTMAttention
from utils.scoring import compute_evaluation_metrics

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def evaluate_checkpoint(checkpoint_path: Path) -> None:
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        # Check fallback
        if (PROJECT_ROOT / "checkpoints" / "best_model.pt").exists():
            ckpt_path = PROJECT_ROOT / "checkpoints" / "best_model.pt"
        else:
            print(f"[Error] Checkpoint not found at {checkpoint_path}. Run `python train.py` first.")
            sys.exit(1)

    meta_file = ckpt_path.parent / "norm_params.json"
    if not meta_file.exists():
        meta_file = PROJECT_ROOT / "checkpoints" / "norm_params.json"

    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)

    seq_len = meta.get("sequence_length", 30)
    max_rul = meta.get("max_rul", 125.0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTMAttention(in_features=len(INFORMATIVE_SENSORS), hidden_dim=64)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()

    train_df, test_df, true_rul = load_cmapss_data(PROJECT_ROOT / "data")
    data_dict = prepare_sliding_windows(train_df, test_df, true_rul, sequence_length=seq_len, max_rul=max_rul)

    X_test_tensor = torch.tensor(data_dict["X_test"], dtype=torch.float32).to(device)
    y_test = data_dict["y_test"]

    with torch.no_grad():
        preds, attns = model(X_test_tensor, return_attention=True)
        preds = np.maximum(0.0, preds.cpu().numpy().ravel())

    metrics = compute_evaluation_metrics(y_test, preds, model_name=meta.get("best_model_name", "BiLSTM+Attention"))

    print("\n" + "=" * 65)
    print("  NASA C-MAPSS FD001 PROGNOSTIC VERIFICATION REPORT")
    print("=" * 65)
    print(f"  Model Architecture : {metrics['model_name']}")
    print(f"  Evaluated Engines  : {metrics['n_samples']} Fleet Units")
    print("-" * 65)
    print(f"  Test RMSE          : {metrics['rmse']:.2f} Flight Cycles")
    print(f"  NASA Score S       : {metrics['nasa_score']:.1f}")
    print(f"  Mean Absolute Error: {metrics['mae']:.2f} Flight Cycles")
    print(f"  Maximum Error      : {metrics['max_err']:.2f} Flight Cycles")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Evaluate BiLSTM+Attention Prognostic Checkpoint")
    parser.add_argument("--checkpoint", type=str, default="results/best_model.pt", help="Path to checkpoint .pt")
    parser.add_argument("--unit", type=int, default=None, help="Specific engine unit ID to inspect")

    args = parser.parse_args()
    evaluate_checkpoint(Path(args.checkpoint))


if __name__ == "__main__":
    main()
