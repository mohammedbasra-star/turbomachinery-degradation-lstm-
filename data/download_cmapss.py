"""
Automated downloader and fallback synthetic generator for NASA C-MAPSS Turbofan Engine Dataset.

Schema:
- 26 columns: unit_nr, time_cycles, 3 operational settings, 21 sensor channels.
- Datasets: FD001 (Sea level, HPC fault), FD002 (Six operating conditions).
"""

from __future__ import annotations

import os
import sys
import zipfile
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CMAPSS_ZIP_URL = "https://data.nasa.gov/download/5v84-7mgq/application%2Fzip"
DATA_DIR = Path(__file__).resolve().parent

INDEX_COLS = ["unit_nr", "time_cycles"]
SETTING_COLS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLS = [f"s_{i}" for i in range(1, 22)]
ALL_COLS = INDEX_COLS + SETTING_COLS + SENSOR_COLS


def download_or_generate_cmapss(data_dir: Path = DATA_DIR) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    train_file = data_dir / "train_FD001.txt"
    test_file = data_dir / "test_FD001.txt"
    rul_file = data_dir / "RUL_FD001.txt"

    if train_file.exists() and test_file.exists() and rul_file.exists():
        print(f"[Dataset] NASA C-MAPSS FD001 files already verified in {data_dir}")
        return

    # Try downloading official archive
    try:
        print("[Dataset] Attempting download from NASA data repository...")
        zip_path = data_dir / "CMAPSSData.zip"
        urllib.request.urlretrieve(CMAPSS_ZIP_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(data_dir)
        if zip_path.exists():
            zip_path.unlink()
        print("[Dataset] Download and extraction complete.")
        return
    except Exception as e:
        print(f"[Dataset] Online download not reachable ({e}). Using high-fidelity thermodynamic generator...")

    # High-fidelity synthesis fallback
    from data.dataset import generate_synthetic_cmapss_fd001
    generate_synthetic_cmapss_fd001(data_dir)


if __name__ == "__main__":
    download_or_generate_cmapss()
