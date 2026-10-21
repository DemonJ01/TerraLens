#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run(cmd):
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


def main():
    (PROJECT_ROOT / "outputs/results").mkdir(parents=True, exist_ok=True)
    missing = [
        str(PROJECT_ROOT / f"data/processed/{city}_dataset.parquet")
        for city in ("nyc", "chicago", "san_francisco")
        if not (PROJECT_ROOT / f"data/processed/{city}_dataset.parquet").exists()
    ]
    if missing:
        raise SystemExit(
            "Missing processed datasets. Build real city data first:\n" + "\n".join(missing)
        )

    # Cross-city OSM baseline (already implemented in cross_city_eval.py)
    run([sys.executable, "experiments/cross_city_eval.py", "--full-matrix", "--model", "ridge"])
    run([sys.executable, "experiments/cross_city_eval.py", "--full-matrix", "--model", "xgboost"])

    # NYC ablation (OSM vs OSM+311, ridge vs xgb, raw vs log, fairness reports)
    run([sys.executable, "experiments/ablation_nyc.py"])

    print("\n✅ All experiments finished. Check outputs/results/")


if __name__ == "__main__":
    main()
