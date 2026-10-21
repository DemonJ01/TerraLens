#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd

PRED_DIR = Path("outputs/preds")
MANIFEST = PRED_DIR / "manifest.csv"
OUT = Path("outputs/results/model_leaderboard.csv")


def main() -> int:
    if not MANIFEST.exists():
        raise SystemExit("manifest.csv not found. Run scripts/cache_predictions.py first.")

    m = pd.read_csv(MANIFEST)

    # If metrics exist, use them directly; otherwise compute from parquets
    need_compute = not set(["r2", "mae", "rmse"]).issubset(m.columns)

    if need_compute:
        rows = []
        for _, r in m.iterrows():
            p = Path(r["path"])
            if not p.exists():
                continue
            df = pd.read_parquet(p)
            if "error" not in df.columns and "y_true" in df.columns and "y_pred" in df.columns:
                df["error"] = df["y_pred"] - df["y_true"]
            if "error" not in df.columns:
                continue
            e = df["error"].astype(float)
            mae = float(e.abs().mean())
            rmse = float((e**2).mean() ** 0.5)
            y = df["y_true"].astype(float)
            p_ = df["y_pred"].astype(float)
            y_mean = float(y.mean())
            ss_res = float(((y - p_) ** 2).sum())
            ss_tot = float(((y - y_mean) ** 2).sum())
            r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")
            rows.append({**r.to_dict(), "mae": mae, "rmse": rmse, "r2": r2})
        m = pd.DataFrame(rows)

    # Pick best per city/feature_set/target by highest R2
    best = (
        m.sort_values(["city", "feature_set", "target", "r2"], ascending=[True, True, True, False])
        .groupby(["city", "feature_set", "target"], as_index=False)
        .head(1)
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    best.to_csv(OUT, index=False)
    print(f"✅ Wrote leaderboard: {OUT} (rows={len(best)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
