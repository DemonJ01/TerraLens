#!/usr/bin/env python3
"""
Make NYC income prediction figures.

Outputs into: outputs/figures/
- nyc_scatter.png
- nyc_residuals.png
- nyc_feature_importance.png
- nyc_pred_map.png (optional, if geopandas + tract gpkg exist)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
FIG_DIR = ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DATASET = ROOT / "data" / "processed" / "nyc_dataset.parquet"
TRACTS_GPKG = ROOT / "data" / "raw" / "census" / "nyc_tracts_2021.gpkg"

TARGET_COL_CANDIDATES = ["income", "median_income", "median_household_income", "acs_income"]


def choose_target(df: pd.DataFrame) -> str:
    for c in TARGET_COL_CANDIDATES:
        if c in df.columns:
            return c
    # fallback: choose numeric col containing "income"
    for c in df.columns:
        if "income" in c.lower():
            return c
    raise RuntimeError("Could not find an income target column in NYC dataset.")


def main() -> int:
    if not DATASET.exists():
        raise SystemExit(f"❌ Missing dataset: {DATASET}. Run dataset build first.")

    df = pd.read_parquet(DATASET)
    target = choose_target(df)

    from sklearn.model_selection import train_test_split

    from scripts.cache_predictions import select_features

    target_values = pd.to_numeric(df[target], errors="coerce")
    df = df.loc[np.isfinite(target_values) & (target_values >= 0)].copy()
    feat_cols = [c for c in select_features(df, "osm_only") if c != target]
    X = df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy()
    y_all = df[target].to_numpy(dtype=float)
    train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42)
    y = y_all[test_idx]

    # Train a simple model (xgboost if installed, else fallback)
    model = None
    model_name = "xgboost"
    try:
        from scripts.model_zoo import prepare_xgboost_runtime

        prepare_xgboost_runtime()
        import xgboost as xgb

        model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=4,
        )
        model.fit(X[train_idx], y_all[train_idx])
        yhat = model.predict(X[test_idx])
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor

        model_name = "histgbrt"
        model = HistGradientBoostingRegressor(random_state=42)
        model.fit(X[train_idx], y_all[train_idx])
        yhat = model.predict(X[test_idx])

    resid = yhat - y

    # Scatter
    plt.figure()
    plt.scatter(y, yhat, s=8, alpha=0.5)
    lo = float(np.nanmin([y.min(), yhat.min()]))
    hi = float(np.nanmax([y.max(), yhat.max()]))
    plt.plot([lo, hi], [lo, hi])
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(f"NYC: Held-out True vs Predicted ({model_name})")
    out1 = FIG_DIR / "nyc_scatter.png"
    plt.savefig(out1, dpi=180, bbox_inches="tight")
    plt.close()

    # Residuals
    plt.figure()
    plt.scatter(y, resid, s=8, alpha=0.5)
    plt.axhline(0)
    plt.xlabel("True")
    plt.ylabel("Residual (pred - true)")
    plt.title(f"NYC: Residuals ({model_name})")
    out2 = FIG_DIR / "nyc_residuals.png"
    plt.savefig(out2, dpi=180, bbox_inches="tight")
    plt.close()

    # Feature importance (if available)
    out3 = FIG_DIR / "nyc_feature_importance.png"
    try:
        if model_name == "xgboost":
            importances = model.feature_importances_
        else:
            # fallback: permutation importance would be slower; skip
            importances = None

        if importances is not None:
            idx = np.argsort(importances)[::-1][:20]
            top_feats = [feat_cols[i] for i in idx]
            top_vals = importances[idx]

            plt.figure()
            plt.barh(list(reversed(top_feats)), list(reversed(top_vals)))
            plt.xlabel("Importance")
            plt.title("NYC: Top-20 Feature Importances")
            plt.savefig(out3, dpi=180, bbox_inches="tight")
            plt.close()
        else:
            # Record why feature importance is unavailable for this model.
            (FIG_DIR / "nyc_feature_importance_note.txt").write_text(
                "Feature importance not available for fallback model.\n", encoding="utf-8"
            )
    except Exception:
        (FIG_DIR / "nyc_feature_importance_note.txt").write_text(
            "Feature importance plot failed.\n", encoding="utf-8"
        )

    # Optional choropleth if geopandas + tracts exist
    try:
        import geopandas as gpd

        if TRACTS_GPKG.exists() and "tract_id" in df.columns:
            gdf = gpd.read_file(TRACTS_GPKG)
            # try common tract id column names
            tract_cols = [
                c for c in gdf.columns if c.lower() in ["tract_id", "geoid", "geoid10", "geoid20"]
            ]
            if tract_cols:
                tc = tract_cols[0]
                # normalize as string
                gdf[tc] = gdf[tc].astype(str)
                df2 = df.iloc[test_idx].copy()
                df2["tract_id"] = df2["tract_id"].astype(str)
                df2["pred_income"] = yhat
                merged = gdf.merge(
                    df2[["tract_id", "pred_income"]], left_on=tc, right_on="tract_id", how="inner"
                )

                ax = merged.plot(column="pred_income", legend=True)
                ax.set_axis_off()
                ax.set_title("NYC: Predicted Income (held-out tracts)")
                out4 = FIG_DIR / "nyc_pred_map.png"
                plt.savefig(out4, dpi=180, bbox_inches="tight")
                plt.close()
    except Exception:
        pass

    print(f"✅ Wrote figures to: {FIG_DIR}")
    print("   - nyc_scatter.png")
    print("   - nyc_residuals.png")
    print("   - nyc_feature_importance.png (or note file)")
    print("   - nyc_pred_map.png (if geopandas + gpkg available)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
