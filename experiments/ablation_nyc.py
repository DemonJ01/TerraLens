#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.model_zoo import prepare_xgboost_runtime  # noqa: E402 - script bootstrap

try:
    prepare_xgboost_runtime()
    from xgboost import XGBRegressor

    HAS_XGB = True
except Exception:
    HAS_XGB = False
    from sklearn.ensemble import HistGradientBoostingRegressor

from terralens.evaluation.fairness import (  # noqa: E402 - script bootstrap
    save_fairness_report,
)
from terralens.evaluation.metrics import (  # noqa: E402 - script bootstrap
    regression_metrics,
)


def infer_cols(df, prefix_list):
    cols = []
    for c in df.columns:
        if any(c.startswith(p) for p in prefix_list):
            if pd.api.types.is_numeric_dtype(df[c]):
                cols.append(c)
    return cols


def fit_predict(X_train, y_train, X_test, model):
    model.fit(X_train, y_train)
    return model.predict(X_test)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-size", type=float, default=0.2)
    args = ap.parse_args()

    df_osm = pd.read_parquet("data/processed/nyc_dataset.parquet")
    enriched_path = Path("data/processed/nyc_dataset_311.parquet")
    df_311 = pd.read_parquet(enriched_path) if enriched_path.exists() else None

    configs = [
        ("osm_only", ["poi_", "road_"]),  # your NYC OSM features are poi_* and road_*
        ("osm_311", ["poi_", "road_", "complaint_", "complaints_"]),
    ]

    models = ["ridge", "xgboost"]
    logs = [True, False]

    rows = []
    out_dir = Path("outputs/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    for feature_set, prefixes in configs:
        df = df_311 if feature_set == "osm_311" else df_osm
        if df is None:
            print("Skipping OSM + 311 ablation: NYC 311 dataset is not available.")
            continue
        df = df.copy()
        income = pd.to_numeric(df["median_income"], errors="coerce")
        df = df.loc[np.isfinite(income) & (income >= 0)].copy()
        feat_cols = infer_cols(df, prefixes)

        if not feat_cols:
            print(f"⚠️ No features found for {feature_set} with prefixes={prefixes}")
            continue

        X = df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
        y = df["median_income"].astype(float).values

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=args.test_size, random_state=args.seed
        )

        for log_target in logs:
            y_tr_fit = np.log1p(y_tr) if log_target else y_tr

            for m in models:
                if m == "ridge":
                    model = make_pipeline(
                        StandardScaler(), Ridge(alpha=1.0, random_state=args.seed)
                    )
                    model_name = "ridge"
                else:
                    if HAS_XGB:
                        model = XGBRegressor(
                            n_estimators=600,
                            max_depth=6,
                            learning_rate=0.05,
                            subsample=0.9,
                            colsample_bytree=0.9,
                            random_state=args.seed,
                            n_jobs=4,
                        )
                        model_name = "xgboost"
                    else:
                        model = HistGradientBoostingRegressor(
                            max_depth=6, learning_rate=0.05, random_state=args.seed
                        )
                        model_name = "hgb_fallback"

                y_pred_fit = fit_predict(X_tr, y_tr_fit, X_te, model)
                if log_target:
                    y_pred_fit = np.clip(y_pred_fit, np.min(y_tr_fit), np.max(y_tr_fit))
                y_pred = np.expm1(y_pred_fit) if log_target else y_pred_fit

                met = regression_metrics(y_te, y_pred)

                rows.append(
                    {
                        "feature_set": feature_set,
                        "log_target": log_target,
                        "model": model_name,
                        "n_features": len(feat_cols),
                        "mae": met["mae"],
                        "rmse": met["rmse"],
                        "r2": met["r2"],
                        "spearman": met["spearman"],
                    }
                )

                # fairness report only for the best-ish configuration later,
                # but we’ll save it for every run for now (still small)
                tag = (
                    f"fairness_nyc_{feature_set}_{model_name}_{'log' if log_target else 'raw'}.json"
                )
                save_fairness_report(
                    y_true=y_te,
                    y_pred=y_pred,
                    out_json=out_dir / tag,
                    metadata={
                        "city": "nyc",
                        "feature_set": feature_set,
                        "model": model_name,
                        "log_target": log_target,
                        "n_features": len(feat_cols),
                    },
                )

                print(
                    f"NYC | {feature_set} | {model_name} | log={log_target} -> "
                    f"R2={met['r2']:.3f} MAE=${met['mae']:,.0f}"
                )

    if not rows:
        raise SystemExit("No ablation results: the dataset has no matching numeric OSM features.")
    res = pd.DataFrame(rows).sort_values(["feature_set", "model", "log_target"])
    out_csv = out_dir / "ablation_nyc.csv"
    res.to_csv(out_csv, index=False)
    print(f"\n✅ Saved NYC ablation summary: {out_csv}")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
