#!/usr/bin/env python3
"""
Cross-city generalization evaluation for TerraLens.

Works with the current processed datasets:
  data/processed/{city}_dataset.parquet

City keys expected:
  nyc, chicago, san_francisco

Outputs:
  outputs/results/cross_city_<train>_to_<test>_<model>.json
  outputs/results/cross_city_summary.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.model_zoo import prepare_xgboost_runtime  # noqa: E402 - script bootstrap

try:
    from scipy.stats import spearmanr
except Exception:
    spearmanr = None

# Optional: XGBoost (preferred). If unavailable, we'll fall back to sklearn.
try:
    prepare_xgboost_runtime()
    from xgboost import XGBRegressor

    HAS_XGB = True
except Exception:
    HAS_XGB = False
    XGBRegressor = None  # type: ignore

CITY_CHOICES = ["nyc", "chicago", "san_francisco"]


@dataclass
class EvalResult:
    train_cities: List[str]
    test_city: str
    model: str
    log_target: bool
    n_train: int
    n_test: int
    n_features: int
    feature_cols: List[str]
    mae: float
    rmse: float
    r2: float
    spearman: float | None


def _ensure_exists(p: Path) -> None:
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")


def load_city_df(city: str, data_dir: Path) -> pd.DataFrame:
    p = data_dir / f"{city}_dataset.parquet"
    _ensure_exists(p)
    df = pd.read_parquet(p)

    # Basic sanity
    required = {"tract_id", "median_income"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{p} missing required columns: {sorted(missing)}")

    return df


def infer_feature_cols(df: pd.DataFrame) -> List[str]:
    """
    Select numeric OSM features, excluding labels and metadata.

    Strategy:
      - exclude known metadata columns
      - keep numeric columns only
      - drop target
    """
    from scripts.cache_predictions import is_complaint_feature

    exclude = {
        "city",
        "tract_id",
        "median_income",
        "median_rent",
        "median_home_value",
        "poverty_rate",
        "unemployment_rate",
        "bachelors_plus_rate",
        "lat",
        "lon",
        "image_paths",
        "geometry",
        "NAME",
        "GEOID",
    }

    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if is_complaint_feature(c):
            continue
        # numeric only
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)

    if not cols:
        raise ValueError(
            "No numeric OSM feature columns remain after excluding labels and metadata."
        )

    return cols


def shared_feature_cols(dfs: List[pd.DataFrame]) -> List[str]:
    """
    Compute intersection of inferred feature columns across all dfs,
    preserving a stable order.
    """
    inferred = [set(infer_feature_cols(df)) for df in dfs]
    common = set.intersection(*inferred)

    if not common:
        # Show helpful debugging info
        per = [sorted(list(s))[:30] for s in inferred]
        raise ValueError(
            "No shared feature columns across datasets.\n"
            f"Per-dataset feature candidates (first 30 each): {per}"
        )

    # Stable order: use first df's inferred order, filter by common
    first_order = infer_feature_cols(dfs[0])
    return [c for c in first_order if c in common]


def make_xy(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    target = pd.to_numeric(df["median_income"], errors="coerce")
    valid = np.isfinite(target) & (target >= 0)
    X = df.loc[valid, feature_cols].copy()
    # Fill any missing numerics safely
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = target.loc[valid].to_numpy(dtype=float)
    return X.values, y


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float | None]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))

    rho: float | None = None
    if spearmanr is not None:
        try:
            rho = float(spearmanr(y_true, y_pred).correlation)
        except Exception:
            rho = None

    return {"mae": mae, "rmse": rmse, "r2": r2, "spearman": rho}


def train_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    model_name: str,
    seed: int,
) -> np.ndarray:
    if model_name == "ridge":
        model = Pipeline(
            [("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0, random_state=seed))]
        )
        model.fit(X_train, y_train)
        return model.predict(X_test)

    if model_name == "xgboost":
        if HAS_XGB:
            model = XGBRegressor(
                n_estimators=600,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_alpha=0.0,
                reg_lambda=1.0,
                random_state=seed,
                n_jobs=4,
            )
            model.fit(X_train, y_train)
            return model.predict(X_test)

        # Use scikit-learn's tree model when XGBoost is unavailable.
        model = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.05, random_state=seed)
        model.fit(X_train, y_train)
        return model.predict(X_test)

    raise ValueError(f"Unknown model: {model_name}")


def run_one(
    train_cities: List[str],
    test_city: str,
    data_dir: Path,
    model: str,
    log_target: bool,
    seed: int,
    test_size: float,
    in_city_split: bool,
) -> EvalResult:
    if not in_city_split and test_city in train_cities:
        raise ValueError("The test city must be excluded from cross-city training cities")
    # Load dfs
    train_dfs = [load_city_df(c, data_dir) for c in train_cities]
    test_df = load_city_df(test_city, data_dir)

    # For in-city baseline we want an in-city split of the same dataset
    if in_city_split:
        base_df = train_dfs[0]
        cols = infer_feature_cols(base_df)
        X, y = make_xy(base_df, cols)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed)
        feature_cols = cols
    else:
        # Cross-city: shared feature intersection across all involved datasets
        feature_cols = shared_feature_cols(train_dfs + [test_df])

        # Combine train
        X_tr_list, y_tr_list = [], []
        for df in train_dfs:
            Xc, yc = make_xy(df, feature_cols)
            X_tr_list.append(Xc)
            y_tr_list.append(yc)

        X_tr = np.vstack(X_tr_list)
        y_tr = np.concatenate(y_tr_list)

        X_te, y_te = make_xy(test_df, feature_cols)

    # Target transform
    if log_target:
        y_tr_fit = np.log1p(y_tr)
    else:
        y_tr_fit = y_tr

    # Train/predict
    y_pred_fit = train_predict(X_tr, y_tr_fit, X_te, model_name=model, seed=seed)

    # Inverse transform
    if log_target:
        # Safety: clip log-predictions to training log-income range
        # to prevent expm1() from exploding on cross-city domain shift.
        y_lo = float(np.min(y_tr_fit))
        y_hi = float(np.max(y_tr_fit))
        y_pred_fit = np.clip(y_pred_fit, y_lo, y_hi)
        y_pred = np.expm1(y_pred_fit)
    else:
        y_pred = y_pred_fit

    # Metrics on original scale
    m = compute_metrics(y_te, y_pred)

    return EvalResult(
        train_cities=train_cities,
        test_city=test_city,
        model=("hist_gb" if model == "xgboost" and not HAS_XGB else model),
        log_target=log_target,
        n_train=int(len(y_tr)),
        n_test=int(len(y_te)),
        n_features=int(len(feature_cols)),
        feature_cols=feature_cols,
        mae=float(m["mae"]),
        rmse=float(m["rmse"]),
        r2=float(m["r2"]),
        spearman=(None if m["spearman"] is None else float(m["spearman"])),
    )


def save_result(res: EvalResult, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_tag = "+".join(res.train_cities)
    transform = "log" if res.log_target else "raw"
    fname = f"cross_city_{train_tag}_to_{res.test_city}_{res.model}_{transform}.json"
    p = out_dir / fname
    with p.open("w") as f:
        json.dump(asdict(res), f, indent=2)
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", choices=CITY_CHOICES, help="Training cities")
    ap.add_argument("--test", choices=CITY_CHOICES, help="Test city")
    ap.add_argument("--model", choices=["ridge", "xgboost"], default="xgboost")
    ap.add_argument("--log-target", action="store_true", default=True)
    ap.add_argument("--no-log-target", dest="log_target", action="store_false")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--full-matrix", action="store_true")
    ap.add_argument("--data-dir", type=str, default="data/processed")
    ap.add_argument("--out-dir", type=str, default="outputs/results")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    results: List[EvalResult] = []

    # 1) Full matrix mode
    if args.full_matrix:
        cities = CITY_CHOICES

        # In-city baselines
        for c in cities:
            res = run_one(
                train_cities=[c],
                test_city=c,
                data_dir=data_dir,
                model=args.model,
                log_target=args.log_target,
                seed=args.seed,
                test_size=args.test_size,
                in_city_split=True,
            )
            results.append(res)
            save_result(res, out_dir)

        # Cross-city single train
        for tr in cities:
            for te in cities:
                if tr == te:
                    continue
                res = run_one(
                    train_cities=[tr],
                    test_city=te,
                    data_dir=data_dir,
                    model=args.model,
                    log_target=args.log_target,
                    seed=args.seed,
                    test_size=args.test_size,
                    in_city_split=False,
                )
                results.append(res)
                save_result(res, out_dir)

        # Multi-train (train on 2, test on 1)
        for te in cities:
            tr = [c for c in cities if c != te]
            res = run_one(
                train_cities=tr,
                test_city=te,
                data_dir=data_dir,
                model=args.model,
                log_target=args.log_target,
                seed=args.seed,
                test_size=args.test_size,
                in_city_split=False,
            )
            results.append(res)
            save_result(res, out_dir)

    else:
        if not args.train or not args.test:
            ap.error("Provide --train and --test, or use --full-matrix")

        # In-city if train==test and single city
        in_city = len(args.train) == 1 and args.train[0] == args.test

        res = run_one(
            train_cities=args.train,
            test_city=args.test,
            data_dir=data_dir,
            model=args.model,
            log_target=args.log_target,
            seed=args.seed,
            test_size=args.test_size,
            in_city_split=in_city,
        )
        results.append(res)
        p = save_result(res, out_dir)
        print(f"✅ Saved: {p}")

    # Summary table
    rows = []
    for r in results:
        rows.append(
            {
                "train": "+".join(r.train_cities),
                "test": r.test_city,
                "model": r.model,
                "log_target": r.log_target,
                "n_train": r.n_train,
                "n_test": r.n_test,
                "n_feat": r.n_features,
                "MAE": r.mae,
                "RMSE": r.rmse,
                "R2": r.r2,
                "Spearman": (np.nan if r.spearman is None else r.spearman),
            }
        )

    if rows:
        df = pd.DataFrame(rows).sort_values(["model", "train", "test"])
        out_csv = out_dir / "cross_city_summary.csv"
        if out_csv.exists():
            previous = pd.read_csv(out_csv)
            df = pd.concat([previous, df], ignore_index=True)
        df = df.drop_duplicates(["train", "test", "model", "log_target"], keep="last")
        df = df.sort_values(["model", "train", "test", "log_target"])
        df.to_csv(out_csv, index=False)

        print("\n" + "=" * 80)
        print("CROSS-CITY SUMMARY (saved to outputs/results/cross_city_summary.csv)")
        print("=" * 80)
        with pd.option_context("display.max_rows", 200, "display.max_columns", 50):
            print(df.to_string(index=False))
        print()

    if args.model == "xgboost" and not HAS_XGB:
        print("⚠️  xgboost not available; used HistGradientBoostingRegressor fallback.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
