#!/usr/bin/env python3
"""Measure generalization to held-out geographic blocks with nested validation.

This evaluation is independent of the dashboard's random tract holdout. Every
outer test block is excluded from preprocessing, target-transform selection and
hyperparameter selection. Inner GroupKFold splits use training blocks only.
The grid is defined from coordinates, never from socioeconomic labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.cache_predictions import RATE_TARGETS, TARGETS, load_dataset, select_features


def score_predictions(truth: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    truth, predictions = np.asarray(truth, float), np.asarray(predictions, float)
    if len(truth) < 2 or not np.isfinite(truth).all() or not np.isfinite(predictions).all():
        raise ValueError("Scoring requires at least two finite truth and prediction values.")
    return {
        "r2": float(r2_score(truth, predictions)),
        "mae": float(mean_absolute_error(truth, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(truth, predictions))),
    }


def tract_id_hash(tract_ids: pd.Series | list[str]) -> str:
    """An order-independent digest of the exact split membership."""
    return hashlib.sha256("\n".join(sorted(map(str, tract_ids))).encode("utf-8")).hexdigest()


def geographic_blocks(coordinates: pd.DataFrame, bins: int = 4) -> tuple[np.ndarray, dict]:
    """Assign tract centroids to a fixed, equally spaced longitude/latitude grid.

    Defining a grid with all available coordinates is a split-design operation;
    no target values or learned predictor parameters are used. GroupKFold keeps
    each occupied cell intact. Adjacent cells may occur in different folds, so
    this is not a spatially buffered test and does not establish cross-city skill.
    """
    if bins < 2:
        raise ValueError("Spatial grid needs at least two bins per axis.")
    values = coordinates.loc[:, ["lon", "lat"]].to_numpy(dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Spatial validation requires finite tract coordinates.")
    if (np.abs(values[:, 0]) > 180).any() or (np.abs(values[:, 1]) > 90).any():
        raise ValueError("Coordinates must be longitude/latitude in degrees.")
    lower, upper = values.min(axis=0), values.max(axis=0)
    widths = upper - lower
    scaled = np.divide(values - lower, widths, out=np.zeros_like(values), where=widths > 0)
    cells = np.minimum(np.floor(scaled * bins).astype(int), bins - 1)
    groups = cells[:, 0] + bins * cells[:, 1]
    counts = pd.Series(groups).value_counts().sort_index()
    design = {
        "kind": "fixed_equal_width_lon_lat_grid",
        "bins_per_axis": bins,
        "longitude_bounds": [float(lower[0]), float(upper[0])],
        "latitude_bounds": [float(lower[1]), float(upper[1])],
        "occupied_blocks": int(len(counts)),
        "block_counts": {str(k): int(v) for k, v in counts.items()},
        "uses_target_values": False,
        "buffer_distance": None,
    }
    return groups, design


def _valid_target(values: pd.Series, target: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return np.isfinite(numeric) & (
        numeric.between(0, 1) if target in RATE_TARGETS else numeric >= 0
    )


def evaluate_model(
    frame: pd.DataFrame,
    feature_names: list[str],
    target: str,
    model_name: str,
    *,
    seed: int = 42,
    spatial_bins: int = 4,
    outer_folds: int = 3,
    inner_folds: int = 3,
    n_jobs: int = 2,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run nested CV without exposing any outer test labels to selection."""
    from scripts.model_selection import select_and_fit_model

    forbidden = set(feature_names) & (set(TARGETS) | {"tract_id", "GEOID"})
    if forbidden:
        raise ValueError(f"Target labels and tract IDs cannot be predictors: {sorted(forbidden)}")
    if not feature_names or len(set(feature_names)) != len(feature_names):
        raise ValueError("Provide unique predictor feature names.")
    valid = _valid_target(frame[target], target)
    data = frame.loc[valid].reset_index(drop=True)
    if data.tract_id.astype(str).duplicated().any():
        raise ValueError("Tract IDs must be unique before spatial validation.")
    if len(data) < 2 * outer_folds:
        raise ValueError("Too few valid target rows for the requested outer folds.")
    features = data[feature_names].astype(float).replace([np.inf, -np.inf], np.nan)
    truth = data[target].astype(float)
    groups, design = geographic_blocks(data[["lat", "lon"]], spatial_bins)
    if len(np.unique(groups)) < outer_folds:
        raise ValueError("Too few occupied spatial blocks for the requested outer folds.")
    outer = GroupKFold(n_splits=outer_folds)
    predictions = np.full(len(data), np.nan)
    baseline = np.full(len(data), np.nan)
    fold_numbers = np.full(len(data), -1, dtype=int)
    folds = []

    for fold, (train_indices, test_indices) in enumerate(outer.split(features, truth, groups)):
        train_groups, test_groups = groups[train_indices], groups[test_indices]
        if set(train_groups) & set(test_groups):
            raise AssertionError("Outer spatial blocks overlap.")
        if len(np.unique(train_groups)) < inner_folds:
            raise ValueError("Outer training partition has too few blocks for inner GroupKFold.")
        if len(test_indices) < 2:
            raise ValueError("An outer fold has fewer than two target rows.")
        x_train = features.iloc[train_indices].reset_index(drop=True)
        y_train = truth.iloc[train_indices].reset_index(drop=True)
        inner = list(GroupKFold(n_splits=inner_folds).split(x_train, y_train, train_groups))
        for inner_train, inner_test in inner:
            if set(train_groups[inner_train]) & set(train_groups[inner_test]):
                raise AssertionError("Inner spatial blocks overlap.")
        estimator, selection = select_and_fit_model(
            model_name,
            x_train,
            y_train,
            target,
            seed=seed,
            cv_folds=inner_folds,
            n_jobs=n_jobs,
            cv_splits=inner,
        )
        test_predictions = np.asarray(estimator.predict(features.iloc[test_indices]), dtype=float)
        if not np.isfinite(test_predictions).all():
            raise ValueError("The fitted model produced nonfinite spatial predictions.")
        if target in RATE_TARGETS and ((test_predictions < 0) | (test_predictions > 1)).any():
            raise ValueError("The fitted model produced a rate outside [0, 1].")
        if target not in RATE_TARGETS and (test_predictions < 0).any():
            raise ValueError("The fitted model produced a negative monetary estimate.")
        train_mean = float(y_train.mean())
        fold_baseline = np.full(len(test_indices), train_mean)
        predictions[test_indices] = test_predictions
        baseline[test_indices] = fold_baseline
        fold_numbers[test_indices] = fold
        train_ids = data.iloc[train_indices].tract_id.astype(str)
        test_ids = data.iloc[test_indices].tract_id.astype(str)
        folds.append(
            {
                "fold": fold,
                "train_rows": len(train_indices),
                "test_rows": len(test_indices),
                "train_blocks": sorted(map(int, np.unique(train_groups))),
                "test_blocks": sorted(map(int, np.unique(test_groups))),
                "train_tract_ids_sha256": tract_id_hash(train_ids),
                "test_tract_ids_sha256": tract_id_hash(test_ids),
                "training_target_mean": train_mean,
                "model_metrics": score_predictions(
                    truth.iloc[test_indices].to_numpy(), test_predictions
                ),
                "training_mean_baseline_metrics": score_predictions(
                    truth.iloc[test_indices].to_numpy(), fold_baseline
                ),
                "selection": selection,
            }
        )
        print(f"  {model_name}/{target} spatial fold {fold + 1}/{outer_folds} complete", flush=True)

    if (fold_numbers < 0).any() or not np.isfinite(predictions).all():
        raise AssertionError("Outer folds did not produce exactly one prediction per tract.")
    result = data[["tract_id", "lat", "lon"]].copy()
    result["tract_id"] = result.tract_id.astype(str)
    result["y_true"] = truth.to_numpy()
    result["y_pred"] = predictions
    result["training_mean_prediction"] = baseline
    result["error"] = predictions - truth.to_numpy()
    result["outer_fold"] = fold_numbers
    result["spatial_block"] = groups
    model_metrics = score_predictions(truth.to_numpy(), predictions)
    baseline_metrics = score_predictions(truth.to_numpy(), baseline)
    baseline_mse = float(np.mean((truth.to_numpy() - baseline) ** 2))
    model_mse = float(np.mean((truth.to_numpy() - predictions) ** 2))
    details = {
        "evaluation": "nested_spatial_block_cv",
        "selection_code_sha256": hashlib.sha256(
            Path(__file__).with_name("model_selection.py").read_bytes()
        ).hexdigest(),
        "model": model_name,
        "target": target,
        "rows": len(data),
        "outer_folds": outer_folds,
        "inner_folds": inner_folds,
        "seed": seed,
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "grid": design,
        "model_metrics": model_metrics,
        "training_mean_baseline_metrics": baseline_metrics,
        "mse_skill_vs_training_mean": 1.0 - model_mse / baseline_mse if baseline_mse > 0 else None,
        "fold_r2_std": float(np.std([f["model_metrics"]["r2"] for f in folds])),
        "folds": folds,
        "limitations": [
            "Grid cells are held out intact; there is no geographic buffer between adjacent cells.",
            "This tests spatial blocks within one prepared city, not transfer to a new city or date.",
            "ACS monetary labels may be top-coded, and the ACS and OSM periods differ.",
        ],
    }
    return result, details


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", required=True, choices=["nyc", "chicago", "san_francisco"])
    parser.add_argument("--models", default="ridge,extra_trees,hist_gb,xgboost")
    parser.add_argument("--targets", default="all")
    parser.add_argument("--feature-set", default="osm_only", choices=["osm_only", "osm_311"])
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/spatial_validation"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--spatial-bins", type=int, default=4)
    parser.add_argument("--outer-folds", type=int, default=3)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--n-jobs", type=int, default=2)
    args = parser.parse_args()
    if min(args.outer_folds, args.inner_folds, args.spatial_bins) < 2:
        parser.error("Fold and grid counts must be at least two.")
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    targets = (
        TARGETS
        if args.targets.lower() == "all"
        else [item.strip() for item in args.targets.split(",") if item.strip()]
    )
    if not models or not targets or set(targets) - set(TARGETS):
        parser.error("Provide at least one model and supported target.")
    if args.feature_set == "osm_311" and args.city != "nyc":
        parser.error("OSM + 311 features are available only for NYC.")
    dataset_path = args.data_dir / f"{args.city}_dataset.parquet"
    enriched_path = args.data_dir / f"{args.city}_dataset_311.parquet"
    if args.feature_set == "osm_311" and enriched_path.is_file():
        dataset_path = enriched_path
    dataset_sha256 = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    data = load_dataset(args.city, args.feature_set, args.data_dir)
    if hashlib.sha256(dataset_path.read_bytes()).hexdigest() != dataset_sha256:
        raise RuntimeError("Input dataset changed while loading spatial validation data.")
    feature_names = select_features(data, args.feature_set)
    if not feature_names:
        parser.error("No predictor columns are available.")
    source_path = dataset_path.with_suffix(".source.json")
    source = json.loads(source_path.read_text()) if source_path.is_file() else {}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for target in targets:
        if target not in data or _valid_target(data[target], target).sum() < 2 * args.outer_folds:
            print(f"Skipping {target}: insufficient valid labels.", flush=True)
            continue
        for model in models:
            predictions, details = evaluate_model(
                data,
                feature_names,
                target,
                model,
                seed=args.seed,
                spatial_bins=args.spatial_bins,
                outer_folds=args.outer_folds,
                inner_folds=args.inner_folds,
                n_jobs=args.n_jobs,
            )
            if hashlib.sha256(dataset_path.read_bytes()).hexdigest() != dataset_sha256:
                raise RuntimeError(
                    "Input dataset changed during spatial validation; results were not saved."
                )
            packages = {
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit-learn": version("scikit-learn"),
            }
            if model == "xgboost":
                packages["xgboost"] = version("xgboost")
            details.update(
                {
                    "city": args.city,
                    "feature_set": args.feature_set,
                    "data_source": source,
                    "dataset_path": str(dataset_path),
                    "dataset_sha256": dataset_sha256,
                    "package_versions": packages,
                }
            )
            stem = f"{args.city}__{model}__{args.feature_set}__{target}"
            prediction_path = args.out_dir / f"{stem}.spatial.parquet"
            predictions.to_parquet(prediction_path, index=False)
            details["prediction_sha256"] = hashlib.sha256(prediction_path.read_bytes()).hexdigest()
            (args.out_dir / f"{stem}.spatial.json").write_text(
                json.dumps(details, indent=2, allow_nan=False) + "\n"
            )
            summaries.append(
                {
                    "city": args.city,
                    "model": model,
                    "feature_set": args.feature_set,
                    "target": target,
                    **details["model_metrics"],
                    **{
                        f"baseline_{key}": value
                        for key, value in details["training_mean_baseline_metrics"].items()
                    },
                    "mse_skill_vs_training_mean": details["mse_skill_vs_training_mean"],
                    "fold_r2_std": details["fold_r2_std"],
                    "n": details["rows"],
                }
            )
            summary = pd.DataFrame(summaries)
            summary.to_csv(args.out_dir / f"{args.city}_spatial_summary.csv", index=False)
            print(
                f"Saved {stem}: spatial pooled R2={details['model_metrics']['r2']:.3f}", flush=True
            )
    if hashlib.sha256(dataset_path.read_bytes()).hexdigest() != dataset_sha256:
        raise RuntimeError(
            "Input dataset changed during spatial validation; generated results need review."
        )
    if not summaries:
        parser.error("No spatial evaluations were generated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
