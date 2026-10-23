#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def build_geo_only_dataset(city: str, limit: int, seed: int) -> pd.DataFrame:
    """Build sample data with synthetic locations/features and optional Census labels."""
    rng = random.Random(seed)

    census_path = Path("data/raw/census") / f"{city}_income_2021.csv"

    if census_path.exists():
        census = pd.read_csv(census_path)
        if "tract_id" not in census.columns or "median_income" not in census.columns:
            raise ValueError(f"Census file missing required columns: {census_path}")

        census = census[["tract_id", "median_income"]].dropna()
        census = census.sample(n=min(limit, len(census)), random_state=seed).reset_index(drop=True)

        n = len(census)
        df = pd.DataFrame(
            {
                "city": city,
                "tract_id": census["tract_id"].astype(str),
                "median_income": census["median_income"].astype(float),
                # Synthetic locations and features for the training smoke check.
                "lat": np.random.default_rng(seed).uniform(40.63, 40.85, size=n),
                "lon": np.random.default_rng(seed + 1).uniform(-74.05, -73.75, size=n),
                "road_density": np.random.default_rng(seed + 2).normal(0.0, 1.0, size=n),
                "poi_density": np.random.default_rng(seed + 3).normal(0.0, 1.0, size=n),
                "landuse_entropy": np.random.default_rng(seed + 4).uniform(0.0, 1.0, size=n),
                # Retain the legacy image_paths schema for dataset compatibility.
                "image_paths": [json.dumps([])] * n,
            }
        )
        return df

    # Fallback dataset: no downloads required
    n = limit
    tract_ids = [f"{city}_dummy_{i:04d}" for i in range(n)]
    df = pd.DataFrame(
        {
            "city": city,
            "tract_id": tract_ids,
            "median_income": [rng.randint(30000, 150000) for _ in range(n)],
            "lat": np.random.default_rng(seed).uniform(40.63, 40.85, size=n),
            "lon": np.random.default_rng(seed + 1).uniform(-74.05, -73.75, size=n),
            "road_density": np.random.default_rng(seed + 2).normal(0.0, 1.0, size=n),
            "poi_density": np.random.default_rng(seed + 3).normal(0.0, 1.0, size=n),
            "landuse_entropy": np.random.default_rng(seed + 4).uniform(0.0, 1.0, size=n),
            "image_paths": [json.dumps([])] * n,
        }
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a TerraLens sample dataset with synthetic locations and features"
    )
    parser.add_argument("--city", required=True, help="City code (nyc, chicago, san_francisco)")
    parser.add_argument("--limit", type=int, default=300, help="Max tracts")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--geo-only", action="store_true", help="Compatibility flag; all features are geographic"
    )
    parser.add_argument("--output", required=True, help="Output parquet path")
    args = parser.parse_args()

    out = Path(args.output)
    _ensure_parent(out)

    df = build_geo_only_dataset(city=args.city, limit=args.limit, seed=args.seed)
    # Replace the synthetic feature columns when measured features exist.
    feat_path = Path("data/features") / f"{args.city}_geo_features.parquet"
    if feat_path.exists():
        try:
            geo = pd.read_parquet(feat_path)
            df = df.drop(
                columns=[
                    c for c in ["road_density", "poi_density", "landuse_entropy"] if c in df.columns
                ],
                errors="ignore",
            )
            df = df.merge(geo, on="tract_id", how="left")
            df = df.fillna({"poi_count_500m": 0.0, "poi_entropy": 0.0, "road_length_m_500m": 0.0})
            print(f"✅ Merged geo features from {feat_path}")
        except Exception as e:
            print(f"⚠️  Failed to merge geo features ({feat_path}): {e}")

    # Merge NYC 311 features when available.
    nyc311_path = Path("data/features") / f"{args.city}_311_features.parquet"
    if nyc311_path.exists():
        f311 = pd.read_parquet(nyc311_path)
        df = df.merge(f311, on="tract_id", how="left")
        # Fill NaNs for common 311 columns (safe even if some columns missing)
        fill_cols = [
            c
            for c in df.columns
            if c.startswith("complaint_") or c in ("complaints_total_density", "complaint_entropy")
        ]
        if fill_cols:
            df[fill_cols] = df[fill_cols].fillna(0.0)
        print(f"✅ Merged 311 features from {nyc311_path}")

    df.to_parquet(out, index=False)

    print(f"✅ Wrote sample dataset: {out} ({len(df)} rows, {len(df.columns)} cols)")
    print("Locations are synthetic. Use build_dataset_multi_city.py for real tract locations.")


if __name__ == "__main__":
    main()
