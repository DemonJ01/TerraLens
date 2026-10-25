#!/usr/bin/env python3
"""
Build tract-level datasets for multiple cities using:
- TIGER tract geometries (.gpkg)
- ACS targets (multi-target) OR income-only fallback (.parquet)
- OSM-derived tract features (.parquet)

Outputs a single parquet per city with:
- city, tract_id, lat, lon, image_paths (empty legacy field)
- targets: median_income + any additional ACS targets (if available)
- geo features from OSM aggregation

Usage:
  python scripts/build_dataset_multi_city.py --city nyc --year 2021 \
    --output data/processed/nyc_dataset.parquet
"""

import argparse
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd


def load_tracts(path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    if "tract_id" not in gdf.columns:
        if "GEOID" in gdf.columns:
            gdf["tract_id"] = gdf["GEOID"].astype(str)
        else:
            raise SystemExit(f"❌ tract file missing GEOID/tract_id: {path}")
    gdf["tract_id"] = gdf["tract_id"].astype(str)
    return gdf


def load_income(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "tract_id" not in df.columns or "median_income" not in df.columns:
        raise SystemExit(f"❌ income parquet must have tract_id + median_income: {path}")
    df = df.copy()
    df["tract_id"] = df["tract_id"].astype(str)
    df["median_income"] = pd.to_numeric(df["median_income"], errors="coerce")
    df = df.dropna(subset=["median_income"])
    return df[["tract_id", "median_income"]]


def load_targets(path: Path) -> pd.DataFrame:
    """
    Load multi-target ACS labels parquet.
    Required: tract_id
    Others: any numeric target columns (median_income, poverty_rate, unemployment_rate, etc.)
    """
    df = pd.read_parquet(path).copy()
    if "tract_id" not in df.columns:
        raise SystemExit(f"❌ targets parquet must include tract_id: {path}")
    df["tract_id"] = df["tract_id"].astype(str)
    for c in df.columns:
        if c == "tract_id":
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # keep rows that have at least one non-null target
    target_cols = [c for c in df.columns if c != "tract_id"]
    if target_cols:
        df = df.dropna(subset=target_cols, how="all")
    return df


def pick_targets_path(city: str, year: int) -> Path | None:
    candidates = [
        Path(f"data/raw/census/{city}_acs_targets_{year}.parquet"),
        Path(f"data/raw/census/{city}_targets_{year}.parquet"),
        Path(f"data/raw/census/{city}_acs_targets.parquet"),
        Path(f"data/raw/census/{city}_targets.parquet"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True, choices=["nyc", "chicago", "san_francisco"])
    ap.add_argument("--year", type=int, default=2021)
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--geo-features",
        type=Path,
        help="Optional measured feature table; defaults to the city's prepared features",
    )
    ap.add_argument("--limit", type=int, default=0, help="Optional cap after merge (0 = no cap)")
    args = ap.parse_args()

    city = args.city
    year = args.year

    tracts_path = Path(f"data/raw/census/{city}_tracts_{year}.gpkg")
    income_path = Path(f"data/raw/census/{city}_income_{year}.parquet")
    geo_path = args.geo_features or Path(f"data/features/{city}_geo_features.parquet")

    for p in [tracts_path, geo_path]:
        if not p.exists():
            raise SystemExit(f"❌ Missing required input: {p}")

    tracts = load_tracts(tracts_path)

    geo = pd.read_parquet(geo_path).copy()
    if "tract_id" not in geo.columns:
        raise SystemExit(f"❌ geo features missing tract_id: {geo_path}")
    geo["tract_id"] = geo["tract_id"].astype(str)

    # Prefer multi-target ACS labels if present; else fall back to income-only
    targets_path = pick_targets_path(city, year)
    if targets_path is not None:
        targets = load_targets(targets_path)
        # Ensure median_income exists (some target packs might omit it)
        if "median_income" not in targets.columns:
            if not income_path.exists():
                raise SystemExit(f"❌ Missing income fallback: {income_path}")
            income = load_income(income_path)
            targets = targets.merge(income, on="tract_id", how="left", validate="one_to_one")
        print(f"✅ Using targets: {targets_path} (cols={len(targets.columns) - 1})")
    else:
        if not income_path.exists():
            raise SystemExit(f"❌ Missing ACS targets or income fallback: {income_path}")
        income = load_income(income_path)
        targets = income.copy()
        print("⚠️  No targets parquet found; using income-only labels.")

    # Merge labels into tracts
    gdf = tracts.merge(targets, on="tract_id", how="inner", validate="one_to_one")

    # Centroids for lat/lon
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)

    gdf_proj = gdf.to_crs("EPSG:3857")
    cent_proj = gdf_proj.geometry.centroid
    cent_ll = gpd.GeoSeries(cent_proj, crs="EPSG:3857").to_crs("EPSG:4326")
    gdf["lat"] = cent_ll.y
    gdf["lon"] = cent_ll.x

    # Collect target cols from merged gdf
    ignore = {"tract_id", "geometry", "lat", "lon"}
    target_cols = [c for c in gdf.columns if c not in ignore and c in targets.columns]

    if "median_income" not in target_cols:
        raise SystemExit("❌ median_income missing after merge (targets + income fallback failed).")

    base = {
        "city": city,
        "tract_id": gdf["tract_id"].astype(str),
        "lat": gdf["lat"].astype(float),
        "lon": gdf["lon"].astype(float),
        "image_paths": ["[]"] * len(gdf),  # Retained for compatibility with older datasets.
    }
    df = pd.DataFrame(base)

    # Add all targets
    for c in target_cols:
        df[c] = gdf[c]

    # Merge geo features
    df = df.merge(geo, on="tract_id", how="left", validate="one_to_one")
    feature_cols = [c for c in geo.columns if c != "tract_id" and c in df.columns]
    df[feature_cols] = df[feature_cols].fillna(0)

    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    osm_source_path = Path(f"data/raw/osm/{city}.osm.source.json")
    osm_source = {}
    if osm_source_path.is_file():
        try:
            osm_source = json.loads(osm_source_path.read_text())
        except (OSError, ValueError):
            pass
    feature_source_path = geo_path.with_suffix(".source.json")
    feature_source = {}
    if feature_source_path.is_file():
        try:
            feature_source = json.loads(feature_source_path.read_text())
        except (OSError, ValueError):
            pass
    out_path.with_suffix(".source.json").write_text(
        json.dumps(
            {
                "city": city,
                "acs_year": year,
                "acs_period": f"{year - 4}–{year}",
                "acs_source": str(targets_path or income_path),
                "osm_source": osm_source,
                "measured_features": feature_source,
                "rows": len(df),
                "caution": (
                    "ACS labels and OSM snapshot dates differ; "
                    "retrospective analysis, not a forecast."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"✅ Wrote dataset: {out_path} (rows={len(df)}, cols={len(df.columns)})")
    print(
        f"   median_income min/max: {df['median_income'].min():.0f} / "
        f"{df['median_income'].max():.0f}"
    )
    print(f"   target cols: {target_cols}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
