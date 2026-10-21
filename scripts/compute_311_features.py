#!/usr/bin/env python3
import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


def entropy(counts: np.ndarray) -> float:
    s = counts.sum()
    if s <= 0:
        return 0.0
    p = counts / s
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracts", default="data/raw/census/nyc_tracts_2021.gpkg")
    ap.add_argument("--nyc311", default="data/raw/nyc311/nyc311_last_12mo.parquet")
    ap.add_argument("--output", default="data/features/nyc_311_features.parquet")
    ap.add_argument(
        "--topk", type=int, default=8, help="Top complaint types to keep as separate densities"
    )
    args = ap.parse_args()

    tracts = gpd.read_file(args.tracts)
    if "tract_id" not in tracts:
        if "GEOID" not in tracts:
            raise SystemExit("Tracts must include tract_id or GEOID")
        tracts["tract_id"] = tracts["GEOID"].astype(str)
    tracts = tracts.to_crs(3857)
    tracts["tract_area_m2"] = tracts.geometry.area

    df = pd.read_parquet(args.nyc311)
    pts = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), crs=4326
    ).to_crs(3857)

    joined = gpd.sjoin(
        pts, tracts[["tract_id", "tract_area_m2", "geometry"]], how="inner", predicate="within"
    )
    # Total counts per tract
    total = joined.groupby("tract_id").size().rename("complaints_total").reset_index()

    # Top-k categories globally
    top_types = joined["complaint_type"].value_counts().head(args.topk).index.tolist()
    joined["complaint_type_top"] = joined["complaint_type"].where(
        joined["complaint_type"].isin(top_types), other="OTHER"
    )

    # Counts per tract per type
    pivot = (
        joined.groupby(["tract_id", "complaint_type_top"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    # Entropy per tract over types (including OTHER)
    type_cols = [c for c in pivot.columns if c != "tract_id"]
    ent = []
    for _, row in pivot.iterrows():
        ent.append(entropy(row[type_cols].to_numpy(dtype=float)))
    pivot["complaint_entropy"] = ent

    out = (
        tracts[["tract_id", "tract_area_m2"]]
        .merge(total, on="tract_id", how="left")
        .merge(pivot, on="tract_id", how="left")
    )
    out = out.fillna(0)

    # Density features
    out["complaints_total_density"] = out["complaints_total"] / out["tract_area_m2"]

    # Per-type densities
    for c in type_cols:
        out[f"complaint_{c.lower().replace(' ', '_').replace('/', '_')}_density"] = (
            out[c] / out["tract_area_m2"]
        )

    keep = ["tract_id", "complaints_total_density", "complaint_entropy"] + [
        col for col in out.columns if col.startswith("complaint_") and col.endswith("_density")
    ]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out[keep].replace([np.inf, -np.inf], np.nan).fillna(0).to_parquet(args.output, index=False)

    print(f"✅ Wrote 311 tract features -> {args.output} (rows={len(out)}, cols={len(keep)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
