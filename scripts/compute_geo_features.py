#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("compute_geo_features")


def _to_gdf_points(df: pd.DataFrame) -> gpd.GeoDataFrame:
    pts = [Point(xy) for xy in zip(df["lon"].astype(float), df["lat"].astype(float))]
    gdf = gpd.GeoDataFrame(df.copy(), geometry=pts, crs="EPSG:4326")
    return gdf


def _project_meters(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # Web mercator is fine for 500m buffers in a single-city demo
    return gdf.to_crs("EPSG:3857")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute OSM-based geo features around sample points"
    )
    parser.add_argument("--city", required=True)
    parser.add_argument(
        "--dataset", required=True, help="Input parquet produced by build_dataset.py"
    )
    parser.add_argument(
        "--osm_dir", default="data/raw/osm", help="Directory with *_pois.gpkg and *_roads.gpkg"
    )
    parser.add_argument("--buffer_m", type=int, default=500, help="Buffer radius in meters")
    parser.add_argument("--output", required=True, help="Output parquet (features)")
    args = parser.parse_args()

    df = pd.read_parquet(args.dataset)
    if not {"lat", "lon"}.issubset(df.columns):
        logger.error("Dataset missing lat/lon columns.")
        return 1

    pois_path = Path(args.osm_dir) / f"{args.city}_pois.gpkg"
    roads_path = Path(args.osm_dir) / f"{args.city}_roads.gpkg"

    if not pois_path.exists() or not roads_path.exists():
        logger.error("Missing OSM files. Run fetch_osm_features.py first.")
        logger.error(f"Expected: {pois_path} and {roads_path}")
        return 1

    logger.info("Loading dataset points...")
    pts = _to_gdf_points(df[["tract_id", "lat", "lon"]])

    logger.info("Loading POIs + roads...")
    gdf_pois = gpd.read_file(pois_path, layer="pois")
    gdf_edges = gpd.read_file(roads_path, layer="edges")

    # Project everything to meters
    pts_m = _project_meters(pts)
    pois_m = _project_meters(gdf_pois)
    edges_m = _project_meters(gdf_edges)

    # Prepare buffers
    buffers = pts_m.copy()
    buffers["geometry"] = buffers.geometry.buffer(args.buffer_m)

    # Spatial join POIs inside buffers
    logger.info("Computing POI counts...")
    join_p = gpd.sjoin(pois_m, buffers[["tract_id", "geometry"]], how="inner", predicate="within")
    # choose a POI category label
    cat_col = None
    for c in ["amenity", "shop", "leisure", "tourism"]:
        if c in join_p.columns:
            cat_col = c
            break
    if cat_col is None:
        join_p["poi_cat"] = "unknown"
        cat_col = "poi_cat"

    poi_counts = join_p.groupby("tract_id").size().rename("poi_count_500m").reset_index()
    poi_types = join_p.groupby(["tract_id", cat_col]).size().rename("n").reset_index()

    # Diversity proxy: normalized entropy of POI category distribution
    def entropy(group: pd.DataFrame) -> float:
        p = group["n"].to_numpy(dtype=float)
        p = p / p.sum()
        return float(-(p * np.log(p + 1e-12)).sum())

    ent = poi_types.groupby("tract_id").apply(entropy).rename("poi_entropy").reset_index()

    # Roads: road length inside buffer (approx)
    logger.info("Computing road length...")
    # Clip edges to buffers by intersecting (expensive but ok for 300 points)
    road_len = []
    edges_geom = edges_m[["geometry"]].copy()
    for tid, geom in zip(buffers["tract_id"], buffers["geometry"]):
        inter = edges_geom.geometry.intersection(geom)
        # intersection returns geometries; sum lengths
        length = float(np.nansum([g.length for g in inter if g is not None and not g.is_empty]))
        road_len.append((tid, length))
    road_df = pd.DataFrame(road_len, columns=["tract_id", "road_length_m_500m"])

    # Merge features
    out = df[["tract_id"]].copy()
    out = out.merge(poi_counts, on="tract_id", how="left")
    out = out.merge(ent, on="tract_id", how="left")
    out = out.merge(road_df, on="tract_id", how="left")
    out = out.fillna({"poi_count_500m": 0.0, "poi_entropy": 0.0, "road_length_m_500m": 0.0})

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)

    logger.info(f"✅ Wrote geo features -> {out_path} ({len(out)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
