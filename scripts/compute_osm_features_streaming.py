#!/usr/bin/env python3
"""
Streaming OSM -> tract features (low-memory).

Reads GeoJSONSeq line-by-line and aggregates:
- poi_count
- poi_entropy (by amenity/shop/leisure/tourism tag)
- road_length_m (meters)

Usage:
  python3 scripts/compute_osm_features_streaming.py \
    --tracts data/raw/census/nyc_tracts_2021.gpkg \
    --roads data/raw/osm/derived/nyc/roads.geojsonseq \
    --pois  data/raw/osm/derived/nyc/pois.geojsonseq \
    --output data/features/nyc_geo_features.parquet
"""

import argparse
import json
import math
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import shape
from shapely.ops import transform


def entropy_from_counts(counts: dict) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = c / total
        ent -= p * math.log(p + 1e-12)
    return float(ent)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracts", required=True)
    ap.add_argument("--roads", required=True)
    ap.add_argument("--pois", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    tracts_path = Path(args.tracts)
    roads_path = Path(args.roads)
    pois_path = Path(args.pois)
    out_path = Path(args.output)

    if not tracts_path.exists():
        raise SystemExit(f"Missing tracts: {tracts_path}")
    if not roads_path.exists():
        raise SystemExit(f"Missing roads geojsonseq: {roads_path}")
    if not pois_path.exists():
        raise SystemExit(f"Missing pois geojsonseq: {pois_path}")

    # Load tracts, project for meter-accurate lengths
    tracts = gpd.read_file(tracts_path)
    if "tract_id" not in tracts.columns:
        # fallback
        if "GEOID" in tracts.columns:
            tracts["tract_id"] = tracts["GEOID"].astype(str)
        else:
            raise SystemExit("Tracts missing tract_id/GEOID column")

    tracts = tracts[["tract_id", "geometry"]].copy()
    tracts = tracts.to_crs("EPSG:3857")
    sindex = tracts.sindex

    # Accumulators
    road_len = dict((tid, 0.0) for tid in tracts["tract_id"].tolist())
    poi_total = dict((tid, 0) for tid in tracts["tract_id"].tolist())
    poi_by_cat = dict((tid, {}) for tid in tracts["tract_id"].tolist())

    def candidates(geom):
        # returns tract indices likely intersecting
        return list(sindex.intersection(geom.bounds))

    # --- Roads: accumulate length inside tract polygons ---
    with roads_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                feat = json.loads(line)
                geom = shape(feat["geometry"])
            except Exception:
                continue

            # Project geometry to meters
            geom = transform(lambda x, y, z=None: (x, y) if z is None else (x, y), geom)
            # geojson is lon/lat; project by reconstructing via GeoSeries
            gtmp = gpd.GeoSeries([geom], crs="EPSG:4326").to_crs("EPSG:3857").iloc[0]

            for idx in candidates(gtmp):
                poly = tracts.iloc[idx].geometry
                if not gtmp.intersects(poly):
                    continue
                inter = gtmp.intersection(poly)
                if inter.is_empty:
                    continue
                road_len[tracts.iloc[idx]["tract_id"]] += float(inter.length)

    # --- POIs: count points per tract and category entropy ---
    def poi_category(props: dict) -> str:
        # pick one tag type deterministically
        for k in ("amenity", "shop", "leisure", "tourism"):
            v = props.get(k)
            if v:
                return f"{k}:{v}"
        return "other"

    with pois_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                feat = json.loads(line)
                geom = shape(feat["geometry"])
                props = feat.get("properties", {}) or {}
            except Exception:
                continue

            gtmp = gpd.GeoSeries([geom], crs="EPSG:4326").to_crs("EPSG:3857").iloc[0]
            cat = poi_category(props)

            # point-in-polygon
            for idx in candidates(gtmp):
                poly = tracts.iloc[idx].geometry
                if not gtmp.within(poly):
                    continue
                tid = tracts.iloc[idx]["tract_id"]
                poi_total[tid] += 1
                poi_by_cat[tid][cat] = poi_by_cat[tid].get(cat, 0) + 1
                break  # a point belongs to one tract

    # Build output
    rows = []
    for tid in tracts["tract_id"].tolist():
        ent = entropy_from_counts(poi_by_cat[tid])
        rows.append(
            {
                "tract_id": tid,
                "poi_count": float(poi_total[tid]),
                "poi_entropy": float(ent),
                "road_length_m": float(road_len[tid]),
            }
        )

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    print(f"✅ Wrote OSM tract features: {out_path} (rows={len(df)}, cols={len(df.columns)})")
    print(
        f"   poi_count: min={df['poi_count'].min():.1f} max={df['poi_count'].max():.1f} "
        f"mean={df['poi_count'].mean():.2f}"
    )
    print(
        f"   road_length_m: min={df['road_length_m'].min():.1f} "
        f"max={df['road_length_m'].max():.1f} mean={df['road_length_m'].mean():.1f}"
    )


if __name__ == "__main__":
    main()
